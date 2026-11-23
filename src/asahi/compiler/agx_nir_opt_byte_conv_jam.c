/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "util/format/u_format.h"
#include "util/hash_table.h"

/*
 * Unroll-and-jam (J = 4) for 1D narrow-integer (r8_uint) grid-stride
 * convolution loops on AGX3 (shared AGX3B + AGX3C).
 *
 * Matches a compute grid-stride X loop:
 *   for (x = gid.x + base_x; x < W; x += G) {
 *      [optional border check: copy 1 byte if border, else:]
 *      acc = 0.0f;
 *      for (k = 0; k < K; k += 1)
 *         acc = fma(u2f32(load_r8(src, start_base + x + k * step_k)),
 *                   load_r32(weights, k), acc);
 *      store_r8(post_op(acc), dst, row_base + x);
 *   }
 *
 * When invocations are independent (no barriers, atomics, shared memory, or
 * subgroup ops) and runtime guards prove base_x == 0, W >= 4, no index
 * overflow, and disjoint [dst] vs [src]/[weights] byte ranges, the X sweep
 * [0, W) is bijectively re-indexed as:
 *   1. Vectorized sweep: for (x4 = 4 * gid.x; x4 < (W & ~3); x4 += 4 * G)
 *      processing 4 adjacent pixels (x4 .. x4+3) per iteration with coalesced
 *      8x4 r8_uint loads and stores and a single shared uniform weight load
 *      per tap.
 *   2. Exact scalar tail: for (xt = (W & ~3) + gid.x; xt < W; xt += G)
 *      covering the remaining 0..3 pixels.
 *   3. Untouched original scalar loop in the fallback arm whenever any
 *      runtime guard fails.
 */

#define MAX_POST_OPS 8

static bool
is_inv_const_intrinsic(nir_intrinsic_instr *intr)
{
   return intr->intrinsic == nir_intrinsic_load_preamble ||
          intr->intrinsic == nir_intrinsic_load_constant_agx ||
          intr->intrinsic == nir_intrinsic_load_global_constant;
}

static bool
shader_has_inter_invocation_ops(nir_function_impl *impl)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_call)
            return true;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_store_agx:
            /* Idempotency: skip if already vectorized to 4-wide r8/r32 stores. */
            if (((nir_intrinsic_format(intr) == PIPE_FORMAT_R8_UINT &&
                  nir_src_bit_size(intr->src[0]) == 8) ||
                 (nir_intrinsic_format(intr) == PIPE_FORMAT_R32_UINT &&
                  nir_src_bit_size(intr->src[0]) == 32)) &&
                nir_src_num_components(intr->src[0]) == 4)
               return true;
            break;
         case nir_intrinsic_barrier:
         case nir_intrinsic_load_shared:
         case nir_intrinsic_store_shared:
         case nir_intrinsic_shared_atomic:
         case nir_intrinsic_shared_atomic_swap:
         case nir_intrinsic_global_atomic:
         case nir_intrinsic_global_atomic_swap:
         case nir_intrinsic_global_atomic_agx:
         case nir_intrinsic_global_atomic_swap_agx:
         case nir_intrinsic_ssbo_atomic:
         case nir_intrinsic_ssbo_atomic_swap:
         case nir_intrinsic_image_store:
         case nir_intrinsic_image_atomic:
         case nir_intrinsic_image_atomic_swap:
         case nir_intrinsic_bindless_image_store:
         case nir_intrinsic_bindless_image_atomic:
         case nir_intrinsic_bindless_image_atomic_swap:
         case nir_intrinsic_ballot:
         case nir_intrinsic_elect:
         case nir_intrinsic_vote_all:
         case nir_intrinsic_vote_any:
         case nir_intrinsic_vote_feq:
         case nir_intrinsic_vote_ieq:
         case nir_intrinsic_shuffle:
         case nir_intrinsic_shuffle_xor:
         case nir_intrinsic_shuffle_up:
         case nir_intrinsic_shuffle_down:
         case nir_intrinsic_quad_broadcast:
         case nir_intrinsic_quad_swap_horizontal:
         case nir_intrinsic_quad_swap_vertical:
         case nir_intrinsic_quad_swap_diagonal:
         case nir_intrinsic_reduce:
         case nir_intrinsic_inclusive_scan:
         case nir_intrinsic_exclusive_scan:
         case nir_intrinsic_read_invocation:
         case nir_intrinsic_read_first_invocation:
            return true;
         default:
            break;
         }
      }
   }
   return false;
}

static bool
def_dominates_loop(nir_def *def, nir_loop *loop)
{
   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
   return nir_block_dominates(nir_def_block(def), pre);
}

static bool
def_invariant_to_loop(nir_def *def, nir_loop *loop, unsigned depth)
{
   if (depth > 12 || !def)
      return false;
   if (def_dominates_loop(def, loop))
      return true;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const)
      return true;
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (!is_inv_const_intrinsic(intr))
         return false;
      unsigned n = nir_intrinsic_infos[intr->intrinsic].num_srcs;
      for (unsigned i = 0; i < n; ++i) {
         if (!def_invariant_to_loop(intr->src[i].ssa, loop, depth + 1))
            return false;
      }
      return true;
   }
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      unsigned n = nir_op_infos[alu->op].num_inputs;
      for (unsigned i = 0; i < n; ++i) {
         if (!def_invariant_to_loop(alu->src[i].src.ssa, loop, depth + 1))
            return false;
      }
      return true;
   }
   return false;
}

static nir_def *
materialize_inv(nir_builder *b, nir_def *def, nir_loop *loop, unsigned depth)
{
   assert(depth <= 12);
   if (def_dominates_loop(def, loop))
      return def;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const) {
      nir_load_const_instr *lc = nir_instr_as_load_const(instr);
      return nir_build_imm(b, def->num_components, def->bit_size, lc->value);
   }
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      assert(is_inv_const_intrinsic(intr));
      unsigned n = nir_intrinsic_infos[intr->intrinsic].num_srcs;
      nir_def *mat_srcs[4] = {NULL};
      for (unsigned i = 0; i < n; ++i)
         mat_srcs[i] = materialize_inv(b, intr->src[i].ssa, loop, depth + 1);
      nir_intrinsic_instr *clone =
         nir_instr_as_intrinsic(nir_instr_clone(b->shader, instr));
      nir_builder_instr_insert(b, &clone->instr);
      for (unsigned i = 0; i < n; ++i)
         nir_src_rewrite(&clone->src[i], mat_srcs[i]);
      return &clone->def;
   }
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   unsigned n = nir_op_infos[alu->op].num_inputs;
   nir_def *srcs[4] = {NULL};
   for (unsigned i = 0; i < n; ++i) {
      nir_def *mat = materialize_inv(b, alu->src[i].src.ssa, loop, depth + 1);
      if (mat->num_components > 1 && def->num_components == 1)
         mat = nir_channel(b, mat, alu->src[i].swizzle[0]);
      srcs[i] = mat;
   }
   return nir_build_alu_src_arr(b, alu->op, srcs);
}

static nir_def *
materialize_scalar_inv(nir_builder *b, nir_scalar s, nir_loop *loop)
{
   nir_def *d = materialize_inv(b, s.def, loop, 0);
   return d->num_components == 1 ? d : nir_channel(b, d, s.comp);
}

static bool
block_in_cf_node(nir_block *block, nir_cf_node *node)
{
   for (nir_cf_node *cur = &block->cf_node; cur != NULL; cur = cur->parent) {
      if (cur == node)
         return true;
   }
   return false;
}

static bool
loop_has_live_outs(nir_loop *loop)
{
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (!def)
            continue;
         nir_foreach_use_including_if(src, def) {
            nir_block *use_blk = nir_src_is_if(src)
               ? nir_cf_node_as_block(
                    nir_cf_node_prev(&nir_src_use_if(src)->cf_node))
               : nir_src_use_instr(src)->block;
            if (!block_in_cf_node(use_blk, &loop->cf_node))
               return true;
         }
      }
   }
   return false;
}

static nir_def *
chase_movs(nir_def *def)
{
   while (def && nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_def_as_alu(def);
      if (alu->op == nir_op_mov && def->num_components == 1 &&
          alu->src[0].src.ssa->num_components == 1) {
         def = alu->src[0].src.ssa;
         continue;
      }
      break;
   }
   return def;
}

static bool
is_gid_x(nir_def *def, nir_def **gid_x_out)
{
   nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(def, 0));
   if (nir_def_instr(s.def)->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(s.def);
   if (intr->intrinsic == nir_intrinsic_load_global_invocation_id &&
       s.comp == 0) {
      if (gid_x_out)
         *gid_x_out = def;
      return true;
   }
   return false;
}

static bool
match_gid_x_plus_base(nir_def *init, nir_loop *loop, nir_def **gid_x_out,
                      nir_def **base_x_out, nir_scalar *base_mul0_out,
                      nir_scalar *base_mul1_out)
{
   base_mul0_out->def = NULL;
   base_mul1_out->def = NULL;

   if (is_gid_x(init, gid_x_out)) {
      *base_x_out = NULL;
      return true;
   }

   nir_def *d = chase_movs(init);
   if (nir_def_instr(d)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_def_as_alu(d);
   if (alu->op == nir_op_iadd) {
      for (unsigned i = 0; i < 2; ++i) {
         nir_scalar sa = nir_scalar_chase_movs(
            nir_get_scalar(alu->src[i].src.ssa, alu->src[i].swizzle[0]));
         nir_def *b = alu->src[1 - i].src.ssa;
         if (nir_def_instr(sa.def)->type == nir_instr_type_intrinsic &&
             nir_def_as_intrinsic(sa.def)->intrinsic ==
                nir_intrinsic_load_global_invocation_id &&
             sa.comp == 0 && b->num_components == 1 && !b->divergent &&
             def_dominates_loop(b, loop)) {
            *gid_x_out = sa.def;
            *base_x_out = b;
            return true;
         }
      }
      return false;
   }

   if (alu->op == nir_op_imadshl_agx &&
       nir_src_is_const(alu->src[3].src) &&
       nir_src_as_uint(alu->src[3].src) == 0) {
      nir_scalar addend = nir_scalar_chase_movs(
         nir_get_scalar(alu->src[2].src.ssa, alu->src[2].swizzle[0]));
      nir_scalar m0 = nir_scalar_chase_movs(
         nir_get_scalar(alu->src[0].src.ssa, alu->src[0].swizzle[0]));
      nir_scalar m1 = nir_scalar_chase_movs(
         nir_get_scalar(alu->src[1].src.ssa, alu->src[1].swizzle[0]));
      if (nir_def_instr(addend.def)->type == nir_instr_type_intrinsic &&
          nir_def_as_intrinsic(addend.def)->intrinsic ==
             nir_intrinsic_load_global_invocation_id &&
          addend.comp == 0 && !m0.def->divergent && !m1.def->divergent &&
          def_dominates_loop(m0.def, loop) &&
          def_dominates_loop(m1.def, loop)) {
         *gid_x_out = addend.def;
         *base_x_out = NULL;
         *base_mul0_out = m0;
         *base_mul1_out = m1;
         return true;
      }
   }

   return false;
}

/* Match value == x_phi + x_loop_invariant_expr (supporting iadd, isub,
 * imadshl_agx with shift 0, and mov). Note that outer-loop variables such as
 * row coordinate `y` may be marked divergent when wg_size.y > 1 yet remain
 * strictly loop-invariant to x_loop. */
static bool
extract_affine_x(nir_def *val, nir_def *x_phi, nir_loop *x_loop,
                 nir_builder *b, nir_def **out_base, unsigned depth)
{
   if (depth > 8 || !val || val->num_components != 1 || val->bit_size != 32)
      return false;

   val = chase_movs(val);
   if (val == x_phi) {
      if (b && out_base)
         *out_base = nir_imm_int(b, 0);
      return true;
   }

   if (nir_def_instr(val)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_def_as_alu(val);

   if (alu->op == nir_op_iadd) {
      for (unsigned i = 0; i < 2; ++i) {
         nir_def *a = alu->src[i].src.ssa;
         nir_def *c = alu->src[1 - i].src.ssa;
         if (a->num_components != 1 || c->num_components != 1)
            continue;
         if (def_invariant_to_loop(c, x_loop, 0)) {
            nir_def *sub_base = NULL;
            if (extract_affine_x(a, x_phi, x_loop, b, &sub_base, depth + 1)) {
               if (b && out_base)
                  *out_base =
                     nir_iadd(b, sub_base, materialize_inv(b, c, x_loop, 0));
               return true;
            }
         }
      }
      return false;
   }

   if (alu->op == nir_op_isub) {
      nir_def *a = alu->src[0].src.ssa;
      nir_def *c = alu->src[1].src.ssa;
      if (a->num_components == 1 && c->num_components == 1 &&
          def_invariant_to_loop(c, x_loop, 0)) {
         nir_def *sub_base = NULL;
         if (extract_affine_x(a, x_phi, x_loop, b, &sub_base, depth + 1)) {
            if (b && out_base)
               *out_base =
                  nir_isub(b, sub_base, materialize_inv(b, c, x_loop, 0));
            return true;
         }
      }
      return false;
   }

   if (alu->op == nir_op_imadshl_agx) {
      nir_def *m0 = alu->src[0].src.ssa;
      nir_def *m1 = alu->src[1].src.ssa;
      nir_def *addend = alu->src[2].src.ssa;
      nir_def *shl = alu->src[3].src.ssa;
      if (m0->num_components == 1 && m1->num_components == 1 &&
          addend->num_components == 1 && shl->num_components == 1 &&
          nir_src_is_const(alu->src[3].src) &&
          nir_src_as_uint(alu->src[3].src) == 0 &&
          def_invariant_to_loop(m0, x_loop, 0) &&
          def_invariant_to_loop(m1, x_loop, 0)) {
         nir_def *sub_base = NULL;
         if (extract_affine_x(addend, x_phi, x_loop, b, &sub_base, depth + 1)) {
            if (b && out_base)
               *out_base = nir_imadshl_agx(b, materialize_inv(b, m0, x_loop, 0),
                                           materialize_inv(b, m1, x_loop, 0),
                                           sub_base, nir_imm_int(b, 0));
            return true;
         }
      }
      return false;
   }

   return false;
}

enum pix_f32_cvt {
   PIX_CVT_NONE = 0,
   PIX_CVT_U2F32,
   PIX_CVT_I2F32,
};

static bool
is_supported_pix_base(unsigned mem_base, unsigned pix_bits)
{
   if (pix_bits == 8)
      return mem_base == 0 || mem_base == AGX_NIR_MEMORY_INDEX_BYTES;
   return mem_base == 0;
}

static bool
is_pix_load_agx(nir_intrinsic_instr *intr, unsigned *bits_out,
                enum pipe_format *fmt_out)
{
   if (intr->intrinsic != nir_intrinsic_load_agx ||
       intr->def.num_components != 1 ||
       !(nir_intrinsic_access(intr) & ACCESS_NON_WRITEABLE) ||
       (nir_intrinsic_access(intr) & (ACCESS_COHERENT | ACCESS_VOLATILE)))
      return false;

   enum pipe_format fmt = nir_intrinsic_format(intr);
   unsigned bits = intr->def.bit_size;
   if (((bits == 8 && fmt == PIPE_FORMAT_R8_UINT) ||
        (bits == 32 && fmt == PIPE_FORMAT_R32_UINT)) &&
       is_supported_pix_base(nir_intrinsic_base(intr), bits)) {
      if (bits_out)
         *bits_out = bits;
      if (fmt_out)
         *fmt_out = fmt;
      return true;
   }
   return false;
}

static bool
is_pix_store_agx(nir_intrinsic_instr *intr, unsigned *bits_out,
                 enum pipe_format *fmt_out)
{
   if (intr->intrinsic != nir_intrinsic_store_agx ||
       nir_src_num_components(intr->src[0]) != 1 ||
       !(nir_intrinsic_access(intr) & ACCESS_NON_READABLE) ||
       (nir_intrinsic_access(intr) & (ACCESS_COHERENT | ACCESS_VOLATILE)))
      return false;

   enum pipe_format fmt = nir_intrinsic_format(intr);
   unsigned bits = nir_src_bit_size(intr->src[0]);
   if (((bits == 8 && fmt == PIPE_FORMAT_R8_UINT) ||
        (bits == 32 && fmt == PIPE_FORMAT_R32_UINT)) &&
       is_supported_pix_base(nir_intrinsic_base(intr), bits)) {
      if (bits_out)
         *bits_out = bits;
      if (fmt_out)
         *fmt_out = fmt;
      return true;
   }
   return false;
}

static bool
is_r32_weight_load_agx(nir_intrinsic_instr *intr)
{
   return intr->intrinsic == nir_intrinsic_load_agx &&
          intr->def.num_components == 1 && intr->def.bit_size == 32 &&
          nir_intrinsic_format(intr) == PIPE_FORMAT_R32_UINT &&
          (nir_intrinsic_access(intr) & ACCESS_NON_WRITEABLE) &&
          !(nir_intrinsic_access(intr) & ACCESS_COHERENT) &&
          !(nir_intrinsic_access(intr) & ACCESS_VOLATILE);
}

/* Match pixel-to-f32 conversion for r8_uint or r32_uint pixels:
 *   8-bit : u2f32(pix8), u2f32/i2f32(u2u32/u2u16(pix8)) -> PIX_CVT_U2F32
 *   32-bit: pix32 (direct f32) -> PIX_CVT_NONE, i2f32(pix32) -> PIX_CVT_I2F32,
 *           u2f32(pix32) -> PIX_CVT_U2F32. */
static bool
match_pix_to_f32(nir_def *val, nir_def *pix, unsigned pix_bits,
                 enum pix_f32_cvt *cvt_out)
{
   val = chase_movs(val);
   if (pix_bits == 32 && val == pix) {
      *cvt_out = PIX_CVT_NONE;
      return true;
   }
   if (nir_def_instr(val)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *cvt = nir_def_as_alu(val);
   if (pix_bits == 32) {
      if (cvt->op == nir_op_i2f32 && chase_movs(cvt->src[0].src.ssa) == pix) {
         *cvt_out = PIX_CVT_I2F32;
         return true;
      }
      if (cvt->op == nir_op_u2f32 && chase_movs(cvt->src[0].src.ssa) == pix) {
         *cvt_out = PIX_CVT_U2F32;
         return true;
      }
      return false;
   }
   if (cvt->op == nir_op_u2f32 && chase_movs(cvt->src[0].src.ssa) == pix) {
      *cvt_out = PIX_CVT_U2F32;
      return true;
   }
   if (cvt->op == nir_op_i2f32 || cvt->op == nir_op_u2f32) {
      nir_def *mid = chase_movs(cvt->src[0].src.ssa);
      if (nir_def_instr(mid)->type == nir_instr_type_alu) {
         nir_alu_instr *widen = nir_def_as_alu(mid);
         if ((widen->op == nir_op_u2u32 || widen->op == nir_op_u2u16) &&
             chase_movs(widen->src[0].src.ssa) == pix) {
            *cvt_out = PIX_CVT_U2F32;
            return true;
         }
      }
   }
   return false;
}

static bool
is_pure_post_op(nir_alu_instr *alu)
{
   switch (alu->op) {
   case nir_op_mov:
   case nir_op_f2i32:
   case nir_op_f2u32:
   case nir_op_f2i16:
   case nir_op_f2u16:
   case nir_op_f2i8:
   case nir_op_f2u8:
   case nir_op_u2u8:
   case nir_op_i2i8:
   case nir_op_u2u32:
   case nir_op_i2i32:
   case nir_op_fround_even:
   case nir_op_ffloor:
   case nir_op_ftrunc:
   case nir_op_fadd:
   case nir_op_fmul:
   case nir_op_ffma:
   case nir_op_fmin:
   case nir_op_fmax:
   case nir_op_fsat:
   case nir_op_iadd:
   case nir_op_isub:
   case nir_op_imin:
   case nir_op_imax:
   case nir_op_umin:
   case nir_op_umax:
      return true;
   default:
      return false;
   }
}

/* Match border_cond = (x < r_lo) || (x >= r_hi) expressed as ior or
 * bcsel(lt, true, ge). */
static bool
match_x_border_cond(nir_def *cond, nir_def *x_phi, nir_loop *x_loop,
                    nir_def **r_lo_out, nir_def **r_hi_out)
{
   cond = chase_movs(cond);
   if (nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_def_as_alu(cond);
   nir_def *a = NULL, *b = NULL;

   if (alu->op == nir_op_ior) {
      a = alu->src[0].src.ssa;
      b = alu->src[1].src.ssa;
   } else if (alu->op == nir_op_bcsel) {
      if (nir_src_is_const(alu->src[1].src) &&
          nir_src_as_bool(alu->src[1].src)) {
         a = alu->src[0].src.ssa;
         b = alu->src[2].src.ssa;
      } else {
         return false;
      }
   } else {
      return false;
   }

   nir_def *r_lo = NULL, *r_hi = NULL;
   nir_def *arms[2] = {chase_movs(a), chase_movs(b)};
   for (unsigned i = 0; i < 2; ++i) {
      if (nir_def_instr(arms[i])->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *cmp = nir_def_as_alu(arms[i]);
      if (cmp->op == nir_op_ilt &&
          chase_movs(cmp->src[0].src.ssa) == x_phi &&
          !cmp->src[1].src.ssa->divergent &&
          def_invariant_to_loop(cmp->src[1].src.ssa, x_loop, 0)) {
         r_lo = cmp->src[1].src.ssa;
      } else if (cmp->op == nir_op_ige &&
                 chase_movs(cmp->src[0].src.ssa) == x_phi &&
                 !cmp->src[1].src.ssa->divergent &&
                 def_invariant_to_loop(cmp->src[1].src.ssa, x_loop, 0)) {
         r_hi = cmp->src[1].src.ssa;
      } else {
         return false;
      }
   }

   if (!r_lo || !r_hi)
      return false;

   *r_lo_out = r_lo;
   *r_hi_out = r_hi;
   return true;
}

static nir_def *
phi_src_for_block(nir_phi_instr *phi, nir_block *pred)
{
   nir_foreach_phi_src(src, phi) {
      if (src->pred == pred)
         return src->src.ssa;
   }
   return NULL;
}

struct byte_conv_match {
   nir_loop *x_loop;
   nir_def *x_phi;
   nir_def *gid_x;
   nir_def *base_x; /* NULL if 0 or mul0*mul1 */
   nir_scalar base_mul0;
   nir_scalar base_mul1;
   nir_def *step_x; /* NULL if step_mul0 * step_mul1 */
   nir_scalar step_mul0;
   nir_scalar step_mul1;
   nir_scalar bound_x;
   bool x_is_unsigned;

   unsigned pix_bits;
   enum pipe_format pix_fmt;
   enum pix_f32_cvt pix_cvt;

   /* Optional border branch */
   bool has_border;
   bool border_is_x_dep;
   nir_def *border_cond; /* when !border_is_x_dep */
   nir_def *r_lo;        /* when border_is_x_dep */
   nir_def *r_hi;        /* when border_is_x_dep */
   nir_def *copy_idx;    /* affine in x_phi */
   unsigned copy_load_access;
   unsigned copy_load_base;
   bool copy_load_sign_extend;
   unsigned copy_store_access;
   unsigned copy_store_base;
   bool copy_store_sign_extend;

   /* Optional k_nonempty guard around k_loop */
   nir_def *k_guard; /* NULL if none */

   /* Inner 1D convolution loop */
   nir_def *k_bound;
   bool k_is_unsigned;
   nir_def *conv_start_idx; /* affine in x_phi */
   nir_def *conv_idx_step;  /* convergent, loop-invariant to x_loop */
   nir_def *src_ptr;
   nir_def *w_ptr;
   nir_def *dst_ptr;
   unsigned conv_load_access;
   unsigned conv_load_base;
   bool conv_load_sign_extend;
   unsigned w_load_access;
   unsigned w_load_base;
   bool w_load_sign_extend;
   unsigned store_access;
   unsigned store_base;
   bool store_sign_extend;
   nir_def *out_idx; /* affine in x_phi */

   /* Ordered chain of post-ops from acc -> out_byte */
   nir_alu_instr *post_ops[MAX_POST_OPS];
   unsigned num_post_ops;
   nir_def *acc_src;
};

static bool
collect_post_ops(nir_def *out_val, nir_def *acc_val, nir_loop *x_loop,
                 nir_alu_instr **ops_out, unsigned *num_ops_out)
{
   nir_alu_instr *rev[MAX_POST_OPS];
   unsigned n = 0;
   nir_def *cur = out_val;

   while (cur != acc_val) {
      if (n >= MAX_POST_OPS || nir_def_instr(cur)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *alu = nir_def_as_alu(cur);
      if (!is_pure_post_op(alu) || cur->num_components != 1)
         return false;

      unsigned num_srcs = nir_op_infos[alu->op].num_inputs;
      nir_def *next = NULL;
      for (unsigned i = 0; i < num_srcs; ++i) {
         nir_def *s = alu->src[i].src.ssa;
         if (s->num_components != 1)
            return false;
         if (s == acc_val || !def_invariant_to_loop(s, x_loop, 0)) {
            if (next != NULL)
               return false;
            next = s;
         }
      }
      if (!next)
         return false;
      rev[n++] = alu;
      cur = next;
   }

   for (unsigned i = 0; i < n; ++i)
      ops_out[i] = rev[n - 1 - i];
   *num_ops_out = n;
   return true;
}

static bool
match_inner_conv_loop(nir_loop *k_loop, nir_loop *x_loop, nir_def *x_phi,
                      struct byte_conv_match *m, nir_def **acc_out)
{
   if (nir_loop_has_continue_construct(k_loop))
      return false;

   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&k_loop->cf_node));
   nir_block *hdr = nir_loop_first_block(k_loop);
   nir_block *latch = nir_loop_last_block(k_loop);
   if (util_dynarray_num_elements(&hdr->predecessors, nir_block *) != 2)
      return false;

   /* Count phis in hdr: expect exactly 3 (k, idx, acc). */
   nir_phi_instr *phis[3];
   unsigned nphi = 0;
   nir_foreach_phi(phi, hdr) {
      if (nphi >= 3 || phi->def.num_components != 1 || phi->def.bit_size != 32)
         return false;
      phis[nphi++] = phi;
   }
   if (nphi != 3)
      return false;

   /* Must contain exactly one exit `if (...) break;` and no nested loops. */
   nir_if *exit_if = NULL;
   unsigned num_ifs = 0;
   foreach_list_typed(nir_cf_node, node, node, &k_loop->body) {
      if (node->type == nir_cf_node_loop)
         return false;
      if (node->type == nir_cf_node_if) {
         if (num_ifs++ > 0)
            return false;
         exit_if = nir_cf_node_as_if(node);
      }
   }
   if (!exit_if)
      return false;

   /* Check exit_if is `if (cond) { break; } else {}` with empty else. */
   nir_block *then_blk = nir_if_first_then_block(exit_if);
   nir_block *else_blk = nir_if_first_else_block(exit_if);
   if (then_blk != nir_if_last_then_block(exit_if) ||
       else_blk != nir_if_last_else_block(exit_if))
      return false;
   if (!exec_list_is_empty(&else_blk->instr_list))
      return false;
   nir_instr *last_then = nir_block_last_instr(then_blk);
   if (!last_then || last_then->type != nir_instr_type_jump ||
       nir_instr_as_jump(last_then)->type != nir_jump_break ||
       nir_block_first_instr(then_blk) != last_then)
      return false;

   /* Identify loads and FMA inside k_loop. Reject any stores/side-effects. */
   nir_intrinsic_instr *loads[2] = {NULL, NULL};
   unsigned nloads = 0;
   nir_alu_instr *fma_alu = NULL;
   unsigned instr_count = 0;

   nir_foreach_block_in_cf_node(blk, &k_loop->cf_node) {
      nir_foreach_instr(instr, blk) {
         if (instr->type == nir_instr_type_phi ||
             instr->type == nir_instr_type_jump ||
             instr->type == nir_instr_type_load_const)
            continue;
         if (++instr_count > 24)
            return false;
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (is_inv_const_intrinsic(intr)) {
               continue;
            } else if (is_pix_load_agx(intr, NULL, NULL) ||
                       is_r32_weight_load_agx(intr)) {
               if (nloads >= 2)
                  return false;
               loads[nloads++] = intr;
            } else {
               return false;
            }
         } else if (instr->type == nir_instr_type_alu) {
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_ffma) {
               if (fma_alu)
                  return false;
               fma_alu = alu;
            }
         } else {
            return false;
         }
      }
   }

   if (nloads != 2 || !fma_alu)
      return false;

   /* Identify acc_phi first via FMA accumulator input. */
   nir_phi_instr *acc_phi = NULL;
   for (unsigned i = 0; i < 3; ++i) {
      if (chase_movs(fma_alu->src[2].src.ssa) == &phis[i]->def)
         acc_phi = phis[i];
   }
   if (!acc_phi)
      return false;

   /* Identify k_phi (init == 0, step == +1) and idx_phi (remaining phi). */
   nir_phi_instr *k_phi = NULL, *idx_phi = NULL;
   nir_def *k_next = NULL;
   for (unsigned i = 0; i < 3; ++i) {
      if (phis[i] == acc_phi)
         continue;
      nir_def *init = phi_src_for_block(phis[i], pre);
      nir_def *next = phi_src_for_block(phis[i], latch);
      if (init && next &&
          nir_def_instr(init)->type == nir_instr_type_load_const &&
          nir_instr_as_load_const(nir_def_instr(init))->value[0].u32 == 0) {
         nir_def *cn = chase_movs(next);
         if (nir_def_instr(cn)->type == nir_instr_type_alu &&
             nir_def_as_alu(cn)->op == nir_op_iadd) {
            nir_alu_instr *sa = nir_def_as_alu(cn);
            bool ok = false;
            for (unsigned j = 0; j < 2; ++j) {
               if (chase_movs(sa->src[j].src.ssa) == &phis[i]->def &&
                   nir_src_is_const(sa->src[1 - j].src) &&
                   nir_src_as_uint(sa->src[1 - j].src) == 1)
                  ok = true;
            }
            if (ok && !k_phi) {
               k_phi = phis[i];
               k_next = cn;
               continue;
            }
         }
      }
      if (!idx_phi)
         idx_phi = phis[i];
   }
   if (!k_phi || !idx_phi)
      return false;

   /* Match loads[0..1] to pix_load (indexed by idx_phi) and w_load (k_phi). */
   nir_intrinsic_instr *pix_load = NULL, *w_load = NULL;
   for (unsigned i = 0; i < 2; ++i) {
      nir_def *off = chase_movs(loads[i]->src[1].ssa);
      if (off == &idx_phi->def)
         pix_load = loads[i];
      else if (off == &k_phi->def)
         w_load = loads[i];
   }
   unsigned pix_bits = 0;
   enum pipe_format pix_fmt = PIPE_FORMAT_NONE;
   if (!pix_load || !w_load ||
       !is_pix_load_agx(pix_load, &pix_bits, &pix_fmt) ||
       !is_r32_weight_load_agx(w_load))
      return false;
   if (m->pix_bits && (m->pix_bits != pix_bits || m->pix_fmt != pix_fmt))
      return false;

   nir_def *exit_cond = chase_movs(exit_if->condition.ssa);
   if (nir_def_instr(exit_cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *exit_cmp = nir_def_as_alu(exit_cond);
   bool k_unsigned = false;
   if (exit_cmp->op == nir_op_ige)
      k_unsigned = false;
   else if (exit_cmp->op == nir_op_uge)
      k_unsigned = true;
   else
      return false;

   nir_def *cmp_lhs = chase_movs(exit_cmp->src[0].src.ssa);
   nir_def *k_bound = exit_cmp->src[1].src.ssa;
   bool bottom_tested = false;
   if (cmp_lhs == &k_phi->def) {
      bottom_tested = false;
   } else if (cmp_lhs == k_next) {
      bottom_tested = true;
   } else {
      return false;
   }
   if (k_bound->num_components != 1 || k_bound->divergent ||
       !def_invariant_to_loop(k_bound, x_loop, 0))
      return false;

   /* Verify acc_phi: init == 0.0f, latch == fma_alu, fma computes
    * ffma(pix_to_f32(pix_load), w_load, acc_phi). */
   nir_def *acc_init = phi_src_for_block(acc_phi, pre);
   nir_def *acc_latch = phi_src_for_block(acc_phi, latch);
   if (!acc_init || !acc_latch ||
       nir_def_instr(acc_init)->type != nir_instr_type_load_const ||
       nir_instr_as_load_const(nir_def_instr(acc_init))->value[0].f32 != 0.0f)
      return false;
   if (chase_movs(acc_latch) != &fma_alu->def)
      return false;

   bool fma_ok = false;
   enum pix_f32_cvt pix_cvt = PIX_CVT_NONE;
   for (unsigned i = 0; i < 2; ++i) {
      nir_def *s0 = chase_movs(fma_alu->src[i].src.ssa);
      nir_def *s1 = chase_movs(fma_alu->src[1 - i].src.ssa);
      if (s1 == &w_load->def &&
          match_pix_to_f32(s0, &pix_load->def, pix_bits, &pix_cvt))
         fma_ok = true;
   }
   if (!fma_ok)
      return false;

   m->pix_bits = pix_bits;
   m->pix_fmt = pix_fmt;
   m->pix_cvt = pix_cvt;

   /* Verify idx_phi: init is affine in x_phi, latch is iadd(idx_phi, step)
    * where step is convergent and invariant to x_loop. */
   nir_def *idx_init = phi_src_for_block(idx_phi, pre);
   nir_def *idx_next = phi_src_for_block(idx_phi, latch);
   if (!idx_init || !idx_next ||
       !extract_affine_x(idx_init, x_phi, x_loop, NULL, NULL, 0))
      return false;

   idx_next = chase_movs(idx_next);
   if (nir_def_instr(idx_next)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *idx_step_alu = nir_def_as_alu(idx_next);
   if (idx_step_alu->op != nir_op_iadd)
      return false;
   nir_def *idx_step = NULL;
   for (unsigned i = 0; i < 2; ++i) {
      if (chase_movs(idx_step_alu->src[i].src.ssa) == &idx_phi->def) {
         nir_def *cand = idx_step_alu->src[1 - i].src.ssa;
         if (!cand->divergent && def_invariant_to_loop(cand, x_loop, 0))
            idx_step = cand;
      }
   }
   if (!idx_step)
      return false;

   /* Base pointers must be convergent and invariant to x_loop. */
   nir_def *src_ptr = pix_load->src[0].ssa;
   nir_def *w_ptr = w_load->src[0].ssa;
   if (src_ptr->divergent || w_ptr->divergent)
      return false;
   if (!def_invariant_to_loop(src_ptr, x_loop, 0) ||
       !def_invariant_to_loop(w_ptr, x_loop, 0))
      return false;

   m->k_bound = k_bound;
   m->k_is_unsigned = k_unsigned;
   m->conv_start_idx = idx_init;
   m->conv_idx_step = idx_step;
   m->src_ptr = src_ptr;
   m->w_ptr = w_ptr;
   m->conv_load_access = nir_intrinsic_access(pix_load);
   m->conv_load_base = nir_intrinsic_base(pix_load);
   m->conv_load_sign_extend = nir_intrinsic_sign_extend(pix_load);
   m->w_load_access = nir_intrinsic_access(w_load);
   m->w_load_base = nir_intrinsic_base(w_load);
   m->w_load_sign_extend = nir_intrinsic_sign_extend(w_load);
   *acc_out = bottom_tested ? &fma_alu->def : &acc_phi->def;
   return true;
}

static bool
match_copy_arm(struct exec_list *cf_list, nir_loop *x_loop, nir_def *x_phi,
               struct byte_conv_match *m)
{
   nir_block *blk = NULL;
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      if (node->type != nir_cf_node_block || blk != NULL)
         return false;
      blk = nir_cf_node_as_block(node);
   }
   if (!blk)
      return false;

   nir_intrinsic_instr *ld = NULL, *st = NULL;
   unsigned ld_bits = 0, st_bits = 0;
   enum pipe_format ld_fmt = PIPE_FORMAT_NONE, st_fmt = PIPE_FORMAT_NONE;
   nir_foreach_instr(instr, blk) {
      if (instr->type == nir_instr_type_alu ||
          instr->type == nir_instr_type_load_const)
         continue;
      if (instr->type != nir_instr_type_intrinsic)
         return false;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (is_inv_const_intrinsic(intr))
         continue;
      if (is_pix_load_agx(intr, &ld_bits, &ld_fmt)) {
         if (ld)
            return false;
         ld = intr;
      } else if (is_pix_store_agx(intr, &st_bits, &st_fmt)) {
         if (st)
            return false;
         st = intr;
      } else {
         return false;
      }
   }
   if (!ld || !st || ld_bits != st_bits || ld_fmt != st_fmt)
      return false;
   if (chase_movs(st->src[0].ssa) != &ld->def)
      return false;
   if (chase_movs(ld->src[1].ssa) != chase_movs(st->src[2].ssa))
      return false;
   if (!extract_affine_x(ld->src[1].ssa, x_phi, x_loop, NULL, NULL, 0))
      return false;

   m->pix_bits = ld_bits;
   m->pix_fmt = ld_fmt;
   m->copy_idx = ld->src[1].ssa;
   m->copy_load_access = nir_intrinsic_access(ld);
   m->copy_load_base = nir_intrinsic_base(ld);
   m->copy_load_sign_extend = nir_intrinsic_sign_extend(ld);
   m->copy_store_access = nir_intrinsic_access(st);
   m->copy_store_base = nir_intrinsic_base(st);
   m->copy_store_sign_extend = nir_intrinsic_sign_extend(st);
   return true;
}

static bool
match_conv_arm(struct exec_list *cf_list, nir_loop *x_loop, nir_def *x_phi,
               struct byte_conv_match *m)
{
   nir_loop *k_loop = NULL;
   nir_if *guard_if = NULL;
   nir_block *tail_blk = NULL;
   nir_intrinsic_instr *store = NULL;
   unsigned st_bits = 0;
   enum pipe_format st_fmt = PIPE_FORMAT_NONE;

   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      if (node->type == nir_cf_node_loop) {
         if (k_loop || guard_if)
            return false;
         k_loop = nir_cf_node_as_loop(node);
      } else if (node->type == nir_cf_node_if) {
         if (k_loop || guard_if)
            return false;
         guard_if = nir_cf_node_as_if(node);
         if (guard_if->condition.ssa->divergent ||
             !def_invariant_to_loop(guard_if->condition.ssa, x_loop, 0))
            return false;
         nir_block *eb = nir_if_first_else_block(guard_if);
         if (eb != nir_if_last_else_block(guard_if) ||
             !exec_list_is_empty(&eb->instr_list))
            return false;
         foreach_list_typed(nir_cf_node, gn, node, &guard_if->then_list) {
            if (gn->type == nir_cf_node_loop) {
               if (k_loop)
                  return false;
               k_loop = nir_cf_node_as_loop(gn);
            } else if (gn->type == nir_cf_node_if) {
               return false;
            }
         }
         m->k_guard = guard_if->condition.ssa;
      } else if (node->type == nir_cf_node_block) {
         nir_block *blk = nir_cf_node_as_block(node);
         nir_foreach_instr(instr, blk) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (is_pix_store_agx(intr, &st_bits, &st_fmt)) {
                  if (store)
                     return false;
                  store = intr;
                  tail_blk = blk;
               } else if (!is_inv_const_intrinsic(intr)) {
                  return false;
               }
            }
         }
      }
   }

   if (!k_loop || !store || !tail_blk)
      return false;

   nir_def *acc_val = NULL;
   if (!match_inner_conv_loop(k_loop, x_loop, x_phi, m, &acc_val))
      return false;
   if (st_bits != m->pix_bits || st_fmt != m->pix_fmt)
      return false;

   if (guard_if) {
      nir_def *merged_acc = NULL;
      nir_foreach_phi(phi, tail_blk) {
         if (merged_acc || phi->def.num_components != 1 ||
             phi->def.bit_size != 32)
            return false;
         nir_def *tval = phi_src_for_block(phi, nir_if_last_then_block(guard_if));
         nir_def *eval = phi_src_for_block(phi, nir_if_last_else_block(guard_if));
         if (!tval || !eval || chase_movs(tval) != acc_val)
            return false;
         if (nir_def_instr(eval)->type != nir_instr_type_load_const ||
             nir_instr_as_load_const(nir_def_instr(eval))->value[0].f32 != 0.0f)
            return false;
         merged_acc = &phi->def;
      }
      if (!merged_acc)
         return false;
      acc_val = merged_acc;
   }

   nir_def *dst_ptr = store->src[1].ssa;
   if (dst_ptr->divergent || !def_invariant_to_loop(dst_ptr, x_loop, 0))
      return false;
   if (!extract_affine_x(store->src[2].ssa, x_phi, x_loop, NULL, NULL, 0))
      return false;

   if (!collect_post_ops(store->src[0].ssa, acc_val, x_loop,
                         m->post_ops, &m->num_post_ops))
      return false;

   m->dst_ptr = dst_ptr;
   m->out_idx = store->src[2].ssa;
   m->store_access = nir_intrinsic_access(store);
   m->store_base = nir_intrinsic_base(store);
   m->store_sign_extend = nir_intrinsic_sign_extend(store);
   m->acc_src = acc_val;
   return true;
}

static bool
match_x_loop(nir_loop *x_loop, struct byte_conv_match *m)
{
   memset(m, 0, sizeof(*m));
   if (nir_loop_has_continue_construct(x_loop))
      return false;

   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&x_loop->cf_node));
   nir_block *hdr = nir_loop_first_block(x_loop);
   nir_block *latch = nir_loop_last_block(x_loop);
   if (util_dynarray_num_elements(&hdr->predecessors, nir_block *) != 2)
      return false;

   /* Header must have exactly one phi: x_phi. */
   nir_phi_instr *x_phi = NULL;
   nir_foreach_phi(phi, hdr) {
      if (x_phi || phi->def.num_components != 1 || phi->def.bit_size != 32)
         return false;
      x_phi = phi;
   }
   if (!x_phi)
      return false;

   if (loop_has_live_outs(x_loop))
      return false;

   nir_def *x_init = phi_src_for_block(x_phi, pre);
   nir_def *x_next = phi_src_for_block(x_phi, latch);
   if (!x_init || !x_next)
      return false;

   if (!match_gid_x_plus_base(x_init, x_loop, &m->gid_x, &m->base_x,
                              &m->base_mul0, &m->base_mul1))
      return false;

   x_next = chase_movs(x_next);
   if (nir_def_instr(x_next)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *x_step_alu = nir_def_as_alu(x_next);
   nir_def *step_x = NULL;
   nir_scalar step_mul0 = {0}, step_mul1 = {0};
   if (x_step_alu->op == nir_op_iadd) {
      for (unsigned i = 0; i < 2; ++i) {
         if (chase_movs(x_step_alu->src[i].src.ssa) == &x_phi->def) {
            nir_def *cand = x_step_alu->src[1 - i].src.ssa;
            if (!cand->divergent && def_invariant_to_loop(cand, x_loop, 0))
               step_x = cand;
         }
      }
   } else if (x_step_alu->op == nir_op_imadshl_agx &&
              nir_src_is_const(x_step_alu->src[3].src) &&
              nir_src_as_uint(x_step_alu->src[3].src) == 0 &&
              chase_movs(x_step_alu->src[2].src.ssa) == &x_phi->def) {
      nir_scalar m0 = nir_scalar_chase_movs(
         nir_get_scalar(x_step_alu->src[0].src.ssa,
                        x_step_alu->src[0].swizzle[0]));
      nir_scalar m1 = nir_scalar_chase_movs(
         nir_get_scalar(x_step_alu->src[1].src.ssa,
                        x_step_alu->src[1].swizzle[0]));
      if (!m0.def->divergent && !m1.def->divergent &&
          def_invariant_to_loop(m0.def, x_loop, 0) &&
          def_invariant_to_loop(m1.def, x_loop, 0)) {
         step_mul0 = m0;
         step_mul1 = m1;
      }
   }
   if (!step_x && !step_mul0.def)
      return false;

   /* First CF node after hdr must be top-tested exit `if (x >= W) break;`. */
   nir_cf_node *after_hdr = nir_cf_node_next(&hdr->cf_node);
   if (!after_hdr || after_hdr->type != nir_cf_node_if)
      return false;
   nir_if *exit_if = nir_cf_node_as_if(after_hdr);
   nir_block *etb = nir_if_first_then_block(exit_if);
   nir_block *eeb = nir_if_first_else_block(exit_if);
   if (etb != nir_if_last_then_block(exit_if) ||
       eeb != nir_if_last_else_block(exit_if) ||
       !exec_list_is_empty(&eeb->instr_list))
      return false;
   nir_instr *ebrk = nir_block_last_instr(etb);
   if (!ebrk || ebrk->type != nir_instr_type_jump ||
       nir_instr_as_jump(ebrk)->type != nir_jump_break ||
       nir_block_first_instr(etb) != ebrk)
      return false;

   nir_def *ec = chase_movs(exit_if->condition.ssa);
   if (nir_def_instr(ec)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *ecmp = nir_def_as_alu(ec);
   if (ecmp->op == nir_op_ige)
      m->x_is_unsigned = false;
   else if (ecmp->op == nir_op_uge)
      m->x_is_unsigned = true;
   else
      return false;
   if (chase_movs(ecmp->src[0].src.ssa) != &x_phi->def)
      return false;
   nir_scalar bound_x = nir_scalar_chase_movs(
      nir_get_scalar(ecmp->src[1].src.ssa, ecmp->src[1].swizzle[0]));
   if (bound_x.def->bit_size != 32 || bound_x.def->divergent ||
       !def_invariant_to_loop(bound_x.def, x_loop, 0))
      return false;

   m->x_loop = x_loop;
   m->x_phi = &x_phi->def;
   m->step_x = step_x;
   m->step_mul0 = step_mul0;
   m->step_mul1 = step_mul1;
   m->bound_x = bound_x;

   /* Inspect the body after exit_if: either a border `if (border) copy else conv`
    * or a direct conv loop. */
   nir_cf_node *body_node = nir_cf_node_next(&exit_if->cf_node);
   if (!body_node || body_node->type != nir_cf_node_block)
      return false;
   nir_cf_node *main_cf = nir_cf_node_next(body_node);
   if (!main_cf)
      return false;

   if (main_cf->type == nir_cf_node_if) {
      nir_if *bif = nir_cf_node_as_if(main_cf);
      if (nir_cf_node_next(main_cf) != &latch->cf_node)
         return false;
      nir_def *bcond = bif->condition.ssa;
      if (def_invariant_to_loop(bcond, x_loop, 0)) {
         m->has_border = true;
         m->border_is_x_dep = false;
         m->border_cond = bcond;
      } else if (match_x_border_cond(bcond, &x_phi->def, x_loop,
                                     &m->r_lo, &m->r_hi)) {
         m->has_border = true;
         m->border_is_x_dep = true;
      } else {
         return false;
      }

      if (!match_copy_arm(&bif->then_list, x_loop, &x_phi->def, m))
         return false;
      if (!match_conv_arm(&bif->else_list, x_loop, &x_phi->def, m))
         return false;
   } else if (main_cf->type == nir_cf_node_loop) {
      if (nir_cf_node_next(main_cf) != &latch->cf_node)
         return false;
      nir_def *acc_val = NULL;
      if (!match_inner_conv_loop(nir_cf_node_as_loop(main_cf), x_loop,
                                 &x_phi->def, m, &acc_val))
         return false;
      nir_intrinsic_instr *store = NULL;
      unsigned st_bits = 0;
      enum pipe_format st_fmt = PIPE_FORMAT_NONE;
      nir_foreach_instr(instr, latch) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (is_pix_store_agx(intr, &st_bits, &st_fmt)) {
               if (store)
                  return false;
               store = intr;
            } else if (!is_inv_const_intrinsic(intr)) {
               return false;
            }
         }
      }
      if (!store || st_bits != m->pix_bits || st_fmt != m->pix_fmt)
         return false;
      nir_def *dst_ptr = store->src[1].ssa;
      if (dst_ptr->divergent || !def_invariant_to_loop(dst_ptr, x_loop, 0))
         return false;
      if (!extract_affine_x(store->src[2].ssa, &x_phi->def, x_loop,
                            NULL, NULL, 0))
         return false;
      if (!collect_post_ops(store->src[0].ssa, acc_val, x_loop,
                            m->post_ops, &m->num_post_ops))
         return false;
      m->dst_ptr = dst_ptr;
      m->out_idx = store->src[2].ssa;
      m->store_access = nir_intrinsic_access(store);
      m->store_base = nir_intrinsic_base(store);
      m->store_sign_extend = nir_intrinsic_sign_extend(store);
      m->acc_src = acc_val;
   } else {
      return false;
   }

   return true;
}

static void
u64_add_u32(nir_builder *b, nir_def *base64, nir_def *off32,
            nir_def **out_lo, nir_def **out_hi)
{
   nir_def *lo = nir_unpack_64_2x32_split_x(b, base64);
   nir_def *hi = nir_unpack_64_2x32_split_y(b, base64);
   nir_def *sum_lo = nir_iadd(b, lo, off32);
   nir_def *carry = nir_bcsel(b, nir_ult(b, sum_lo, lo),
                              nir_imm_int(b, 1), nir_imm_int(b, 0));
   *out_lo = sum_lo;
   *out_hi = nir_iadd(b, hi, carry);
}

static nir_def *
u64_ule(nir_builder *b, nir_def *a_lo, nir_def *a_hi,
        nir_def *b_lo, nir_def *b_hi)
{
   return nir_ior(b, nir_ult(b, a_hi, b_hi),
                  nir_iand(b, nir_ieq(b, a_hi, b_hi), nir_uge(b, b_lo, a_lo)));
}

static nir_def *
u64_intervals_disjoint(nir_builder *b, nir_def *a_lo, nir_def *a_hi,
                       nir_def *a_end_lo, nir_def *a_end_hi,
                       nir_def *b_lo, nir_def *b_hi,
                       nir_def *b_end_lo, nir_def *b_end_hi)
{
   nir_def *a_ok = u64_ule(b, a_lo, a_hi, a_end_lo, a_end_hi);
   nir_def *b_ok = u64_ule(b, b_lo, b_hi, b_end_lo, b_end_hi);
   nir_def *disj = nir_ior(b, u64_ule(b, a_end_lo, a_end_hi, b_lo, b_hi),
                           u64_ule(b, b_end_lo, b_end_hi, a_lo, a_hi));
   return nir_iand(b, nir_iand(b, a_ok, b_ok), disj);
}

static nir_def *
emit_post_ops(nir_builder *b, const struct byte_conv_match *m, nir_def *acc)
{
   nir_def *cur = acc;
   nir_def *prev_orig = m->acc_src;

   for (unsigned i = 0; i < m->num_post_ops; ++i) {
      nir_alu_instr *orig = m->post_ops[i];
      unsigned nsrcs = nir_op_infos[orig->op].num_inputs;
      nir_def *srcs[4] = {NULL};
      for (unsigned s = 0; s < nsrcs; ++s) {
         nir_def *orig_s = orig->src[s].src.ssa;
         srcs[s] = (orig_s == prev_orig)
            ? cur
            : materialize_inv(b, orig_s, m->x_loop, 0);
      }
      cur = nir_build_alu_src_arr(b, orig->op, srcs);
      prev_orig = &orig->def;
   }
   return cur;
}

static void
add_latch_phi_src(nir_phi_instr *phi, nir_block *latch, nir_def *val)
{
   nir_phi_src *ps = nir_phi_instr_add_src(phi, latch, val);
   list_addtail(&ps->src.use_link, &val->uses);
}

static nir_def *
emit_pix_to_f32(nir_builder *b, enum pix_f32_cvt cvt, nir_def *pix)
{
   switch (cvt) {
   case PIX_CVT_NONE:
      return pix;
   case PIX_CVT_U2F32:
      return nir_u2f32(b, pix);
   case PIX_CVT_I2F32:
      return nir_i2f32(b, pix);
   default:
      UNREACHABLE("invalid pix_f32_cvt");
   }
}

/* Guard guarantees k_bound >= 1, so emit a bottom-tested k_loop (saving one
 * header branch per quad). */
static void
emit_jammed_conv_4(nir_builder *b, const struct byte_conv_match *m,
                   nir_def *src_ptr, nir_def *w_ptr, nir_def *dst_ptr,
                   nir_def *start_base, nir_def *out_base, nir_def *x4,
                   nir_def *k_bound, nir_def *idx_step)
{
   nir_def *start_idx4 = nir_iadd(b, start_base, x4);
   nir_def *out_idx4 = nir_iadd(b, out_base, x4);

   nir_def *zero_i = nir_imm_int(b, 0);
   nir_def *zero_f = nir_imm_float(b, 0.0f);

   nir_loop *k_loop = nir_push_loop(b);
   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&k_loop->cf_node));

   nir_phi_instr *k_phi = nir_phi_instr_create(b->shader);
   nir_phi_instr *idx_phi = nir_phi_instr_create(b->shader);
   nir_phi_instr *acc_phi[4];
   nir_def_init(&k_phi->instr, &k_phi->def, 1, 32);
   nir_def_init(&idx_phi->instr, &idx_phi->def, 1, 32);
   nir_phi_instr_add_src(k_phi, pre, zero_i);
   nir_phi_instr_add_src(idx_phi, pre, start_idx4);
   nir_builder_instr_insert(b, &k_phi->instr);
   nir_builder_instr_insert(b, &idx_phi->instr);

   for (unsigned c = 0; c < 4; ++c) {
      acc_phi[c] = nir_phi_instr_create(b->shader);
      nir_def_init(&acc_phi[c]->instr, &acc_phi[c]->def, 1, 32);
      nir_phi_instr_add_src(acc_phi[c], pre, zero_f);
      nir_builder_instr_insert(b, &acc_phi[c]->instr);
   }

   /* Coalesced 4-pixel load + single shared uniform weight load. */
   nir_def *pix4 = nir_load_agx(b, 4, m->pix_bits, src_ptr, &idx_phi->def,
                                .access = m->conv_load_access,
                                .base = m->conv_load_base,
                                .format = m->pix_fmt,
                                .sign_extend = m->conv_load_sign_extend);
   nir_def *w = nir_load_agx(b, 1, 32, w_ptr, &k_phi->def,
                             .access = m->w_load_access,
                             .base = m->w_load_base,
                             .format = PIPE_FORMAT_R32_UINT,
                             .sign_extend = m->w_load_sign_extend);

   nir_def *acc_next[4];
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *pf = emit_pix_to_f32(b, m->pix_cvt, nir_channel(b, pix4, c));
      acc_next[c] = nir_ffma(b, pf, w, &acc_phi[c]->def);
   }

   nir_def *k_next = nir_iadd_imm(b, &k_phi->def, 1);
   nir_def *exit_cond = m->k_is_unsigned
      ? nir_uge(b, k_next, k_bound)
      : nir_ige(b, k_next, k_bound);
   nir_if *eif = nir_push_if(b, exit_cond);
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, eif);

   nir_def *idx_next = nir_iadd(b, &idx_phi->def, idx_step);
   nir_block *latch = nir_loop_last_block(k_loop);

   add_latch_phi_src(k_phi, latch, k_next);
   add_latch_phi_src(idx_phi, latch, idx_next);
   for (unsigned c = 0; c < 4; ++c)
      add_latch_phi_src(acc_phi[c], latch, acc_next[c]);

   nir_pop_loop(b, k_loop);

   nir_def *out_vals[4];
   for (unsigned c = 0; c < 4; ++c)
      out_vals[c] = emit_post_ops(b, m, acc_next[c]);

   nir_def *out4 = nir_vec4(b, out_vals[0], out_vals[1],
                            out_vals[2], out_vals[3]);
   nir_store_agx(b, out4, dst_ptr, out_idx4,
                 .access = m->store_access,
                 .base = m->store_base,
                 .format = m->pix_fmt,
                 .sign_extend = m->store_sign_extend);
}

static void
emit_scalar_conv_body(nir_builder *b, const struct byte_conv_match *m,
                      nir_def *src_ptr, nir_def *w_ptr, nir_def *dst_ptr,
                      nir_def *start_base, nir_def *out_base, nir_def *x,
                      nir_def *k_bound, nir_def *idx_step)
{
   nir_def *start_idx = nir_iadd(b, start_base, x);
   nir_def *out_idx = nir_iadd(b, out_base, x);
   nir_def *zero_i = nir_imm_int(b, 0);
   nir_def *zero_f = nir_imm_float(b, 0.0f);

   nir_loop *kl = nir_push_loop(b);
   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&kl->cf_node));
   nir_phi_instr *k_phi = nir_phi_instr_create(b->shader);
   nir_phi_instr *idx_phi = nir_phi_instr_create(b->shader);
   nir_phi_instr *acc_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&k_phi->instr, &k_phi->def, 1, 32);
   nir_def_init(&idx_phi->instr, &idx_phi->def, 1, 32);
   nir_def_init(&acc_phi->instr, &acc_phi->def, 1, 32);
   nir_phi_instr_add_src(k_phi, pre, zero_i);
   nir_phi_instr_add_src(idx_phi, pre, start_idx);
   nir_phi_instr_add_src(acc_phi, pre, zero_f);
   nir_builder_instr_insert(b, &k_phi->instr);
   nir_builder_instr_insert(b, &idx_phi->instr);
   nir_builder_instr_insert(b, &acc_phi->instr);

   nir_def *pix = nir_load_agx(b, 1, m->pix_bits, src_ptr, &idx_phi->def,
                               .access = m->conv_load_access,
                               .base = m->conv_load_base,
                               .format = m->pix_fmt,
                               .sign_extend = m->conv_load_sign_extend);
   nir_def *w = nir_load_agx(b, 1, 32, w_ptr, &k_phi->def,
                             .access = m->w_load_access,
                             .base = m->w_load_base,
                             .format = PIPE_FORMAT_R32_UINT,
                             .sign_extend = m->w_load_sign_extend);
   nir_def *acc_n =
      nir_ffma(b, emit_pix_to_f32(b, m->pix_cvt, pix), w, &acc_phi->def);
   nir_def *k_n = nir_iadd_imm(b, &k_phi->def, 1);
   nir_def *ec = m->k_is_unsigned
      ? nir_uge(b, k_n, k_bound)
      : nir_ige(b, k_n, k_bound);
   nir_if *eif = nir_push_if(b, ec);
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, eif);

   nir_def *idx_n = nir_iadd(b, &idx_phi->def, idx_step);
   nir_block *latch = nir_loop_last_block(kl);
   add_latch_phi_src(k_phi, latch, k_n);
   add_latch_phi_src(idx_phi, latch, idx_n);
   add_latch_phi_src(acc_phi, latch, acc_n);
   nir_pop_loop(b, kl);

   nir_def *out_b = emit_post_ops(b, m, acc_n);
   nir_store_agx(b, out_b, dst_ptr, out_idx,
                 .access = m->store_access,
                 .base = m->store_base,
                 .format = m->pix_fmt,
                 .sign_extend = m->store_sign_extend);
}

static void
emit_scalar_pixel(nir_builder *b, const struct byte_conv_match *m,
                  nir_def *src_ptr, nir_def *w_ptr, nir_def *dst_ptr,
                  nir_def *start_base, nir_def *out_base, nir_def *copy_base,
                  nir_def *x, nir_def *k_bound, nir_def *idx_step,
                  nir_def *border_cond, nir_def *r_lo, nir_def *r_hi)
{
   if (m->has_border) {
      nir_def *bcond = m->border_is_x_dep
         ? nir_ior(b, nir_ilt(b, x, r_lo), nir_ige(b, x, r_hi))
         : border_cond;
      nir_if *bif = nir_push_if(b, bcond);
      nir_def *cidx = nir_iadd(b, copy_base, x);
      nir_def *v = nir_load_agx(b, 1, m->pix_bits, src_ptr, cidx,
                                .access = m->copy_load_access,
                                .base = m->copy_load_base,
                                .format = m->pix_fmt,
                                .sign_extend = m->copy_load_sign_extend);
      nir_store_agx(b, v, dst_ptr, cidx,
                    .access = m->copy_store_access,
                    .base = m->copy_store_base,
                    .format = m->pix_fmt,
                    .sign_extend = m->copy_store_sign_extend);
      nir_push_else(b, bif);
      emit_scalar_conv_body(b, m, src_ptr, w_ptr, dst_ptr, start_base, out_base,
                            x, k_bound, idx_step);
      nir_pop_if(b, bif);
      return;
   }

   emit_scalar_conv_body(b, m, src_ptr, w_ptr, dst_ptr, start_base, out_base,
                         x, k_bound, idx_step);
}

static void
transform_x_loop(nir_function_impl *impl, struct byte_conv_match *m)
{
   nir_builder b = nir_builder_at(nir_before_cf_node(&m->x_loop->cf_node));

   nir_def *gid_x = m->gid_x->num_components == 1
      ? m->gid_x
      : nir_channel(&b, m->gid_x, 0);
   nir_def *src_ptr = materialize_inv(&b, m->src_ptr, m->x_loop, 0);
   nir_def *w_ptr = materialize_inv(&b, m->w_ptr, m->x_loop, 0);
   nir_def *dst_ptr = materialize_inv(&b, m->dst_ptr, m->x_loop, 0);
   nir_def *bound_x = materialize_scalar_inv(&b, m->bound_x, m->x_loop);
   nir_def *step_x = m->step_x
      ? materialize_inv(&b, m->step_x, m->x_loop, 0)
      : nir_imul(&b, materialize_scalar_inv(&b, m->step_mul0, m->x_loop),
                 materialize_scalar_inv(&b, m->step_mul1, m->x_loop));
   nir_def *k_bound = materialize_inv(&b, m->k_bound, m->x_loop, 0);
   nir_def *idx_step = materialize_inv(&b, m->conv_idx_step, m->x_loop, 0);
   nir_def *border_cond = (m->has_border && !m->border_is_x_dep)
      ? materialize_inv(&b, m->border_cond, m->x_loop, 0)
      : NULL;
   nir_def *r_lo = (m->has_border && m->border_is_x_dep)
      ? materialize_inv(&b, m->r_lo, m->x_loop, 0)
      : NULL;
   nir_def *r_hi = (m->has_border && m->border_is_x_dep)
      ? materialize_inv(&b, m->r_hi, m->x_loop, 0)
      : NULL;

   nir_def *out_base = NULL, *start_base = NULL, *copy_base = NULL;
   bool ok = extract_affine_x(m->out_idx, m->x_phi, m->x_loop, &b, &out_base, 0);
   ok &= extract_affine_x(m->conv_start_idx, m->x_phi, m->x_loop, &b,
                          &start_base, 0);
   if (m->has_border)
      ok &= extract_affine_x(m->copy_idx, m->x_phi, m->x_loop, &b,
                             &copy_base, 0);
   assert(ok);
   (void)ok;

   /* Build runtime safety guard:
    * - base_x == 0 (when present)
    * - bound_x >= 4 and <= max_off
    * - step_x >= 1 and <= 0x10000000
    * - k_bound >= 1 and <= 0x1fffffff
    * - idx_step >= 0 and (k_bound - 1) * idx_step does not overflow
    * - disjoint 64-bit byte intervals for [dst] vs [src] and [dst] vs [weights]
    */
   const unsigned elem_shift = (m->pix_bits == 32) ? 2 : 0;
   const uint32_t max_off = 0x3fffffffu >> elem_shift;

   nir_def *guard = m->x_is_unsigned
      ? nir_iand(&b, nir_uge_imm(&b, bound_x, 4),
                 nir_uge(&b, nir_imm_int(&b, max_off), bound_x))
      : nir_iand(&b, nir_ige_imm(&b, bound_x, 4),
                 nir_ige(&b, nir_imm_int(&b, (int32_t)max_off), bound_x));

   if (m->base_x)
      guard = nir_iand(&b, guard, nir_ieq_imm(&b, m->base_x, 0));
   if (m->base_mul0.def && m->base_mul1.def) {
      nir_def *m0 = materialize_scalar_inv(&b, m->base_mul0, m->x_loop);
      nir_def *m1 = materialize_scalar_inv(&b, m->base_mul1, m->x_loop);
      guard = nir_iand(&b, guard,
                       nir_ior(&b, nir_ieq_imm(&b, m0, 0),
                               nir_ieq_imm(&b, m1, 0)));
   }

   guard = nir_iand(&b, guard,
                    nir_iand(&b, nir_ilt(&b, nir_imm_int(&b, 0), step_x),
                             nir_ige(&b, nir_imm_int(&b, 0x10000000), step_x)));
   guard = nir_iand(&b, guard,
                    nir_iand(&b, nir_ilt(&b, nir_imm_int(&b, 0), k_bound),
                             nir_ige(&b, nir_imm_int(&b, 0x1fffffff), k_bound)));
   guard = nir_iand(&b, guard, nir_ige_imm(&b, idx_step, 0));
   guard = nir_iand(&b, guard, nir_ige_imm(&b, out_base, 0));

   if (m->k_guard) {
      nir_def *kg = materialize_inv(&b, m->k_guard, m->x_loop, 0);
      guard = nir_iand(&b, guard, kg);
   }

   /* Dst byte interval: [dst_ptr + (out_base << elem_shift),
    *                     dst_ptr + ((out_base + bound_x) << elem_shift)). */
   nir_def *dst_lo, *dst_hi, *dst_end_lo, *dst_end_hi;
   nir_def *dst_end_off = nir_iadd(&b, out_base, bound_x);
   guard = nir_iand(&b, guard,
                    nir_iand(&b, nir_uge(&b, dst_end_off, out_base),
                             nir_uge(&b, nir_imm_int(&b, max_off), dst_end_off)));
   nir_def *dst_start_b =
      elem_shift ? nir_ishl_imm(&b, out_base, elem_shift) : out_base;
   nir_def *dst_end_b =
      elem_shift ? nir_ishl_imm(&b, dst_end_off, elem_shift) : dst_end_off;
   u64_add_u32(&b, dst_ptr, dst_start_b, &dst_lo, &dst_hi);
   u64_add_u32(&b, dst_ptr, dst_end_b, &dst_end_lo, &dst_end_hi);

   /* Weights byte interval: [w_ptr, w_ptr + 4 * k_bound). */
   nir_def *w_lo = nir_unpack_64_2x32_split_x(&b, w_ptr);
   nir_def *w_hi = nir_unpack_64_2x32_split_y(&b, w_ptr);
   nir_def *w_end_lo, *w_end_hi;
   u64_add_u32(&b, w_ptr, nir_ishl_imm(&b, k_bound, 2), &w_end_lo, &w_end_hi);
   guard = nir_iand(&b, guard,
                    u64_intervals_disjoint(&b, dst_lo, dst_hi,
                                           dst_end_lo, dst_end_hi,
                                           w_lo, w_hi, w_end_lo, w_end_hi));

   nir_def *km1 = nir_iadd_imm(&b, k_bound, -1);
   nir_def *k_span_hi = nir_umul_high(&b, km1, idx_step);
   nir_def *k_span = nir_imul(&b, km1, idx_step);
   nir_def *conv_span = nir_iadd(&b, k_span, bound_x);
   nir_def *conv_end_off = nir_iadd(&b, start_base, conv_span);
   nir_def *src_conv_lo, *src_conv_hi, *src_conv_end_lo, *src_conv_end_hi;
   nir_def *src_start_b =
      elem_shift ? nir_ishl_imm(&b, start_base, elem_shift) : start_base;
   nir_def *src_end_b =
      elem_shift ? nir_ishl_imm(&b, conv_end_off, elem_shift) : conv_end_off;
   u64_add_u32(&b, src_ptr, src_start_b, &src_conv_lo, &src_conv_hi);
   u64_add_u32(&b, src_ptr, src_end_b, &src_conv_end_lo, &src_conv_end_hi);

   nir_def *conv_non_overlap = nir_iand(
      &b,
      nir_iand(&b, nir_ieq_imm(&b, k_span_hi, 0),
               nir_iand(&b, nir_ige_imm(&b, start_base, 0),
                        nir_iand(&b, nir_uge(&b, conv_span, k_span),
                                 nir_iand(&b, nir_uge(&b, conv_end_off, start_base),
                                          nir_uge(&b, nir_imm_int(&b, max_off),
                                                  conv_end_off))))),
      u64_intervals_disjoint(&b, dst_lo, dst_hi, dst_end_lo, dst_end_hi,
                             src_conv_lo, src_conv_hi,
                             src_conv_end_lo, src_conv_end_hi));

   if (m->has_border) {
      nir_def *copy_end_off = nir_iadd(&b, copy_base, bound_x);
      nir_def *src_copy_lo, *src_copy_hi, *src_copy_end_lo, *src_copy_end_hi;
      nir_def *copy_start_b =
         elem_shift ? nir_ishl_imm(&b, copy_base, elem_shift) : copy_base;
      nir_def *copy_end_b =
         elem_shift ? nir_ishl_imm(&b, copy_end_off, elem_shift) : copy_end_off;
      u64_add_u32(&b, src_ptr, copy_start_b, &src_copy_lo, &src_copy_hi);
      u64_add_u32(&b, src_ptr, copy_end_b, &src_copy_end_lo, &src_copy_end_hi);
      nir_def *copy_non_overlap = nir_iand(
         &b,
         nir_iand(&b, nir_ige_imm(&b, copy_base, 0),
                  nir_iand(&b, nir_uge(&b, copy_end_off, copy_base),
                           nir_uge(&b, nir_imm_int(&b, max_off), copy_end_off))),
         u64_intervals_disjoint(&b, dst_lo, dst_hi, dst_end_lo, dst_end_hi,
                                src_copy_lo, src_copy_hi,
                                src_copy_end_lo, src_copy_end_hi));
      if (!m->border_is_x_dep) {
         guard = nir_iand(&b, guard,
                          nir_bcsel(&b, border_cond, copy_non_overlap,
                                    conv_non_overlap));
      } else {
         nir_def *h_lo_off = nir_iadd(&b, start_base, r_lo);
         nir_def *h_end_off = nir_iadd(&b, start_base, conv_span);
         nir_def *h_src_lo, *h_src_hi, *h_src_end_lo, *h_src_end_hi;
         nir_def *h_lo_b =
            elem_shift ? nir_ishl_imm(&b, h_lo_off, elem_shift) : h_lo_off;
         nir_def *h_end_b =
            elem_shift ? nir_ishl_imm(&b, h_end_off, elem_shift) : h_end_off;
         u64_add_u32(&b, src_ptr, h_lo_b, &h_src_lo, &h_src_hi);
         u64_add_u32(&b, src_ptr, h_end_b, &h_src_end_lo, &h_src_end_hi);
         nir_def *h_conv_ok = nir_iand(
            &b,
            nir_iand(&b, nir_ieq_imm(&b, k_span_hi, 0),
                     nir_iand(&b, nir_ige_imm(&b, h_lo_off, 0),
                              nir_iand(&b, nir_uge(&b, conv_span, k_span),
                                       nir_iand(&b, nir_uge(&b, h_end_off, h_lo_off),
                                                nir_uge(&b, nir_imm_int(&b, max_off),
                                                        h_end_off))))),
            u64_intervals_disjoint(&b, dst_lo, dst_hi, dst_end_lo, dst_end_hi,
                                   h_src_lo, h_src_hi,
                                   h_src_end_lo, h_src_end_hi));
         guard = nir_iand(&b, guard, nir_iand(&b, copy_non_overlap, h_conv_ok));
      }
   } else {
      guard = nir_iand(&b, guard, conv_non_overlap);
   }

   /* Guarded fast path vs untouched original loop fallback. */
   nir_if *fast_if = nir_push_if(&b, guard);

   nir_def *vec_bound = nir_iand_imm(&b, bound_x, ~3u);
   nir_def *vec_init = nir_ishl_imm(&b, gid_x, 2);
   nir_def *vec_step = nir_ishl_imm(&b, step_x, 2);

   /* 1. Vectorized 4-pixel-wide grid-stride X loop. */
   nir_loop *vloop = nir_push_loop(&b);
   nir_block *vpre = nir_cf_node_as_block(nir_cf_node_prev(&vloop->cf_node));
   nir_phi_instr *x4_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&x4_phi->instr, &x4_phi->def, 1, 32);
   nir_phi_instr_add_src(x4_phi, vpre, vec_init);
   nir_builder_instr_insert(&b, &x4_phi->instr);

   nir_def *vexit = m->x_is_unsigned
      ? nir_uge(&b, &x4_phi->def, vec_bound)
      : nir_ige(&b, &x4_phi->def, vec_bound);
   nir_if *veif = nir_push_if(&b, vexit);
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, veif);

   if (m->has_border && !m->border_is_x_dep) {
      nir_if *bif = nir_push_if(&b, border_cond);
      nir_def *cidx4 = nir_iadd(&b, copy_base, &x4_phi->def);
      nir_def *v4 = nir_load_agx(&b, 4, m->pix_bits, src_ptr, cidx4,
                                 .access = m->copy_load_access,
                                 .base = m->copy_load_base,
                                 .format = m->pix_fmt,
                                 .sign_extend = m->copy_load_sign_extend);
      nir_store_agx(&b, v4, dst_ptr, cidx4,
                    .access = m->copy_store_access,
                    .base = m->copy_store_base,
                    .format = m->pix_fmt,
                    .sign_extend = m->copy_store_sign_extend);
      nir_push_else(&b, bif);
      emit_jammed_conv_4(&b, m, src_ptr, w_ptr, dst_ptr,
                         start_base, out_base, &x4_phi->def, k_bound, idx_step);
      nir_pop_if(&b, bif);
   } else if (m->has_border && m->border_is_x_dep) {
      nir_def *x4_hi = nir_iadd_imm(&b, &x4_phi->def, 3);
      nir_def *all_interior = nir_iand(&b, nir_ige(&b, &x4_phi->def, r_lo),
                                       nir_ilt(&b, x4_hi, r_hi));
      nir_if *iif = nir_push_if(&b, all_interior);
      emit_jammed_conv_4(&b, m, src_ptr, w_ptr, dst_ptr,
                         start_base, out_base, &x4_phi->def, k_bound, idx_step);
      nir_push_else(&b, iif);

      nir_def *all_border = nir_ior(&b, nir_ilt(&b, x4_hi, r_lo),
                                    nir_ige(&b, &x4_phi->def, r_hi));
      nir_if *bif = nir_push_if(&b, all_border);
      nir_def *cidx4 = nir_iadd(&b, copy_base, &x4_phi->def);
      nir_def *v4 = nir_load_agx(&b, 4, m->pix_bits, src_ptr, cidx4,
                                 .access = m->copy_load_access,
                                 .base = m->copy_load_base,
                                 .format = m->pix_fmt,
                                 .sign_extend = m->copy_load_sign_extend);
      nir_store_agx(&b, v4, dst_ptr, cidx4,
                    .access = m->copy_store_access,
                    .base = m->copy_store_base,
                    .format = m->pix_fmt,
                    .sign_extend = m->copy_store_sign_extend);
      nir_push_else(&b, bif);

      /* Transition quad straddling the border: run exact 4-pixel scalar loop. */
      nir_def *zero_c = nir_imm_int(&b, 0);
      nir_loop *qloop = nir_push_loop(&b);
      nir_block *qpre = nir_cf_node_as_block(nir_cf_node_prev(&qloop->cf_node));
      nir_phi_instr *c_phi = nir_phi_instr_create(b.shader);
      nir_def_init(&c_phi->instr, &c_phi->def, 1, 32);
      nir_phi_instr_add_src(c_phi, qpre, zero_c);
      nir_builder_instr_insert(&b, &c_phi->instr);
      nir_if *qeif = nir_push_if(&b, nir_uge_imm(&b, &c_phi->def, 4));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, qeif);
      emit_scalar_pixel(&b, m, src_ptr, w_ptr, dst_ptr, start_base, out_base,
                        copy_base, nir_iadd(&b, &x4_phi->def, &c_phi->def),
                        k_bound, idx_step, border_cond, r_lo, r_hi);
      nir_def *c_next = nir_iadd_imm(&b, &c_phi->def, 1);
      add_latch_phi_src(c_phi, nir_loop_last_block(qloop), c_next);
      nir_pop_loop(&b, qloop);

      nir_pop_if(&b, bif);
      nir_pop_if(&b, iif);
   } else {
      emit_jammed_conv_4(&b, m, src_ptr, w_ptr, dst_ptr,
                         start_base, out_base, &x4_phi->def, k_bound, idx_step);
   }

   nir_def *x4_next = nir_iadd(&b, &x4_phi->def, vec_step);
   add_latch_phi_src(x4_phi, nir_loop_last_block(vloop), x4_next);
   nir_pop_loop(&b, vloop);

   /* 2. Exact scalar tail for [vec_bound, bound_x). */
   nir_if *tail_if = nir_push_if(&b, nir_ine(&b, vec_bound, bound_x));
   nir_def *tail_init = nir_iadd(&b, vec_bound, gid_x);
   nir_loop *tloop = nir_push_loop(&b);
   nir_block *tpre = nir_cf_node_as_block(nir_cf_node_prev(&tloop->cf_node));
   nir_phi_instr *xt_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&xt_phi->instr, &xt_phi->def, 1, 32);
   nir_phi_instr_add_src(xt_phi, tpre, tail_init);
   nir_builder_instr_insert(&b, &xt_phi->instr);

   nir_def *texit = m->x_is_unsigned
      ? nir_uge(&b, &xt_phi->def, bound_x)
      : nir_ige(&b, &xt_phi->def, bound_x);
   nir_if *teif = nir_push_if(&b, texit);
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, teif);

   emit_scalar_pixel(&b, m, src_ptr, w_ptr, dst_ptr, start_base, out_base,
                     copy_base, &xt_phi->def, k_bound, idx_step, border_cond,
                     r_lo, r_hi);
   nir_def *xt_next = nir_iadd(&b, &xt_phi->def, step_x);
   add_latch_phi_src(xt_phi, nir_loop_last_block(tloop), xt_next);
   nir_pop_loop(&b, tloop);
   nir_pop_if(&b, tail_if);

   nir_pop_if(&b, fast_if);

   /* 3. Move untouched original x_loop into fast_if's else arm. */
   nir_cf_list orig;
   nir_cf_extract(&orig, nir_before_cf_node(&m->x_loop->cf_node),
                  nir_after_cf_node(&m->x_loop->cf_node));
   nir_cf_reinsert(&orig, nir_after_cf_list(&fast_if->else_list));
   (void)impl;
}

static bool
scan_cf_list(nir_function_impl *impl, struct exec_list *cf_list)
{
   foreach_list_typed_safe(nir_cf_node, node, node, cf_list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (scan_cf_list(impl, &nif->then_list))
            return true;
         if (scan_cf_list(impl, &nif->else_list))
            return true;
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         struct byte_conv_match m;
         if (match_x_loop(loop, &m)) {
            transform_x_loop(impl, &m);
            return true;
         }
         if (scan_cf_list(impl, &loop->body))
            return true;
      }
   }
   return false;
}

bool
agx_nir_opt_byte_conv_jam(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE || nir->info.internal)
      return false;

   bool progress = false;
   nir_divergence_analysis(nir);

   nir_foreach_function_impl(impl, nir) {
      if (!impl->function->is_entrypoint ||
          shader_has_inter_invocation_ops(impl)) {
         nir_no_progress(impl);
         continue;
      }

      nir_metadata_require(impl, nir_metadata_dominance |
                                 nir_metadata_block_index);
      bool func_progress = scan_cf_list(impl, &impl->body);
      progress |= nir_progress(func_progress, impl, nir_metadata_none);
   }

   return progress;
}
