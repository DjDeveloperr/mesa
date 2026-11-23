/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>

#include "agx_compile.h"
#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"

static unsigned failures;

#define CHECK(cond)                                                            \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__,      \
                 #cond);                                                       \
         failures++;                                                           \
      }                                                                        \
   } while (0)

static void
add_latch_phi_src(nir_phi_instr *phi, nir_block *latch, nir_def *val)
{
   nir_phi_src *ps = nir_phi_instr_add_src(phi, latch, val);
   list_addtail(&ps->src.use_link, &val->uses);
}

struct conv_test_opts {
   bool horizontal_border;
   bool non_unit_x_stride;
   bool aliased_access;
   bool unsupported_pixel_r16;
   bool pixel_r32_i2f32;
   bool add_barrier;
   bool use_u2u32_i2f32;
   bool bottom_tested_k;
   bool sunk_invariants;
   bool wg_y_8_divergent_y;
   bool swizzled_bound_x;
   bool add_preamble_func;
   bool imadshl_x_step;
};

static nir_shader *
build_1d_byte_conv_shader(const struct conv_test_opts *opts)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "byte_conv_1d");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = opts->wg_y_8_divergent_y ? 8 : 1;
   b.shader->info.workgroup_size[2] = 1;

   if (opts->add_preamble_func) {
      nir_function *pfn = nir_function_create(b.shader, "preamble");
      pfn->is_preamble = true;
      nir_function_impl *pimpl = nir_function_impl_create(pfn);
      nir_builder pb = nir_builder_at(nir_before_impl(pimpl));
      nir_store_preamble(&pb, nir_imm_int(&pb, 1), .base = 4);
   }

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *gid_x = nir_channel(&b, gid, 0);
   nir_def *wg = nir_load_workgroup_id(&b);

   nir_def *src_ptr = nir_load_preamble(&b, 1, 64, .base = 8);
   nir_def *dst_ptr = nir_load_preamble(&b, 1, 64, .base = 12);
   nir_def *w_ptr = nir_load_preamble(&b, 1, 64, .base = 16);
   nir_def *width_vec = opts->swizzled_bound_x
      ? nir_load_preamble(&b, 2, 32, .base = 20)
      : NULL;
   nir_def *width = opts->swizzled_bound_x
      ? nir_channel(&b, width_vec, 0)
      : nir_load_preamble(&b, 1, 32, .base = 20);
   nir_def *stride = nir_load_preamble(&b, 1, 32, .base = 24);
   nir_def *ksize = nir_load_preamble(&b, 1, 32, .base = 26);
   nir_def *step_x = nir_load_preamble(&b, 1, 32, .base = 28);
   nir_def *radius = nir_load_preamble(&b, 1, 32, .base = 30);
   nir_def *border_hi = opts->sunk_invariants ? NULL : nir_isub(&b, width, radius);
   nir_def *height = nir_load_preamble(&b, 1, 32, .base = 32);
   nir_def *border_y = opts->sunk_invariants
      ? NULL
      : nir_ine_imm(&b, nir_load_preamble(&b, 1, 32, .base = 34), 0);

   if (opts->add_barrier)
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_shared);

   nir_def *y = opts->wg_y_8_divergent_y
      ? nir_channel(&b, gid, 1)
      : nir_channel(&b, wg, 1);
   nir_def *y_minus_r = opts->sunk_invariants ? NULL : nir_isub(&b, y, radius);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *zero_f = nir_imm_float(&b, 0.0f);

   nir_loop *x_loop = nir_push_loop(&b);
   nir_block *x_pre = nir_cf_node_as_block(nir_cf_node_prev(&x_loop->cf_node));
   nir_phi_instr *x_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&x_phi->instr, &x_phi->def, 1, 32);
   nir_phi_instr_add_src(x_phi, x_pre, gid_x);
   nir_builder_instr_insert(&b, &x_phi->instr);

   nir_if *x_exit = nir_push_if(&b, nir_ige(&b, &x_phi->def, width));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, x_exit);

   if (opts->sunk_invariants) {
      /* Mirror post-preamble NIR where loop-invariant border/offset ops live
       * inside x_loop. */
      nir_def *h_minus_r = nir_isub(&b, height, radius);
      border_y = nir_bcsel(&b, nir_ilt(&b, y, radius), nir_imm_true(&b),
                           nir_ige(&b, y, h_minus_r));
      border_hi = nir_isub(&b, width, radius);
      y_minus_r = nir_iadd(&b, y, nir_ineg(&b, radius));
   }

   nir_def *x_eff =
      opts->non_unit_x_stride ? nir_ishl_imm(&b, &x_phi->def, 1) : &x_phi->def;
   nir_def *row_idx = nir_imadshl_agx(&b, y, stride, x_eff, zero);

   unsigned ld_access = opts->aliased_access ? 0 : ACCESS_NON_WRITEABLE;
   unsigned st_access = opts->aliased_access ? 0 : ACCESS_NON_READABLE;
   unsigned pix_bits = opts->unsupported_pixel_r16 ? 16
                     : opts->pixel_r32_i2f32       ? 32
                                                   : 8;
   enum pipe_format pix_fmt = opts->unsupported_pixel_r16 ? PIPE_FORMAT_R16_UINT
                            : opts->pixel_r32_i2f32       ? PIPE_FORMAT_R32_UINT
                                                          : PIPE_FORMAT_R8_UINT;

   nir_def *bcond = opts->horizontal_border
      ? nir_bcsel(&b, nir_ilt(&b, &x_phi->def, radius), nir_imm_true(&b),
                  nir_ige(&b, &x_phi->def, border_hi))
      : border_y;
   nir_if *bif = nir_push_if(&b, bcond);

   nir_def *cpix = nir_load_agx(&b, 1, pix_bits, src_ptr, row_idx,
                                .access = ld_access, .base = 0,
                                .format = pix_fmt, .sign_extend = false);
   nir_store_agx(&b, cpix, dst_ptr, row_idx, .access = st_access, .base = 0,
                 .format = pix_fmt, .sign_extend = false);

   nir_push_else(&b, bif);

   nir_def *start_idx = opts->horizontal_border
      ? nir_iadd(&b, row_idx, nir_ineg(&b, radius))
      : nir_imadshl_agx(&b, y_minus_r, stride, x_eff, zero);
   nir_def *idx_step = opts->horizontal_border ? nir_imm_int(&b, 1) : stride;

   nir_def *acc_result = NULL;
   if (opts->bottom_tested_k) {
      nir_def *k_gt_0 = nir_ilt(&b, zero, ksize);
      nir_if *kg = nir_push_if(&b, k_gt_0);

      nir_loop *k_loop = nir_push_loop(&b);
      nir_block *k_pre = nir_cf_node_as_block(nir_cf_node_prev(&k_loop->cf_node));
      nir_phi_instr *k_phi = nir_phi_instr_create(b.shader);
      nir_phi_instr *idx_phi = nir_phi_instr_create(b.shader);
      nir_phi_instr *acc_phi = nir_phi_instr_create(b.shader);
      nir_def_init(&k_phi->instr, &k_phi->def, 1, 32);
      nir_def_init(&idx_phi->instr, &idx_phi->def, 1, 32);
      nir_def_init(&acc_phi->instr, &acc_phi->def, 1, 32);
      nir_phi_instr_add_src(k_phi, k_pre, zero);
      nir_phi_instr_add_src(idx_phi, k_pre, start_idx);
      nir_phi_instr_add_src(acc_phi, k_pre, zero_f);
      nir_builder_instr_insert(&b, &k_phi->instr);
      nir_builder_instr_insert(&b, &idx_phi->instr);
      nir_builder_instr_insert(&b, &acc_phi->instr);

      nir_def *pix = nir_load_agx(&b, 1, pix_bits, src_ptr, &idx_phi->def,
                                  .access = ld_access, .base = 0,
                                  .format = pix_fmt, .sign_extend = false);
      nir_def *w = nir_load_agx(&b, 1, 32, w_ptr, &k_phi->def,
                                .access = ACCESS_NON_WRITEABLE, .base = 0,
                                .format = PIPE_FORMAT_R32_UINT,
                                .sign_extend = false);
      nir_def *pix_f = opts->pixel_r32_i2f32
         ? nir_i2f32(&b, pix)
         : (opts->use_u2u32_i2f32 ? nir_i2f32(&b, nir_u2u32(&b, pix))
                                  : nir_u2f32(&b, pix));
      nir_def *acc_next = nir_ffma(&b, pix_f, w, &acc_phi->def);
      nir_def *k_next = nir_iadd_imm(&b, &k_phi->def, 1);
      nir_if *k_exit = nir_push_if(&b, nir_ige(&b, k_next, ksize));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, k_exit);

      nir_def *idx_next = nir_iadd(&b, &idx_phi->def, idx_step);
      nir_block *k_latch = nir_loop_last_block(k_loop);
      add_latch_phi_src(k_phi, k_latch, k_next);
      add_latch_phi_src(idx_phi, k_latch, idx_next);
      add_latch_phi_src(acc_phi, k_latch, acc_next);
      nir_pop_loop(&b, k_loop);
      nir_pop_if(&b, kg);

      nir_phi_instr *mphi = nir_phi_instr_create(b.shader);
      nir_def_init(&mphi->instr, &mphi->def, 1, 32);
      nir_phi_instr_add_src(mphi, nir_if_last_then_block(kg), acc_next);
      nir_phi_instr_add_src(mphi, nir_if_last_else_block(kg), zero_f);
      nir_builder_instr_insert(&b, &mphi->instr);
      acc_result = &mphi->def;
   } else {
      nir_loop *k_loop = nir_push_loop(&b);
      nir_block *k_pre = nir_cf_node_as_block(nir_cf_node_prev(&k_loop->cf_node));
      nir_phi_instr *k_phi = nir_phi_instr_create(b.shader);
      nir_phi_instr *idx_phi = nir_phi_instr_create(b.shader);
      nir_phi_instr *acc_phi = nir_phi_instr_create(b.shader);
      nir_def_init(&k_phi->instr, &k_phi->def, 1, 32);
      nir_def_init(&idx_phi->instr, &idx_phi->def, 1, 32);
      nir_def_init(&acc_phi->instr, &acc_phi->def, 1, 32);
      nir_phi_instr_add_src(k_phi, k_pre, zero);
      nir_phi_instr_add_src(idx_phi, k_pre, start_idx);
      nir_phi_instr_add_src(acc_phi, k_pre, zero_f);
      nir_builder_instr_insert(&b, &k_phi->instr);
      nir_builder_instr_insert(&b, &idx_phi->instr);
      nir_builder_instr_insert(&b, &acc_phi->instr);

      nir_if *k_exit = nir_push_if(&b, nir_ige(&b, &k_phi->def, ksize));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, k_exit);

      nir_def *pix = nir_load_agx(&b, 1, pix_bits, src_ptr, &idx_phi->def,
                                  .access = ld_access, .base = 0,
                                  .format = pix_fmt, .sign_extend = false);
      nir_def *w = nir_load_agx(&b, 1, 32, w_ptr, &k_phi->def,
                                .access = ACCESS_NON_WRITEABLE, .base = 0,
                                .format = PIPE_FORMAT_R32_UINT,
                                .sign_extend = false);
      nir_def *pix_f = opts->pixel_r32_i2f32
         ? nir_i2f32(&b, pix)
         : (opts->use_u2u32_i2f32 ? nir_i2f32(&b, nir_u2u32(&b, pix))
                                  : nir_u2f32(&b, pix));
      nir_def *acc_next = nir_ffma(&b, pix_f, w, &acc_phi->def);
      nir_def *k_next = nir_iadd_imm(&b, &k_phi->def, 1);
      nir_def *idx_next = nir_iadd(&b, &idx_phi->def, idx_step);
      nir_block *k_latch = nir_loop_last_block(k_loop);
      add_latch_phi_src(k_phi, k_latch, k_next);
      add_latch_phi_src(idx_phi, k_latch, idx_next);
      add_latch_phi_src(acc_phi, k_latch, acc_next);
      nir_pop_loop(&b, k_loop);
      acc_result = &acc_phi->def;
   }

   nir_def *out_i = nir_f2i32(&b, acc_result);
   nir_def *out_v = opts->unsupported_pixel_r16 ? nir_u2u16(&b, out_i)
                  : opts->pixel_r32_i2f32       ? out_i
                                                : nir_u2u8(&b, out_i);
   nir_store_agx(&b, out_v, dst_ptr, row_idx, .access = st_access, .base = 0,
                 .format = pix_fmt, .sign_extend = false);

   nir_pop_if(&b, bif);

   nir_def *x_next = opts->imadshl_x_step
      ? nir_imadshl_agx(&b, step_x, nir_imm_int(&b, 32), &x_phi->def, zero)
      : nir_iadd(&b, &x_phi->def, step_x);
   add_latch_phi_src(x_phi, nir_loop_last_block(x_loop), x_next);
   nir_pop_loop(&b, x_loop);

   nir_validate_shader(b.shader, "initial");
   return b.shader;
}

static void
count_vec4_pix_ops(nir_shader *s, enum pipe_format fmt, unsigned bits,
                   unsigned *loads4, unsigned *stores4)
{
   *loads4 = 0;
   *stores4 = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_agx &&
                nir_intrinsic_format(intr) == fmt &&
                intr->def.bit_size == bits && intr->def.num_components == 4)
               (*loads4)++;
            if (intr->intrinsic == nir_intrinsic_store_agx &&
                nir_intrinsic_format(intr) == fmt &&
                nir_src_bit_size(intr->src[0]) == bits &&
                nir_src_num_components(intr->src[0]) == 4)
               (*stores4)++;
         }
      }
   }
}

static void
count_vec4_r8_ops(nir_shader *s, unsigned *loads4, unsigned *stores4)
{
   count_vec4_pix_ops(s, PIPE_FORMAT_R8_UINT, 8, loads4, stores4);
}

static void
check_compiles_both_agx3(nir_shader *proto)
{
   const struct {
      enum agx_chip chip;
      enum agx_isa_encoding encoding;
   } targets[2] = {
      {AGX_CHIP_G17P, AGX_ISA_ENCODING_AGX3B},
      {AGX_CHIP_G17S, AGX_ISA_ENCODING_AGX3C},
   };

   for (unsigned t = 0; t < 2; ++t) {
      nir_shader *clone = nir_shader_clone(NULL, proto);
      clone->info.io_lowered = true;
      struct agx_shader_key key = {
         .dev.chip = targets[t].chip,
         .dev.encoding = targets[t].encoding,
         .dev.soft_fault = true,
         .reserved_preamble = 40,
         .no_preamble = true,
      };
      struct agx_shader_part part = {0};
      bool ok = agx_compile_shader_nir(clone, &key, &part);
      CHECK(ok);
      CHECK(part.info.binary_size > 0);
      CHECK(part.info.scratch_size == 0);
      CHECK(part.info.nr_gprs <= 63);
      free(part.binary);
      ralloc_free(clone);
   }
}

int
main(void)
{
   /* 1. Positive: vertical 1D r8_uint conv (X-invariant border). */
   {
      struct conv_test_opts opts = {0};
      nir_shader *s = build_1d_byte_conv_shader(&opts);
      CHECK(agx_nir_opt_byte_conv_jam(s));
      nir_validate_shader(s, "after vertical jam");
      unsigned l4 = 0, s4 = 0;
      count_vec4_r8_ops(s, &l4, &s4);
      CHECK(l4 >= 2);
      CHECK(s4 >= 2);
      check_compiles_both_agx3(s);
      ralloc_free(s);
   }

   /* 2. Positive: horizontal 1D r8_uint conv (X-dependent border). */
   {
      struct conv_test_opts opts = {.horizontal_border = true};
      nir_shader *s = build_1d_byte_conv_shader(&opts);
      CHECK(agx_nir_opt_byte_conv_jam(s));
      nir_validate_shader(s, "after horizontal jam");
      unsigned l4 = 0, s4 = 0;
      count_vec4_r8_ops(s, &l4, &s4);
      CHECK(l4 >= 2);
      CHECK(s4 >= 2);
      check_compiles_both_agx3(s);
      ralloc_free(s);
   }

   /* 3. Positive: post-preamble vertical shape (u2u32+i2f32, sunk invariants,
    * bottom-tested inner loop with k > 0 guard, and preamble function). */
   {
      struct conv_test_opts opts = {
         .use_u2u32_i2f32 = true,
         .bottom_tested_k = true,
         .sunk_invariants = true,
         .add_preamble_func = true,
      };
      nir_shader *s = build_1d_byte_conv_shader(&opts);
      CHECK(agx_nir_opt_byte_conv_jam(s));
      nir_validate_shader(s, "after vertical post-preamble jam");
      unsigned l4 = 0, s4 = 0;
      count_vec4_r8_ops(s, &l4, &s4);
      CHECK(l4 >= 2);
      CHECK(s4 >= 2);
      check_compiles_both_agx3(s);
      ralloc_free(s);
   }

   /* 4. Positive: post-preamble horizontal shape (wg=32x8x1 divergent y,
    * swizzled bound_x, sunk border_hi, u2u32+i2f32, bottom-tested k). */
   {
      struct conv_test_opts opts = {
         .horizontal_border = true,
         .use_u2u32_i2f32 = true,
         .bottom_tested_k = true,
         .sunk_invariants = true,
         .wg_y_8_divergent_y = true,
         .swizzled_bound_x = true,
         .add_preamble_func = true,
      };
      nir_shader *s = build_1d_byte_conv_shader(&opts);
      CHECK(agx_nir_opt_byte_conv_jam(s));
      nir_validate_shader(s, "after horizontal post-preamble jam");
      unsigned l4 = 0, s4 = 0;
      count_vec4_r8_ops(s, &l4, &s4);
      CHECK(l4 >= 2);
      CHECK(s4 >= 2);
      check_compiles_both_agx3(s);
      ralloc_free(s);
   }

   /* 3. Negative: non-unit X stride (2 * x) must be rejected. */
   {
      struct conv_test_opts opts = {.non_unit_x_stride = true};
      nir_shader *s = build_1d_byte_conv_shader(&opts);
      CHECK(!agx_nir_opt_byte_conv_jam(s));
      ralloc_free(s);
   }

   /* 4. Negative: missing readonly/writeonly access qualifiers rejected. */
   {
      struct conv_test_opts opts = {.aliased_access = true};
      nir_shader *s = build_1d_byte_conv_shader(&opts);
      CHECK(!agx_nir_opt_byte_conv_jam(s));
      ralloc_free(s);
   }

   /* 5. Positive: vertical 1D r32_uint i2f32 conv with imadshl_agx x_next. */
   {
      struct conv_test_opts opts = {
         .pixel_r32_i2f32 = true,
         .bottom_tested_k = true,
         .sunk_invariants = true,
         .imadshl_x_step = true,
         .add_preamble_func = true,
      };
      nir_shader *s = build_1d_byte_conv_shader(&opts);
      CHECK(agx_nir_opt_byte_conv_jam(s));
      nir_validate_shader(s, "after vertical r32 jam");
      unsigned l4 = 0, s4 = 0;
      count_vec4_pix_ops(s, PIPE_FORMAT_R32_UINT, 32, &l4, &s4);
      CHECK(l4 >= 2);
      CHECK(s4 >= 2);
      check_compiles_both_agx3(s);
      ralloc_free(s);
   }

   /* 6. Negative: unsupported 16-bit r16_uint pixel format rejected. */
   {
      struct conv_test_opts opts = {.unsupported_pixel_r16 = true};
      nir_shader *s = build_1d_byte_conv_shader(&opts);
      CHECK(!agx_nir_opt_byte_conv_jam(s));
      ralloc_free(s);
   }

   /* 6. Negative: workgroup barrier prevents grid-stride X re-indexing. */
   {
      struct conv_test_opts opts = {.add_barrier = true};
      nir_shader *s = build_1d_byte_conv_shader(&opts);
      CHECK(!agx_nir_opt_byte_conv_jam(s));
      ralloc_free(s);
   }

   return failures ? 1 : 0;
}
