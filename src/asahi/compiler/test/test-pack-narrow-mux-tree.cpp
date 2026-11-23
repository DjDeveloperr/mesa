/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_nir.h"
#include "agx_compiler.h"
}

static inline nir_builder
make_compute_shader(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

static inline nir_def *
make_buffer_addr(nir_builder *b, nir_def *base_u64, nir_def *byte_offset)
{
   return nir_iadd(b, base_u64, nir_u2u64(b, byte_offset));
}

static inline nir_def *
make_desc_base(nir_builder *b, int preamble_base)
{
   nir_def *set = nir_load_preamble(b, 1, 64, .base = preamble_base);
   nir_def *desc = nir_load_global_constant(
      b, 4, 32, set,
      .access = (gl_access_qualifier)ACCESS_CAN_SPECULATE,
      .align_mul = 16);
   return nir_pack_64_2x32_split(b, nir_channel(b, desc, 0),
                                 nir_channel(b, desc, 1));
}

static inline unsigned
count_intrinsics(nir_shader *shader, nir_intrinsic_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               n++;
         }
      }
   }
   return n;
}

static inline unsigned
count_alu_op(nir_shader *shader, nir_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               n++;
         }
      }
   }
   return n;
}

static inline unsigned
count_ifs(nir_shader *shader)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         if (block->cf_node.parent->type == nir_cf_node_if) {
            nir_if *nif = nir_cf_node_as_if(block->cf_node.parent);
            if (block == nir_if_first_then_block(nif))
               n++;
         }
      }
   }
   return n;
}

TEST(PackNarrowMuxTree, PackNarrowMuxTreeAndFoldIsub)
{
   for (int mode = 0; mode < 3; ++mode) {
      /* mode 0: contiguous layout (all 16 leaves loaded in preheader)
       * mode 1: strided layout (leaves 0,4,8,12 loaded in dominating block)
       * mode 2: unbounded index (no mask) -> must reject safely
       */
      nir_builder b = make_compute_shader("pack_mux_tree");
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out_base = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *center = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_def *cond = nir_ine_imm(&b, center, 0);

      nir_def *leaves[16] = {};
      if (mode == 1) {
         for (unsigned i = 0; i < 16; i += 4) {
            leaves[i] = nir_load_agx(&b, 1, 8, base, nir_imm_int(&b, i),
                                     .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                     .format = PIPE_FORMAT_R8_UINT);
         }
      }

      nir_if *guard_if = nir_push_if(&b, cond);
      for (unsigned i = 0; i < 16; ++i) {
         if (!leaves[i]) {
            leaves[i] = nir_load_agx(&b, 1, 8, base, nir_imm_int(&b, i),
                                     .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                     .format = PIPE_FORMAT_R8_UINT);
         }
      }

      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_variable *iv_var = nir_local_variable_create(b.impl, glsl_uint_type(), "iv");
      nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, iv_var, nir_imm_int(&b, 0), 1);

      nir_loop *loop = nir_push_loop(&b);
      nir_def *iv = nir_load_var(&b, iv_var);
      nir_if *brk = nir_push_if(&b, nir_ige_imm(&b, iv, 24));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, brk);

      nir_def *idx = (mode == 2) ? iv : nir_iand_imm(&b, iv, 15);
      /* Build 15-bcsel balanced tree over leaves[0..15] */
      auto build_tree = [&](auto &&self, unsigned lo, unsigned hi) -> nir_def * {
         if (hi - lo == 1)
            return leaves[lo];
         unsigned mid = (lo + hi) / 2;
         nir_def *l = self(self, lo, mid);
         nir_def *r = self(self, mid, hi);
         return nir_bcsel(&b, nir_ilt_imm(&b, idx, (int32_t)mid), l, r);
      };
      nir_def *picked = build_tree(build_tree, 0, 16);
      nir_def *diff = nir_iadd(&b, center, nir_ineg(&b, nir_u2u32(&b, picked)));
      nir_store_var(&b, acc_var, nir_iadd(&b, nir_load_var(&b, acc_var), diff), 1);
      nir_store_var(&b, iv_var, nir_iadd_imm(&b, iv, 1), 1);
      nir_pop_loop(&b, loop);

      nir_store_agx(&b, nir_load_var(&b, acc_var), out_base, nir_imm_int(&b, 0),
                    .format = PIPE_FORMAT_R32_UINT);
      nir_pop_if(&b, guard_if);

      nir_lower_vars_to_ssa(b.shader);
      nir_validate_shader(b.shader, "before pack_mux");

      bool progress = agx_nir_opt_pack_narrow_mux_tree(b.shader);
      nir_validate_shader(b.shader, "after pack_mux");
      nir_opt_dce(b.shader);

      unsigned ubfe_cnt = 0, isub_cnt = 0, bcsel8_cnt = 0;
      nir_foreach_block(blk, b.impl) {
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_ubitfield_extract)
               ubfe_cnt++;
            else if (alu->op == nir_op_isub)
               isub_cnt++;
            else if (alu->op == nir_op_bcsel && alu->def.bit_size == 8)
               bcsel8_cnt++;
         }
      }

      EXPECT_TRUE(progress);
      EXPECT_EQ(isub_cnt, 1u);
      if (mode == 2) {
         EXPECT_EQ(ubfe_cnt, 0u);
         EXPECT_EQ(bcsel8_cnt, 15u);
      } else {
         EXPECT_EQ(ubfe_cnt, 1u);
         EXPECT_EQ(bcsel8_cnt, 0u);
      }

      ralloc_free(b.shader);
   }
}

TEST(PackNarrowMuxTree, PackNarrowMuxTreePreNegateAndMaskPeel)
{
   /* Verify the exact 8-bit complement identity across all byte pairs. */
   for (int c = 0; c < 256; ++c) {
      for (int bval = 0; bval < 256; ++bval) {
         int inv_b = (~bval) & 0xff;
         int inv_c = 255 - c;
         ASSERT_EQ(inv_b - inv_c, c - bval);
      }
   }

   for (int neg_is_true = 0; neg_is_true < 2; ++neg_is_true) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "pack_mux_prenegate");
      nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 2);
      nir_def *inv_cond = nir_ilt_imm(&b, gid, 4);

      nir_def *c8 = nir_load_agx(&b, 1, 8, base, nir_imm_int(&b, 99),
                                 .access = (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_SPECULATE),
                                 .base = 0, .format = PIPE_FORMAT_R8_UINT,
                                 .sign_extend = false);
      nir_def *c32 = nir_u2u32(&b, c8);

      nir_def *v[8];
      for (unsigned i = 0; i < 8; i += 2) {
         v[i] = nir_load_agx(&b, 1, 8, base, nir_iadd_imm(&b, gid, i),
                             .access = (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_SPECULATE),
                             .base = 0, .format = PIPE_FORMAT_R8_UINT,
                             .sign_extend = false);
      }

      nir_variable *i_var = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_int_type(), "acc");
      nir_store_var(&b, i_var, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

      nir_push_if(&b, nir_ilt_imm(&b, gid, 1000));
      for (unsigned i = 1; i < 8; i += 2) {
         v[i] = nir_load_agx(&b, 1, 8, base, nir_iadd_imm(&b, gid, i),
                             .access = (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_SPECULATE),
                             .base = 0, .format = PIPE_FORMAT_R8_UINT,
                             .sign_extend = false);
      }

      nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, i_var);
         nir_push_if(&b, nir_ige_imm(&b, i, 24));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         nir_def *idx = nir_iand_imm(&b, i, 7);
         nir_def *s01 = nir_bcsel(&b, nir_ilt_imm(&b, idx, 1), v[0], v[1]);
         nir_def *s23 = nir_bcsel(&b, nir_ilt_imm(&b, idx, 3), v[2], v[3]);
         nir_def *s45 = nir_bcsel(&b, nir_ilt_imm(&b, idx, 5), v[4], v[5]);
         nir_def *s67 = nir_bcsel(&b, nir_ilt_imm(&b, idx, 7), v[6], v[7]);
         nir_def *s03 = nir_bcsel(&b, nir_ilt_imm(&b, idx, 2), s01, s23);
         nir_def *s47 = nir_bcsel(&b, nir_ilt_imm(&b, idx, 6), s45, s67);
         nir_def *mux = nir_bcsel(&b, nir_ilt_imm(&b, idx, 4), s03, s47);

         nir_def *b32 = nir_u2u32(&b, mux);
         nir_def *diff = nir_iadd(&b, c32, nir_ineg(&b, b32));
         nir_def *ndiff = nir_ineg(&b, diff);
         nir_def *signed_diff = neg_is_true ? nir_bcsel(&b, inv_cond, ndiff, diff)
                                            : nir_bcsel(&b, inv_cond, diff, ndiff);

         nir_store_var(&b, acc_var, nir_iadd(&b, nir_load_var(&b, acc_var), signed_diff), 1);
         nir_store_var(&b, i_var, nir_iadd_imm(&b, i, 1), 1);
      }
      nir_pop_loop(&b, NULL);
      nir_pop_if(&b, NULL);

      nir_store_agx(&b, nir_load_var(&b, acc_var), out, gid,
                    .access = ACCESS_NON_READABLE, .base = 0,
                    .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      NIR_PASS(_, b.shader, agx_nir_opt_pack_narrow_mux_tree);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "after pack mux prenegate");

      unsigned loop_ineg = 0, loop_iand7 = 0, loop_isub = 0, loop_ubfe = 0;
      nir_foreach_block(blk, b.impl) {
         bool in_loop = false;
         for (nir_cf_node *n = blk->cf_node.parent; n; n = n->parent)
            if (n->type == nir_cf_node_loop)
               in_loop = true;
         if (!in_loop)
            continue;
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_ineg)
               loop_ineg++;
            else if (alu->op == nir_op_isub)
               loop_isub++;
            else if (alu->op == nir_op_ubitfield_extract)
               loop_ubfe++;
            else if (alu->op == nir_op_iand) {
               for (unsigned s = 0; s < 2; ++s) {
                  nir_scalar sc = nir_scalar_resolved(alu->src[s].src.ssa, 0);
                  if (nir_scalar_is_const(sc) && nir_scalar_as_uint(sc) == 7)
                     loop_iand7++;
               }
            }
         }
      }
      EXPECT_EQ(loop_ubfe, 1u);
      EXPECT_EQ(loop_isub, 1u);
      EXPECT_EQ(loop_ineg, 0u);
      EXPECT_EQ(loop_iand7, 0u);
      ralloc_free(b.shader);
   }
}
