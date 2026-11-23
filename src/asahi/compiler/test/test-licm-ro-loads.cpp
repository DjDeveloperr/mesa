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

TEST(LicmRoLoads, PureReadonlyLoopLicmHoist)
{
   /* mode 0: pure read-only loop with header load_agx (count) and loop-invariant
    *         body load_agx (desc_a) -> Phase 1 hoists count before loop,
    *         Phase 2 wraps loop in zero-speculation guard and hoists desc_a.
    * mode 1: loop contains store_agx -> neither load hoisted.
    * mode 2: loop contains barrier -> neither load hoisted.
    * mode 3: load_agx is ACCESS_COHERENT -> neither load hoisted.
    * mode 4: budget too tight (budget = 4 halves) -> body 32x4 load not hoisted. */
   for (int mode = 0; mode < 5; ++mode) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "ro_loop_licm");
      nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *p_cnt = nir_load_preamble(&b, 2, 32, .base = 0);
      nir_def *a_cnt = nir_pack_64_2x32_split(&b, nir_channel(&b, p_cnt, 0),
                                              nir_channel(&b, p_cnt, 1));
      nir_def *p_a = nir_load_preamble(&b, 2, 32, .base = 4);
      nir_def *a_a = nir_pack_64_2x32_split(&b, nir_channel(&b, p_a, 0),
                                            nir_channel(&b, p_a, 1));
      nir_def *p_b = nir_load_preamble(&b, 2, 32, .base = 8);
      nir_def *a_b = nir_pack_64_2x32_split(&b, nir_channel(&b, p_b, 0),
                                            nir_channel(&b, p_b, 1));
      nir_def *p_o = nir_load_preamble(&b, 2, 32, .base = 12);
      nir_def *a_o = nir_pack_64_2x32_split(&b, nir_channel(&b, p_o, 0),
                                            nir_channel(&b, p_o, 1));

      gl_access_qualifier acc =
         mode == 3 ? (gl_access_qualifier)(ACCESS_COHERENT | ACCESS_CAN_SPECULATE)
                   : (gl_access_qualifier)ACCESS_CAN_SPECULATE;

      nir_variable *i_var = nir_local_variable_create(b.impl, glsl_int_type(), "i");
      nir_variable *sum_var = nir_local_variable_create(b.impl, glsl_int_type(), "sum");
      nir_store_var(&b, i_var, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, sum_var, nir_imm_int(&b, 0), 1);

      nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, i_var);
         nir_def *cnt = nir_load_agx(&b, 1, 32, a_cnt, nir_imm_int(&b, 0),
                                     .access = acc, .base = 0,
                                     .format = PIPE_FORMAT_R32_UINT,
                                     .sign_extend = false);
         nir_push_if(&b, nir_ige(&b, i, cnt));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         if (mode == 2) {
            nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                        .memory_scope = SCOPE_WORKGROUP,
                        .memory_semantics = NIR_MEMORY_ACQ_REL,
                        .memory_modes = nir_var_mem_ssbo);
         }

         nir_def *va = nir_load_agx(&b, 4, 32, a_a, gid,
                                    .access = acc, .base = 0,
                                    .format = PIPE_FORMAT_R32_UINT,
                                    .sign_extend = false);
         nir_def *vb = nir_load_agx(&b, 4, 32, a_b, i,
                                    .access = ACCESS_CAN_SPECULATE, .base = 0,
                                    .format = PIPE_FORMAT_R32_UINT,
                                    .sign_extend = false);
         nir_def *x = nir_ixor(&b, nir_channel(&b, va, 0), nir_channel(&b, vb, 0));
         nir_def *s = nir_iadd(&b, nir_load_var(&b, sum_var), x);
         if (mode == 1) {
            nir_store_agx(&b, s, a_o, i,
                          .access = ACCESS_NON_READABLE, .base = 0,
                          .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
         }
         nir_store_var(&b, sum_var, s, 1);
         nir_store_var(&b, i_var, nir_iadd_imm(&b, i, 1), 1);
      }
      nir_pop_loop(&b, NULL);

      nir_store_agx(&b, nir_load_var(&b, sum_var), a_o, gid,
                    .access = ACCESS_NON_READABLE, .base = 0,
                    .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);

      unsigned budget = (mode == 4) ? 4 : 64;
      NIR_PASS(_, b.shader, agx_nir_opt_licm, budget, true);
      nir_validate_shader(b.shader, "after pure readonly loop licm");

      unsigned loads_in_loop = 0;
      unsigned loads_outside_loop = 0;
      nir_foreach_block(blk, b.impl) {
         bool in_loop = false;
         for (nir_cf_node *n = blk->cf_node.parent; n; n = n->parent) {
            if (n->type == nir_cf_node_loop) {
               in_loop = true;
               break;
            }
         }
         nir_foreach_instr(instr, blk) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_agx) {
               if (in_loop)
                  loads_in_loop++;
               else
                  loads_outside_loop++;
            }
         }
      }

      if (mode == 0) {
         /* Both cnt (Phase 1) and va (Phase 2) hoisted; only vb stays in loop. */
         EXPECT_EQ(loads_in_loop, 1u);
         EXPECT_EQ(loads_outside_loop, 2u);

         /* Verify the Phase 2 zero-trip guard polarity and merge-phi else
          * source so flipped-guard and undef-init regressions are caught. */
         nir_if *guard = NULL;
         foreach_list_typed(nir_cf_node, node, node, &b.impl->body) {
            if (node->type == nir_cf_node_if) {
               guard = nir_cf_node_as_if(node);
               break;
            }
         }
         ASSERT_NE(guard, nullptr);

         /* Guard condition must be inot(0 >= cnt), entering when cnt > 0. */
         nir_scalar gcond = nir_get_scalar(guard->condition.ssa, 0);
         ASSERT_TRUE(nir_scalar_is_alu(gcond));
         ASSERT_EQ(nir_scalar_alu_op(gcond), nir_op_inot);
         nir_scalar brk = nir_scalar_chase_alu_src(gcond, 0);
         ASSERT_TRUE(nir_scalar_is_alu(brk));
         ASSERT_EQ(nir_scalar_alu_op(brk), nir_op_ige);
         nir_scalar lhs = nir_scalar_chase_alu_src(brk, 0);
         nir_scalar rhs = nir_scalar_chase_alu_src(brk, 1);
         EXPECT_TRUE(nir_scalar_is_const(lhs));
         EXPECT_EQ(nir_scalar_as_uint(lhs), 0u);
         EXPECT_EQ(nir_def_instr(rhs.def)->type, nir_instr_type_intrinsic);
         EXPECT_EQ(nir_instr_as_intrinsic(nir_def_instr(rhs.def))->intrinsic,
                   nir_intrinsic_load_agx);

         /* Merge phi after guard must take the preheader init value (0) on
          * the zero-trip else path and the loop header phi on the then path. */
         nir_block *after = nir_cf_node_as_block(nir_cf_node_next(&guard->cf_node));
         nir_block *then_end = nir_if_last_then_block(guard);
         nir_block *else_end = nir_if_last_else_block(guard);
         nir_phi_instr *merge_phi = NULL;
         nir_foreach_phi(phi, after) {
            merge_phi = phi;
            break;
         }
         ASSERT_NE(merge_phi, nullptr);
         nir_phi_src *else_src = nir_phi_get_src_from_block(merge_phi, else_end);
         nir_phi_src *then_src = nir_phi_get_src_from_block(merge_phi, then_end);
         ASSERT_NE(else_src, nullptr);
         ASSERT_NE(then_src, nullptr);
         EXPECT_TRUE(nir_src_is_const(else_src->src));
         EXPECT_EQ(nir_src_as_uint(else_src->src), 0u);
         EXPECT_EQ(nir_def_instr(then_src->src.ssa)->type, nir_instr_type_phi);
      } else if (mode == 4) {
         /* Budget 4 halves forbids hoisting va (8 halves). */
         EXPECT_GE(loads_in_loop, 2u);
      } else {
         /* Stores, barriers, and coherent loads block hoisting. */
         EXPECT_EQ(loads_in_loop, 3u);
         EXPECT_EQ(loads_outside_loop, 0u);
      }

      ralloc_free(b.shader);
   }

   /* Mode 5: early-break linear search with an exit phi must bail safely. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "ro_loop_early_break_phi");
      nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *p_keys = nir_load_preamble(&b, 2, 32, .base = 0);
      nir_def *a_keys = nir_pack_64_2x32_split(&b, nir_channel(&b, p_keys, 0),
                                               nir_channel(&b, p_keys, 1));
      nir_def *p_data = nir_load_preamble(&b, 2, 32, .base = 4);
      nir_def *a_data = nir_pack_64_2x32_split(&b, nir_channel(&b, p_data, 0),
                                               nir_channel(&b, p_data, 1));
      nir_def *p_o = nir_load_preamble(&b, 2, 32, .base = 8);
      nir_def *a_o = nir_pack_64_2x32_split(&b, nir_channel(&b, p_o, 0),
                                            nir_channel(&b, p_o, 1));
      nir_def *n = nir_load_preamble(&b, 1, 32, .base = 12);

      nir_variable *i_var = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
      nir_variable *f_var = nir_local_variable_create(b.impl, glsl_int_type(), "f");
      nir_store_var(&b, i_var, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, f_var, nir_imm_int(&b, -1), 1);

      nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, i_var);
         nir_push_if(&b, nir_uge(&b, i, n));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         nir_def *k = nir_load_agx(&b, 1, 32, a_keys, nir_imm_int(&b, 0),
                                   .access = ACCESS_CAN_SPECULATE, .base = 0,
                                   .format = PIPE_FORMAT_R32_UINT,
                                   .sign_extend = false);
         nir_def *d = nir_load_agx(&b, 1, 32, a_data, i,
                                   .access = ACCESS_CAN_SPECULATE, .base = 0,
                                   .format = PIPE_FORMAT_R32_UINT,
                                   .sign_extend = false);
         nir_push_if(&b, nir_ieq(&b, d, k));
         {
            nir_store_var(&b, f_var, i, 1);
            nir_jump(&b, nir_jump_break);
         }
         nir_pop_if(&b, NULL);
         nir_store_var(&b, i_var, nir_iadd_imm(&b, i, 1), 1);
      }
      nir_pop_loop(&b, NULL);

      nir_store_agx(&b, nir_load_var(&b, f_var), a_o, gid,
                    .access = ACCESS_NON_READABLE, .base = 0,
                    .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);

      NIR_PASS(_, b.shader, agx_nir_opt_licm, 64, true);
      nir_validate_shader(b.shader, "after early-break exit-phi licm");

      unsigned loads_in_loop = 0;
      nir_foreach_block(blk, b.impl) {
         bool in_loop = false;
         for (nir_cf_node *cf = blk->cf_node.parent; cf; cf = cf->parent) {
            if (cf->type == nir_cf_node_loop) {
               in_loop = true;
               break;
            }
         }
         if (!in_loop)
            continue;
         nir_foreach_instr(instr, blk) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_agx)
               loads_in_loop++;
         }
      }
      EXPECT_EQ(loads_in_loop, 2u);
      ralloc_free(b.shader);
   }

   /* Mode 6: speculative == false (e.g. graphics / internal shaders) must NOT
    * hoist pure read-only loop loads. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "ro_loop_no_spec");
      nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *p_a = nir_load_preamble(&b, 2, 32, .base = 0);
      nir_def *a_a = nir_pack_64_2x32_split(&b, nir_channel(&b, p_a, 0),
                                            nir_channel(&b, p_a, 1));
      nir_def *p_o = nir_load_preamble(&b, 2, 32, .base = 4);
      nir_def *a_o = nir_pack_64_2x32_split(&b, nir_channel(&b, p_o, 0),
                                            nir_channel(&b, p_o, 1));
      nir_def *n = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_variable *i_var = nir_local_variable_create(b.impl, glsl_int_type(), "i");
      nir_variable *s_var = nir_local_variable_create(b.impl, glsl_int_type(), "s");
      nir_store_var(&b, i_var, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, s_var, nir_imm_int(&b, 0), 1);
      nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, i_var);
         nir_push_if(&b, nir_ige(&b, i, n));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);
         nir_def *va = nir_load_agx(&b, 1, 32, a_a, gid,
                                    .access = ACCESS_CAN_SPECULATE, .base = 0,
                                    .format = PIPE_FORMAT_R32_UINT,
                                    .sign_extend = false);
         nir_store_var(&b, s_var, nir_iadd(&b, nir_load_var(&b, s_var), va), 1);
         nir_store_var(&b, i_var, nir_iadd_imm(&b, i, 1), 1);
      }
      nir_pop_loop(&b, NULL);
      nir_store_agx(&b, nir_load_var(&b, s_var), a_o, gid,
                    .access = ACCESS_NON_READABLE, .base = 0,
                    .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      NIR_PASS(_, b.shader, agx_nir_opt_licm, 64, false);
      nir_validate_shader(b.shader, "after non-speculative licm");

      unsigned loads_in_loop = 0;
      nir_foreach_block(blk, b.impl) {
         bool in_loop = false;
         for (nir_cf_node *cf = blk->cf_node.parent; cf; cf = cf->parent) {
            if (cf->type == nir_cf_node_loop) {
               in_loop = true;
               break;
            }
         }
         if (!in_loop)
            continue;
         nir_foreach_instr(instr, blk) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_agx)
               loads_in_loop++;
         }
      }
      EXPECT_EQ(loads_in_loop, 1u);
      ralloc_free(b.shader);
   }
}
