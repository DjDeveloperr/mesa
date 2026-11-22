/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned
count_op(nir_shader *nir, nir_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, nir) {
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

static uint32_t
ref_op(nir_op op, uint32_t nu, uint32_t du)
{
   int32_t ns = (int32_t)nu;
   int32_t ds = (int32_t)du;
   switch (op) {
   case nir_op_udiv:
      return nu / du;
   case nir_op_umod:
      return nu % du;
   case nir_op_idiv:
      if (ns == INT32_MIN && ds == -1)
         return (uint32_t)INT32_MIN;
      return (uint32_t)(ns / ds);
   case nir_op_irem:
      if (ns == INT32_MIN && ds == -1)
         return 0;
      return (uint32_t)(ns % ds);
   case nir_op_imod: {
      if (ns == INT32_MIN && ds == -1)
         return 0;
      int32_t r = ns % ds;
      if (r != 0 && ((r ^ ds) < 0))
         r += ds;
      return (uint32_t)r;
   }
   default:
      abort();
   }
}

static uint32_t
eval_lowered(nir_shader *lowered, uint32_t n_val, uint32_t d_val)
{
   nir_shader *clone = nir_shader_clone(NULL, lowered);
   nir_function_impl *impl = nir_shader_get_entrypoint(clone);
   nir_builder b = nir_builder_create(impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_local_invocation_id) {
            b.cursor = nir_before_instr(instr);
            nir_def *v = nir_imm_ivec3(&b, (int)n_val, 0, 0);
            nir_def_replace(&intr->def, v);
         } else if (intr->intrinsic == nir_intrinsic_load_push_constant) {
            b.cursor = nir_before_instr(instr);
            nir_def *v = nir_imm_int(&b, (int)d_val);
            nir_def_replace(&intr->def, v);
         }
      }
   }

   bool progress;
   do {
      progress = false;
      progress |= nir_opt_constant_folding(clone);
      progress |= nir_opt_copy_prop(clone);
      progress |= nir_opt_dce(clone);
   } while (progress);

   uint32_t result = 0;
   bool found = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_store_global) {
            nir_intrinsic_instr *st = nir_instr_as_intrinsic(instr);
            if (!nir_src_is_const(st->src[0])) {
               fprintf(stderr, "FAIL: lowered shader did not constant-fold for n=0x%x d=0x%x\n",
                       n_val, d_val);
               nir_print_shader(clone, stderr);
               exit(1);
            }
            result = nir_src_as_uint(st->src[0]);
            found = true;
         }
      }
   }
   if (!found) {
      fprintf(stderr, "FAIL: store_global missing\n");
      exit(1);
   }
   ralloc_free(clone);
   return result;
}

static void
check_op(nir_op op)
{
   const nir_shader_compiler_options opts = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &opts,
                                                  "uniform_idiv");
   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *div = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0));
   nir_def *res = nir_build_alu2(&b, op, lid, div);
   nir_store_global(&b, res, nir_imm_int64(&b, 0x1000));

   bool progress = agx_nir_opt_uniform_idiv(b.shader);
   nir_validate_shader(b.shader, "after agx_nir_opt_uniform_idiv");
   if (!progress || count_op(b.shader, op) != 0) {
      fprintf(stderr, "FAIL: op %s was not lowered\n", nir_op_infos[op].name);
      exit(1);
   }

   static const uint32_t dividends[] = {
      0u, 1u, 2u, 7u, 15u, 16u, 1000u, 0x7ffffffeu, 0x7fffffffu,
      0x80000000u, 0x80000001u, 0xfffffff9u, 0xfffffffeu, 0xffffffffu,
   };
   static const uint32_t divisors[] = {
      1u, 2u, 3u, 5u, 7u, 16u, 255u, 65535u, 0x7fffffffu,
      0x80000000u, 0x80000001u, 0xfffffff9u, 0xfffffffeu, 0xffffffffu,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(dividends); ++i) {
      for (unsigned j = 0; j < ARRAY_SIZE(divisors); ++j) {
         uint32_t got = eval_lowered(b.shader, dividends[i], divisors[j]);
         uint32_t exp = ref_op(op, dividends[i], divisors[j]);
         if (got != exp) {
            fprintf(stderr, "FAIL: %s(0x%08x, 0x%08x) = 0x%08x, expected 0x%08x\n",
                    nir_op_infos[op].name, dividends[i], divisors[j], got, exp);
            exit(1);
         }
      }
   }

   ralloc_free(b.shader);
}

static void
check_skipped_cases(void)
{
   const nir_shader_compiler_options opts = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &opts,
                                                  "uniform_idiv_skip");
   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *lid1 = nir_channel(&b, nir_load_local_invocation_id(&b), 1);
   nir_def *wgid = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *ssbo = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0));
   nir_def *sh = nir_load_shared(&b, 1, 32, nir_imm_int(&b, 0));
   nir_def *pc0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0));
   nir_def *pc1 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4));

   /* Divergent, const, non-preamble-movable (wgid, ssbo, shared), and
    * preamble-movable/preamble-movable divisions must all stay untouched. */
   nir_def *q1 = nir_udiv(&b, lid, lid1);
   nir_def *q2 = nir_udiv(&b, lid, nir_imm_int(&b, 7));
   nir_def *q3 = nir_udiv(&b, lid, wgid);
   nir_def *q4 = nir_udiv(&b, lid, ssbo);
   nir_def *q5 = nir_udiv(&b, lid, sh);
   nir_def *q6 = nir_udiv(&b, pc0, pc1);

   nir_def *sum = nir_iadd(&b, nir_iadd(&b, q1, q2), nir_iadd(&b, q3, q4));
   sum = nir_iadd(&b, sum, nir_iadd(&b, q5, q6));
   nir_store_global(&b, sum, nir_imm_int64(&b, 0x1000));

   bool progress = agx_nir_opt_uniform_idiv(b.shader);
   nir_validate_shader(b.shader, "after skipped agx_nir_opt_uniform_idiv");
   if (progress || count_op(b.shader, nir_op_udiv) != 6) {
      fprintf(stderr, "FAIL: non-preamble/const/uniform-uniform divisors should be skipped (progress=%d udivs=%u)\n",
              progress, count_op(b.shader, nir_op_udiv));
      exit(1);
   }
   ralloc_free(b.shader);
}

static void
check_agx_intrinsic_coexistence(void)
{
   /* Presence of AGX intrinsics unknown to nir_divergence_analysis must not
    * crash or prevent lowering a valid push-constant divisor. */
   const nir_shader_compiler_options opts = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &opts,
                                                  "uniform_idiv_agx_intr");
   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *sys = nir_load_sysval_agx(&b, 1, 32, .binding = 0, .desc_set = 0);
   nir_def *div = nir_iadd_imm(&b, sys, 1);
   nir_def *q = nir_udiv(&b, lid, div);
   nir_store_global(&b, q, nir_imm_int64(&b, 0x1000));

   bool progress = agx_nir_opt_uniform_idiv(b.shader);
   nir_validate_shader(b.shader, "after agx intrinsic uniform_idiv");
   if (!progress || count_op(b.shader, nir_op_udiv) != 0) {
      fprintf(stderr, "FAIL: load_sysval_agx divisor should lower cleanly\n");
      exit(1);
   }
   ralloc_free(b.shader);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   check_op(nir_op_udiv);
   check_op(nir_op_idiv);
   check_op(nir_op_umod);
   check_op(nir_op_irem);
   check_op(nir_op_imod);
   check_skipped_cases();
   check_agx_intrinsic_coexistence();
   glsl_type_singleton_decref();
   return 0;
}
