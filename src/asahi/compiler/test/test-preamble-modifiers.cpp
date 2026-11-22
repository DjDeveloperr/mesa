/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
}

static void
check_modifier(bool g17p, bool widen)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "uniform modifiers");
   nir_def *value = agx_test_load_preamble(&b, 1, widen ? 16 : 32, 4);
   value = widen ? nir_f2f32(&b, value) : nir_fabs(&b, value);
   nir_def *x = nir_channel(&b, nir_load_frag_coord(&b), 0);
   nir_def *result = nir_fmul(&b, x, value);
   agx_test_store_output(&b, nir_vec4(&b, result, result, result, result),
                         nir_imm_int(&b, 0), nir_type_float32,
                         FRAG_RESULT_DATA0);
   unsigned sizes[3] = {16, 0, 0};
   bool progress = agx_nir_opt_preamble(b.shader, sizes, g17p, false, NULL);
   const bool hoisted = g17p && !widen;
   if (hoisted) {
      EXPECT_TRUE(progress);
   }
   unsigned modifiers = 0, exported = 0;
   nir_function_impl *main = nir_shader_get_entrypoint(b.shader);
   nir_foreach_block(block, main) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_alu) {
            nir_op op = nir_instr_as_alu(instr)->op;
            modifiers += op == (widen ? nir_op_f2f32 : nir_op_fabs);
         } else if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            exported += intr->intrinsic == nir_intrinsic_load_preamble &&
                        nir_intrinsic_base(intr) >= 16;
         }
      }
   }
   EXPECT_EQ(modifiers, hoisted ? 0u : 1u);
   EXPECT_EQ(exported, hoisted ? 1u : 0u);
   if (hoisted) {
      EXPECT_EQ(sizes[0], 18u);
   }
   ralloc_free(b.shader);
}

TEST(PreambleModifiers, HoistsUniformAbsoluteValueOnAGX3B)
{
   check_modifier(true, false);
   check_modifier(false, false);
}

TEST(PreambleModifiers, PreservesFreeUniformHalfWidening)
{
   check_modifier(true, true);
   check_modifier(false, true);
}

/* Uniform fneg feeding a float comparison cannot fold into AGX3C compare-select
 * and is non-free on AGX3C, so preamble optimization hoists it while leaving
 * AGX3B, AGX2, and non-compare uses untouched. */
TEST(PreambleModifiers, HoistsUniformNegateFeedingFloatCompareOnAGX3C)
{
   struct variant {
      const char *name;
      bool g17p;
      bool agx3c;
      bool compare_use;
      bool expect_hoisted;
   };

   const variant cases[] = {
      {"agx3c_cmp", true, true, true, true},
      {"agx3b_cmp", true, false, true, false},
      {"agx2_cmp", false, false, true, false},
      {"agx3c_fadd", true, true, false, false},
      {"agx3b_fadd", true, false, false, false},
   };

   for (const variant &c : cases) {
      SCOPED_TRACE(c.name);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "uniform fneg compare");
      nir_def *value = agx_test_load_preamble(&b, 1, 32, 4);
      nir_def *neg = nir_fneg(&b, value);
      nir_def *x = nir_channel(&b, nir_load_frag_coord(&b), 0);
      nir_def *result =
         c.compare_use
            ? nir_bcsel(&b, nir_flt(&b, x, neg), nir_imm_float(&b, 1.0f),
                        nir_imm_float(&b, 0.0f))
            : nir_fadd(&b, x, neg);
      agx_test_store_output(&b, nir_vec4(&b, result, result, result, result),
                            nir_imm_int(&b, 0), nir_type_float32,
                            FRAG_RESULT_DATA0);
      unsigned sizes[3] = {16, 0, 0};
      agx_nir_opt_preamble(b.shader, sizes, c.g17p, c.agx3c, NULL);

      unsigned fnegs = 0, exported = 0;
      nir_function_impl *main = nir_shader_get_entrypoint(b.shader);
      nir_foreach_block(block, main) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu) {
               fnegs += nir_instr_as_alu(instr)->op == nir_op_fneg;
            } else if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               exported += intr->intrinsic == nir_intrinsic_load_preamble &&
                           nir_intrinsic_base(intr) >= 16;
            }
         }
      }
      EXPECT_EQ(fnegs, c.expect_hoisted ? 0u : 1u);
      EXPECT_EQ(exported, c.expect_hoisted ? 1u : 0u);
      ralloc_free(b.shader);
   }
}
