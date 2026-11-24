/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <cstring>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"
#include "agx_debug.h"
#include "agx_nir.h"
}

static bool
m5_cf_inside_if(const nir_instr *instr)
{
   for (nir_cf_node *cf = instr->block->cf_node.parent; cf; cf = cf->parent) {
      if (cf->type == nir_cf_node_if)
         return true;
   }
   return false;
}

/* A uniform-address atomic add whose old value each lane reads runs once per
 * subgroup on M5: one elected lane adds the active lane count, and every lane
 * adds its exclusive scan to the broadcast old value. */
TEST(UniformAtomics, M5UniformAddressElectsOneLane)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "M5 uniform-address atomic");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *counter = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *out = agx_test_load_preamble(&b, 1, 64, 4);
   /* Set the indices through the accessors (see agx_test_load_preamble()). */
   nir_def *old = nir_global_atomic(&b, 32, counter, nir_imm_int(&b, 1));
   nir_intrinsic_set_atomic_op(nir_def_as_intrinsic(old), nir_atomic_op_iadd);
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_intrinsic_instr *store = nir_store_global(
      &b, old, nir_iadd(&b, out, nir_u2u64(&b, nir_imul_imm(&b, lane, 4))));
   nir_intrinsic_set_align(store, 4, 0);

   agx_preprocess_nir_m5(b.shader);

   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   key.no_preamble = true;
   key.reserved_preamble = 8;
   ASSERT_TRUE(agx_supports_uniform_atomics(&key.dev, MESA_SHADER_COMPUTE));

   agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
      << agx_last_compile_refusal();
   EXPECT_NE(part.binary, nullptr);

   unsigned atomics = 0, elected_atomics = 0;
   bool active_index = false, active_count = false;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (strstr(nir_intrinsic_infos[intr->intrinsic].name, "atomic")) {
               atomics++;
               elected_atomics += m5_cf_inside_if(instr);
            }
            active_index |= intr->intrinsic ==
                            nir_intrinsic_load_active_subgroup_invocation_agx;
            active_count |= intr->intrinsic ==
                            nir_intrinsic_load_active_subgroup_count_agx;
         }
      }
   }

   /* One atomic, under the elect branch, rather than one per lane. */
   EXPECT_EQ(atomics, 1u);
   EXPECT_EQ(elected_atomics, 1u);
   /* SR 34 elects the lane and numbers the lanes for the scan. */
   EXPECT_TRUE(active_index);
   /* The elected lane adds the active lane count. */
   EXPECT_TRUE(active_count);

   free(part.binary);
   ralloc_free(b.shader);
}

/* The stages whose SR 34 reads the AGX3C packer admits, without the fragment
 * stage, where the active-count reduction also counts helper invocations. */
TEST(UniformAtomics, M5Stages)
{
   agx_device_key m5 = {};
   m5.encoding = AGX_ISA_ENCODING_AGX3C;
   m5.chip = AGX_CHIP_G17S;
   for (mesa_shader_stage stage :
        {MESA_SHADER_COMPUTE, MESA_SHADER_VERTEX, MESA_SHADER_TESS_CTRL})
      EXPECT_TRUE(agx_supports_uniform_atomics(&m5, stage)) << stage;
   for (mesa_shader_stage stage :
        {MESA_SHADER_FRAGMENT, MESA_SHADER_GEOMETRY, MESA_SHADER_TESS_EVAL,
         MESA_SHADER_TASK, MESA_SHADER_MESH})
      EXPECT_FALSE(agx_supports_uniform_atomics(&m5, stage)) << stage;

   agx_device_key g18g = m5;
   g18g.chip = AGX_CHIP_G18G;
   g18g.emit_agx3d = true;
   EXPECT_FALSE(agx_supports_uniform_atomics(&g18g, MESA_SHADER_COMPUTE));

   agx_device_key g17p = {};
   g17p.encoding = AGX_ISA_ENCODING_AGX3B;
   g17p.chip = AGX_CHIP_G17P;
   EXPECT_TRUE(agx_supports_uniform_atomics(&g17p, MESA_SHADER_FRAGMENT));
   EXPECT_FALSE(agx_supports_uniform_atomics(&g17p, MESA_SHADER_GEOMETRY));
}
