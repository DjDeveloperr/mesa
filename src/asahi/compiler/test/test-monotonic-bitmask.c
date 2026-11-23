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

struct mask_test_opts {
   uint32_t bound_k;
   uint32_t step_s;
   uint32_t init_mask;
   bool duplicate_offset;
   bool extra_or_high_bit;
};

static nir_shader *
build_bitmask_loop_shader(const struct mask_test_opts *opts)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "monotonic_bitmask");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *gid_x = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *src_ptr = nir_load_preamble(&b, 1, 64, .base = 8);
   nir_def *dst_ptr = nir_load_preamble(&b, 1, 64, .base = 12);

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *init_m = nir_imm_int(&b, opts->init_mask);

   nir_loop *loop = nir_push_loop(&b);
   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
   nir_phi_instr *iv_phi = nir_phi_instr_create(b.shader);
   nir_phi_instr *mask_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&iv_phi->instr, &iv_phi->def, 1, 32);
   nir_def_init(&mask_phi->instr, &mask_phi->def, 1, 32);
   nir_phi_instr_add_src(iv_phi, pre, zero);
   nir_phi_instr_add_src(mask_phi, pre, init_m);
   nir_builder_instr_insert(&b, &iv_phi->instr);
   nir_builder_instr_insert(&b, &mask_phi->instr);

   nir_if *exit_if =
      nir_push_if(&b, nir_ige(&b, &iv_phi->def, nir_imm_int(&b, opts->bound_k)));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, exit_if);

   nir_def *cur_mask = &mask_phi->def;
   for (uint32_t s = 0; s < opts->step_s; ++s) {
      uint32_t off = (opts->duplicate_offset && s > 0) ? 0 : s;
      nir_def *pos = off ? nir_iadd_imm(&b, &iv_phi->def, off) : &iv_phi->def;
      nir_def *sh = nir_iand_imm(&b, pos, 31u);
      nir_def *bit = nir_ishl(&b, nir_imm_int(&b, 1), sh);
      nir_def *with_bit = nir_ior(&b, cur_mask, bit);
      nir_def *taut = nir_ine(&b, with_bit, cur_mask);

      nir_if *guard = nir_push_if(&b, taut);
      nir_def *a = nir_load_agx(&b, 1, 8, src_ptr, nir_iadd(&b, gid_x, pos),
                                .access = ACCESS_NON_WRITEABLE, .base = 0,
                                .format = PIPE_FORMAT_R8_UINT,
                                .sign_extend = false);
      nir_def *c = nir_ult(&b, nir_u2u32(&b, a), nir_imm_int(&b, 128));
      nir_def *sel = nir_bcsel(&b, c, with_bit, cur_mask);
      nir_pop_if(&b, guard);

      nir_phi_instr *mphi = nir_phi_instr_create(b.shader);
      nir_def_init(&mphi->instr, &mphi->def, 1, 32);
      nir_phi_instr_add_src(mphi, nir_if_last_then_block(guard), sel);
      nir_phi_instr_add_src(mphi, nir_if_last_else_block(guard), cur_mask);
      nir_builder_instr_insert(&b, &mphi->instr);
      cur_mask = &mphi->def;
   }

   if (opts->extra_or_high_bit)
      cur_mask = nir_ior(&b, cur_mask, nir_imm_int(&b, 0x80000000u));

   nir_def *iv_next = nir_iadd_imm(&b, &iv_phi->def, opts->step_s);
   nir_block *latch = nir_loop_last_block(loop);
   add_latch_phi_src(iv_phi, latch, iv_next);
   add_latch_phi_src(mask_phi, latch, cur_mask);
   nir_pop_loop(&b, loop);

   nir_store_agx(&b, &mask_phi->def, dst_ptr, gid_x,
                 .access = ACCESS_NON_READABLE, .base = 2,
                 .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);

   nir_validate_shader(b.shader, "initial");
   return b.shader;
}

static unsigned
count_ifs_in_loop(nir_shader *s)
{
   unsigned n = 0;
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (node->type != nir_cf_node_loop)
         continue;
      nir_loop *loop = nir_cf_node_as_loop(node);
      foreach_list_typed(nir_cf_node, inner, node, &loop->body) {
         if (inner->type == nir_cf_node_if)
            n++;
      }
   }
   return n;
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
         .reserved_preamble = 24,
         .no_preamble = true,
      };
      struct agx_shader_part part = {0};
      bool ok = agx_compile_shader_nir(clone, &key, &part);
      CHECK(ok);
      CHECK(part.info.binary_size > 0);
      free(part.binary);
      ralloc_free(clone);
   }
}

int
main(void)
{
   /* 1. Positive: unit-step 32-bit loop removes tautological inner if. */
   {
      struct mask_test_opts opts = {.bound_k = 32, .step_s = 1};
      nir_shader *s = build_bitmask_loop_shader(&opts);
      CHECK(count_ifs_in_loop(s) == 2);
      CHECK(agx_nir_opt_monotonic_bitmask(s));
      nir_opt_dead_cf(s);
      nir_opt_dce(s);
      nir_validate_shader(s, "after step1");
      /* Only the loop-exit if remains. */
      CHECK(count_ifs_in_loop(s) == 1);
      check_compiles_both_agx3(s);
      ralloc_free(s);
   }

   /* 2. Positive: 2x-unrolled step=2, bound=31 removes BOTH inner ifs. */
   {
      struct mask_test_opts opts = {.bound_k = 31, .step_s = 2};
      nir_shader *s = build_bitmask_loop_shader(&opts);
      CHECK(count_ifs_in_loop(s) == 3);
      CHECK(agx_nir_opt_monotonic_bitmask(s));
      nir_opt_dead_cf(s);
      nir_opt_dce(s);
      nir_validate_shader(s, "after step2");
      CHECK(count_ifs_in_loop(s) == 1);
      check_compiles_both_agx3(s);
      ralloc_free(s);
   }

   /* 3. Negative: non-zero initial mask cannot prove bit 0 is clear. */
   {
      struct mask_test_opts opts = {.bound_k = 32, .step_s = 1, .init_mask = 1};
      nir_shader *s = build_bitmask_loop_shader(&opts);
      CHECK(!agx_nir_opt_monotonic_bitmask(s));
      ralloc_free(s);
   }

   /* 4. Negative: bound > 32 wraps bit index modulo 32. */
   {
      struct mask_test_opts opts = {.bound_k = 64, .step_s = 1};
      nir_shader *s = build_bitmask_loop_shader(&opts);
      CHECK(!agx_nir_opt_monotonic_bitmask(s));
      ralloc_free(s);
   }

   /* 5. Negative: extra non-monotone OR invalidates invariant. */
   {
      struct mask_test_opts opts = {
         .bound_k = 32, .step_s = 1, .extra_or_high_bit = true};
      nir_shader *s = build_bitmask_loop_shader(&opts);
      CHECK(!agx_nir_opt_monotonic_bitmask(s));
      ralloc_free(s);
   }

   /* 6. Negative: duplicate bit offset within one iteration rejected. */
   {
      struct mask_test_opts opts = {
         .bound_k = 31, .step_s = 2, .duplicate_offset = true};
      nir_shader *s = build_bitmask_loop_shader(&opts);
      CHECK(!agx_nir_opt_monotonic_bitmask(s));
      ralloc_free(s);
   }

   return failures ? 1 : 0;
}
