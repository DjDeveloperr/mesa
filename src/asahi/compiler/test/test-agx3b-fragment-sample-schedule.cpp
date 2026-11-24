/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_test.h"

#include <gtest/gtest.h>
#include <vector>

/* AGX3B fragment sample-issue scheduling: low-pressure multi-tap fragment
 * blocks hoist independent texture samples up to AGX_ILP_AGX3B_FS_SAMPLE_HALVES
 * (48 halves) while charging aligned RA vector widths and keeping each
 * 4-channel sample destination's 8-half aligned slot charged across its split
 * until the last live channel dies. */

namespace {

struct frag_shader {
   void *mem;
   agx_shader_key key;
   agx_builder *b;
   agx_block *body;
};

static void
frag_shader_init(frag_shader *s, bool loop,
                 enum agx_isa_encoding encoding = AGX_ISA_ENCODING_AGX3B)
{
   s->mem = ralloc_context(NULL);
   s->key = {};
   s->key.dev.encoding = encoding;
   s->b = agx_test_builder(s->mem);
   s->b->shader->key = &s->key;
   s->b->shader->stage = MESA_SHADER_FRAGMENT;
   s->b->shader->nir = rzalloc(s->mem, nir_shader);
   s->b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   s->body = agx_start_block(s->b->shader);
   if (loop) {
      agx_block *entry = s->body;
      s->body = agx_test_block(s->b->shader);
      agx_block *exit = agx_test_block(s->b->shader);
      s->body->loop_header = true;
      agx_block_add_successor(entry, s->body);
      agx_block_add_successor(s->body, s->body);
      agx_block_add_successor(s->body, exit);
   }
}

/* Emit n independent texture samples/loads with compiler-accurate 4-channel
 * 32-bit destinations followed by a 4-way split (matching agx_compile.c and
 * agx_expand_tex_to), consuming `channels` live components in a running FADD
 * chain, optionally with `live` 32-bit values kept alive across all samples. */
static void
emit_multi_tap(frag_shader *s, unsigned n, unsigned channels, unsigned live,
               bool is_load, unsigned coord_comps = 2, bool is_shadow = false)
{
   assert(channels >= 1 && channels <= 4);
   assert(coord_comps == 2 || coord_comps == 3);
   agx_builder *b = s->b;
   b->cursor = agx_after_block(s->body);

   agx_index u = agx_preload(b, agx_register(2, AGX_SIZE_32));
   agx_index v = agx_preload(b, agx_register(4, AGX_SIZE_32));
   agx_index layer =
      coord_comps == 3 ? agx_preload(b, agx_register(6, AGX_SIZE_32)) : agx_null();
   agx_index held[64];
   assert(live <= 64);
   for (unsigned i = 0; i < live; ++i)
      held[i] = agx_mov_imm(b, 32, 0x3f800000u + i);

   agx_index acc[4] = {};

   for (unsigned i = 0; i < n; ++i) {
      agx_index off = agx_mov_imm(b, 32, 0x3d000000u + (i << 16));
      agx_index cu = agx_fadd(b, u, off);
      agx_index cv = agx_fadd(b, v, off);
      agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, coord_comps);
      agx_instr *collect = agx_collect_to(b, coords, coord_comps);
      collect->src[0] = cu;
      collect->src[1] = cv;
      if (coord_comps == 3)
         collect->src[2] = layer;

      agx_index dest = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *tex = agx_texture_sample_to(
         b, dest, coords, agx_immediate(0), agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(),
         coord_comps == 3 ? AGX_DIM_2D_ARRAY : AGX_DIM_2D,
         AGX_LOD_MODE_AUTO_LOD, BITFIELD_MASK(channels), false, is_shadow,
         false, AGX_GATHER_NONE, false);
      if (is_load)
         tex->op = AGX_OPCODE_TEXTURE_LOAD;

      agx_instr *split = agx_split(b, 4, dest);
      for (unsigned c = 0; c < 4; ++c) {
         split->dest[c] =
            c < channels ? agx_temp(b->shader, AGX_SIZE_32) : agx_null();
      }
      for (unsigned c = 0; c < channels; ++c) {
         acc[c] = i == 0 ? agx_fadd(b, split->dest[c], split->dest[c])
                         : agx_fadd(b, acc[c], split->dest[c]);
      }
   }

   agx_index out = acc[0];
   for (unsigned c = 1; c < channels; ++c)
      out = agx_fadd(b, out, acc[c]);
   for (unsigned i = 0; i < live; ++i)
      out = agx_fadd(b, out, held[i]);
   agx_unit_test(b, out);
}

static bool
is_tex_sample_or_load(const agx_instr *I)
{
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
          I->op == AGX_OPCODE_TEXTURE_LOAD;
}

/* Count texture samples/loads issued before the first instruction that reads
 * any sample's result (directly or through its split). */
static unsigned
samples_before_first_consumer(agx_block *block, unsigned alloc)
{
   std::vector<bool> from_tex(alloc, false);
   unsigned samples = 0;
   agx_foreach_instr_in_block(block, I) {
      if (is_tex_sample_or_load(I)) {
         ++samples;
         agx_foreach_ssa_dest(I, d)
            from_tex[I->dest[d].value] = true;
         continue;
      }
      if (I->op == AGX_OPCODE_SPLIT && I->src[0].type == AGX_INDEX_NORMAL &&
          from_tex[I->src[0].value]) {
         agx_foreach_ssa_dest(I, d)
            from_tex[I->dest[d].value] = true;
         continue;
      }
      bool reads_tex = false;
      agx_foreach_ssa_src(I, s)
         reads_tex |= from_tex[I->src[s].value];
      if (reads_tex)
         break;
   }
   return samples;
}

/* Maximum number of coordinate collects live simultaneously. */
static unsigned
max_live_coord_collects(agx_block *block, unsigned alloc)
{
   std::vector<bool> is_coord(alloc, false);
   agx_foreach_instr_in_block(block, I) {
      if (is_tex_sample_or_load(I) && I->src[0].type == AGX_INDEX_NORMAL)
         is_coord[I->src[0].value] = true;
   }
   unsigned live = 0, peak = 0;
   agx_foreach_instr_in_block(block, I) {
      if (I->op == AGX_OPCODE_COLLECT && I->dest[0].type == AGX_INDEX_NORMAL &&
          is_coord[I->dest[0].value]) {
         ++live;
         peak = MAX2(peak, live);
      }
      if (is_tex_sample_or_load(I) && I->src[0].type == AGX_INDEX_NORMAL &&
          is_coord[I->src[0].value] && live > 0) {
         --live;
      }
   }
   return peak;
}

} /* namespace */

TEST(AGX3BFragmentSampleSchedule, FiveTapVec3SamplesIssueAheadOfConsumers)
{
   frag_shader s;
   frag_shader_init(&s, false);
   emit_multi_tap(&s, 5, 3, 0, false);
   agx_ilp_schedule(s.b->shader);
   EXPECT_EQ(samples_before_first_consumer(s.body, s.b->shader->alloc), 5u);
   ralloc_free(s.mem);
}

TEST(AGX3BFragmentSampleSchedule, FourTapVec4TextureLoadsIssueAheadOfConsumers)
{
   frag_shader s;
   frag_shader_init(&s, false);
   emit_multi_tap(&s, 4, 4, 0, true);
   agx_ilp_schedule(s.b->shader);
   EXPECT_EQ(samples_before_first_consumer(s.body, s.b->shader->alloc), 4u);
   ralloc_free(s.mem);
}

TEST(AGX3BFragmentSampleSchedule, NineTapSingleChannelShadowDrainsInsideCap)
{
   frag_shader s;
   frag_shader_init(&s, false);
   /* 9 single-channel 2D-array shadow taps: each 4-channel sample dest keeps
    * its 8-half aligned slot charged across split and each 3-word coordinate
    * collect charges 8 aligned halves, so the scheduler must drain early taps
    * without hoisting all 9 or piling up future coordinate collects. */
   emit_multi_tap(&s, 9, 1, 0, false, 3, true);
   agx_ilp_schedule(s.b->shader);
   const unsigned hoisted =
      samples_before_first_consumer(s.body, s.b->shader->alloc);
   EXPECT_EQ(hoisted, 4u);
   EXPECT_LE(max_live_coord_collects(s.body, s.b->shader->alloc), 2u);
   ralloc_free(s.mem);
}

TEST(AGX3BFragmentSampleSchedule, NineTapSingleChannelColorDrainsInsideCap)
{
   frag_shader s;
   frag_shader_init(&s, false);
   /* Non-shadow single-channel 9-tap blur uses the exact same aligned-slot
    * model (no shadow-specific special case) and must likewise drain early
    * taps instead of letting split drop live pressure and hoist all 9. */
   emit_multi_tap(&s, 9, 1, 0, false, 2, false);
   agx_ilp_schedule(s.b->shader);
   const unsigned hoisted =
      samples_before_first_consumer(s.body, s.b->shader->alloc);
   EXPECT_EQ(hoisted, 5u);
   EXPECT_LE(max_live_coord_collects(s.body, s.b->shader->alloc), 2u);
   ralloc_free(s.mem);
}

TEST(AGX3BFragmentSampleSchedule, ThreeTapSamplesStayInPressureOrder)
{
   frag_shader s;
   frag_shader_init(&s, false);
   emit_multi_tap(&s, 3, 3, 0, false);
   agx_ilp_schedule(s.b->shader);
   EXPECT_EQ(samples_before_first_consumer(s.body, s.b->shader->alloc), 1u);
   ralloc_free(s.mem);
}

/* 26 values held across the taps put the block past the low-pressure budget:
 * the 48-half cap would refuse every hoist, while the program's occupancy cap
 * lets all five samples issue ahead of their consumers. */
TEST(AGX3BFragmentSampleSchedule, HighPressureBlockTakesOccupancySlack)
{
   frag_shader s;
   frag_shader_init(&s, false);
   emit_multi_tap(&s, 5, 3, 26, false);
   agx_ilp_schedule(s.b->shader);
   EXPECT_EQ(samples_before_first_consumer(s.body, s.b->shader->alloc), 5u);
   ralloc_free(s.mem);
}

TEST(AGX3BFragmentSampleSchedule, WideNineTapBlockRespects48HalfCap)
{
   frag_shader s;
   frag_shader_init(&s, false);
   emit_multi_tap(&s, 9, 4, 0, false);
   agx_ilp_schedule(s.b->shader);
   const unsigned hoisted =
      samples_before_first_consumer(s.body, s.b->shader->alloc);
   EXPECT_EQ(hoisted, 5u);
   ralloc_free(s.mem);
}

TEST(AGX3BFragmentSampleSchedule, AGX2EncodingRetainsFixedSlackCap)
{
   frag_shader s;
   frag_shader_init(&s, false, AGX_ISA_ENCODING_AGX2);
   emit_multi_tap(&s, 5, 3, 0, false);
   agx_ilp_schedule(s.b->shader);
   EXPECT_EQ(samples_before_first_consumer(s.body, s.b->shader->alloc), 1u);
   ralloc_free(s.mem);
}

TEST(AGX3BFragmentSampleSchedule, LoopBlockStaysUnchanged)
{
   frag_shader s;
   frag_shader_init(&s, true);
   emit_multi_tap(&s, 5, 3, 0, false);
   agx_ilp_schedule(s.b->shader);
   EXPECT_EQ(samples_before_first_consumer(s.body, s.b->shader->alloc), 1u);
   ralloc_free(s.mem);
}
