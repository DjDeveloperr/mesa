/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_copy_forward.h"

#include <string.h>

static void
append_elided(uint64_t *elided_tags, uint32_t *elided_count, uint64_t tag)
{
   if (!elided_tags || !elided_count)
      return;
   if (*elided_count < HK_COPY_FORWARD_MAX_ENTRIES)
      elided_tags[(*elided_count)++] = tag;
}

static void
deactivate_entry(struct hk_copy_forward_state *state,
                 struct hk_copy_forward_entry *e)
{
   if (!e->active)
      return;
   e->active = false;
   if (state->active_count > 0)
      state->active_count--;
}

void
hk_copy_forward_reset(struct hk_copy_forward_state *state)
{
   memset(state, 0, sizeof(*state));
}

void
hk_copy_forward_invalidate_all(struct hk_copy_forward_state *state)
{
   if (state->active_count == 0)
      return;
   memset(state, 0, sizeof(*state));
}

bool
hk_copy_forward_range_valid(const struct hk_copy_forward_range *r)
{
   if (!r || r->size == 0)
      return false;
   if (r->address + r->size < r->address)
      return false;
   if (r->handle != 0 && r->offset + r->size < r->offset)
      return false;
   return true;
}

bool
hk_copy_forward_ranges_overlap(const struct hk_copy_forward_range *a,
                               const struct hk_copy_forward_range *b)
{
   if (!hk_copy_forward_range_valid(a) || !hk_copy_forward_range_valid(b))
      return true;

   const uint64_t a_va1 = a->address + a->size;
   const uint64_t b_va1 = b->address + b->size;
   if (a->address < b_va1 && b->address < a_va1)
      return true;

   if (a->handle != 0 && a->handle == b->handle) {
      const uint64_t a_bo1 = a->offset + a->size;
      const uint64_t b_bo1 = b->offset + b->size;
      if (a->offset < b_bo1 && b->offset < a_bo1)
         return true;
   }

   return false;
}

bool
hk_copy_forward_range_contains(const struct hk_copy_forward_range *outer,
                               const struct hk_copy_forward_range *inner)
{
   if (!hk_copy_forward_range_valid(outer) ||
       !hk_copy_forward_range_valid(inner))
      return false;

   /* If both ranges carry a BO handle, require the same backing BO and
    * matching relative offset within it so aliased VAs never match falsely. */
   if (outer->handle != 0 && inner->handle != 0) {
      if (outer->handle != inner->handle)
         return false;
      const uint64_t o_bo1 = outer->offset + outer->size;
      const uint64_t i_bo1 = inner->offset + inner->size;
      if (inner->offset < outer->offset || i_bo1 > o_bo1)
         return false;
      const uint64_t o_va1 = outer->address + outer->size;
      const uint64_t i_va1 = inner->address + inner->size;
      if (inner->address < outer->address || i_va1 > o_va1)
         return false;
      return (inner->address - outer->address) ==
             (inner->offset - outer->offset);
   }

   /* If only one side has a BO handle, we cannot prove BO identity. */
   if (outer->handle != inner->handle)
      return false;

   const uint64_t o_va1 = outer->address + outer->size;
   const uint64_t i_va1 = inner->address + inner->size;
   return inner->address >= outer->address && i_va1 <= o_va1;
}

void
hk_copy_forward_note_write(struct hk_copy_forward_state *state,
                           const struct hk_copy_forward_range *write_range,
                           bool write_only,
                           uint64_t *elided_tags,
                           uint32_t *elided_count)
{
   if (elided_count)
      *elided_count = 0;

   if (state->active_count == 0)
      return;

   if (!hk_copy_forward_range_valid(write_range)) {
      hk_copy_forward_invalidate_all(state);
      return;
   }

   for (uint32_t i = 0; i < HK_COPY_FORWARD_MAX_ENTRIES; ++i) {
      struct hk_copy_forward_entry *e = &state->entries[i];
      if (!e->active)
         continue;

      if (hk_copy_forward_ranges_overlap(write_range, &e->src))
         e->src_mutated = true;

      if (hk_copy_forward_ranges_overlap(write_range, &e->dst)) {
         if (write_only &&
             hk_copy_forward_range_contains(write_range, &e->dst) &&
             e->unforwarded_reads == 0) {
            append_elided(elided_tags, elided_count, e->tag);
         } else {
            e->unforwarded_reads++;
         }
         deactivate_entry(state, e);
      }
   }
}

bool
hk_copy_forward_record_copy(struct hk_copy_forward_state *state,
                            const struct hk_copy_forward_range *src,
                            const struct hk_copy_forward_range *dst,
                            uint64_t tag,
                            uint64_t *elided_tags,
                            uint32_t *elided_count)
{
   if (elided_count)
      *elided_count = 0;

   if (!hk_copy_forward_range_valid(src) || !hk_copy_forward_range_valid(dst))
      return false;
   if (src->size != dst->size)
      return false;

   /* Reject overlapping src/dst and self-copies; self-copy elision is handled
    * independently at dispatch selection time. */
   if (hk_copy_forward_ranges_overlap(src, dst)) {
      hk_copy_forward_note_readwrite(state, dst);
      return false;
   }

   /* If this copy's `src` reads from a previously tracked copy's `dst`, mark
    * that prior copy as read (do not chain-elide it implicitly). */
   for (uint32_t i = 0; i < HK_COPY_FORWARD_MAX_ENTRIES; ++i) {
      struct hk_copy_forward_entry *e = &state->entries[i];
      if (!e->active)
         continue;
      if (hk_copy_forward_ranges_overlap(src, &e->dst))
         e->unforwarded_reads++;
   }

   /* Writing `dst` completely may prove an earlier dead-scratch copy targeting
    * `dst` elidable. */
   hk_copy_forward_note_write(state, dst, true, elided_tags, elided_count);

   /* Find a free slot; if full, conservatively drop the oldest slot without
    * eliding it. */
   uint32_t slot = HK_COPY_FORWARD_MAX_ENTRIES;
   for (uint32_t i = 0; i < HK_COPY_FORWARD_MAX_ENTRIES; ++i) {
      if (!state->entries[i].active) {
         slot = i;
         break;
      }
   }
   if (slot == HK_COPY_FORWARD_MAX_ENTRIES) {
      deactivate_entry(state, &state->entries[0]);
      slot = 0;
   }

   state->entries[slot] = (struct hk_copy_forward_entry){
      .active = true,
      .src_mutated = false,
      .forwarded_reads = 0,
      .unforwarded_reads = 0,
      .tag = tag,
      .src = *src,
      .dst = *dst,
   };
   state->active_count++;
   return true;
}

bool
hk_copy_forward_resolve_readonly_read(
   struct hk_copy_forward_state *state,
   const struct hk_copy_forward_range *read_range,
   uint64_t *forwarded_va_out)
{
   if (state->active_count == 0 || !hk_copy_forward_range_valid(read_range))
      return false;

   for (uint32_t i = 0; i < HK_COPY_FORWARD_MAX_ENTRIES; ++i) {
      struct hk_copy_forward_entry *e = &state->entries[i];
      if (!e->active)
         continue;
      if (!hk_copy_forward_ranges_overlap(read_range, &e->dst))
         continue;

      if (!e->src_mutated &&
          hk_copy_forward_range_contains(&e->dst, read_range)) {
         const uint64_t rel = read_range->address - e->dst.address;
         e->forwarded_reads++;
         if (forwarded_va_out)
            *forwarded_va_out = e->src.address + rel;
         return true;
      }

      /* Partially overlapping or post-src-mutation read cannot be forwarded;
       * mark `dst` live so the copy cannot be elided. */
      e->unforwarded_reads++;
   }

   return false;
}

void
hk_copy_forward_note_readwrite(struct hk_copy_forward_state *state,
                               const struct hk_copy_forward_range *range)
{
   if (state->active_count == 0)
      return;

   if (!hk_copy_forward_range_valid(range)) {
      hk_copy_forward_invalidate_all(state);
      return;
   }

   for (uint32_t i = 0; i < HK_COPY_FORWARD_MAX_ENTRIES; ++i) {
      struct hk_copy_forward_entry *e = &state->entries[i];
      if (!e->active)
         continue;

      if (hk_copy_forward_ranges_overlap(range, &e->src))
         e->src_mutated = true;

      if (hk_copy_forward_ranges_overlap(range, &e->dst)) {
         e->unforwarded_reads++;
         deactivate_entry(state, e);
      }
   }
}
