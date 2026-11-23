/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_slot_core.h"

#include <string.h>

/* Active writer slot for token r, or 0 if retired or synchronously written. */
static inline uint8_t
slot_reg_current(const uint8_t *reg_slot, const uint16_t *reg_gen,
                 const uint16_t *slot_gen, uint16_t r)
{
   const uint8_t s = reg_slot[r];
   return (s && reg_gen[r] == slot_gen[s]) ? s : 0;
}

static inline void
slot_clear_writer(uint8_t *reg_slot, const uint16_t *reg_gen,
                  const uint16_t *slot_gen, unsigned *slot_writer_count,
                  uint16_t r)
{
   const uint8_t s = reg_slot[r];
   if (s) {
      if (reg_gen[r] == slot_gen[s])
         slot_writer_count[s]--;
      reg_slot[r] = 0;
   }
}

static inline void
slot_set_writer(uint8_t *reg_slot, uint16_t *reg_gen,
                const uint16_t *slot_gen, unsigned *slot_writer_count,
                uint16_t r, uint8_t slot)
{
   slot_clear_writer(reg_slot, reg_gen, slot_gen, slot_writer_count, r);
   reg_slot[r] = slot;
   reg_gen[r] = slot_gen[slot];
   slot_writer_count[slot]++;
}

static void
slot_retire_writers(uint8_t *reg_slot, uint16_t *slot_gen,
                    unsigned *slot_writer_count, unsigned slot,
                    unsigned token_count)
{
   slot_writer_count[slot] = 0;
   if (++slot_gen[slot] == 0) {
      /* Wraparound guard after 65,535 generations on one slot. */
      for (unsigned r = 0; r < token_count; ++r) {
         if (reg_slot[r] == slot)
            reg_slot[r] = 0;
      }
      slot_gen[slot] = 1;
   }
}

static void
slot_retire_readers(uint16_t *reader_slots, uint16_t *reader_list,
                    unsigned *reader_list_n, bool *reader_list_overflow,
                    uint16_t *active_reader_slots, uint16_t wait_mask,
                    unsigned token_count)
{
   if (!(wait_mask & *active_reader_slots))
      return;

   *active_reader_slots &= (uint16_t)~wait_mask;
   if (!*reader_list_overflow) {
      unsigned dst = 0;
      for (unsigned k = 0; k < *reader_list_n; ++k) {
         const uint16_t r = reader_list[k];
         reader_slots[r] &= (uint16_t)~wait_mask;
         if (reader_slots[r])
            reader_list[dst++] = r;
      }
      *reader_list_n = dst;
   } else {
      for (unsigned r = 0; r < token_count; ++r)
         reader_slots[r] &= (uint16_t)~wait_mask;
      if (*active_reader_slots == 0) {
         *reader_list_n = 0;
         *reader_list_overflow = false;
      }
   }
}

/* Most regions touch only GPRs. Do not scan the entire static TPS address
 * space every time a scoreboard generation retires. Every writer alias and
 * payload reader below comes from a source or destination in this region,
 * so tokens beyond this bound remain empty throughout the assignment.
 *
 * Leave malformed-input handling to the assignment pass, including which
 * prefix of the output it has written when it rejects an instruction. */
static unsigned
agx3_slot_token_count(const struct agx3_slot_instr *instrs, unsigned n)
{
   unsigned count = 1;
   for (unsigned i = 0; i < n; ++i) {
      const struct agx3_slot_instr *I = &instrs[i];
      if (I->nr_srcs > AGX3_SLOT_MAX_REGS || I->nr_dsts > AGX3_SLOT_MAX_REGS)
         return AGX3_SLOT_NUM_TOKENS;

      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         if (I->src[s] >= AGX3_SLOT_NUM_TOKENS)
            return AGX3_SLOT_NUM_TOKENS;
         if (I->src[s] >= count)
            count = I->src[s] + 1u;
      }
      for (unsigned d = 0; d < I->nr_dsts; ++d) {
         if (I->dst[d] >= AGX3_SLOT_NUM_TOKENS)
            return AGX3_SLOT_NUM_TOKENS;
         if (I->dst[d] >= count)
            count = I->dst[d] + 1u;
      }
   }
   return count;
}

static bool
agx3_slot_assign_initial(const struct agx3_slot_instr *instrs, unsigned n,
                            unsigned nslots, const struct agx3_slot_entry *entry,
                            unsigned entry_count,
                            const struct agx3_slot_live_range *live,
                            unsigned live_count, struct agx3_slot_result *out)
{
   if (instrs == NULL || out == NULL || nslots < 1 || nslots > 8 ||
       (entry_count && entry == NULL) || entry_count > AGX3_SLOT_NUM_TOKENS ||
       (live_count && !live))
      return false;

   unsigned token_count = agx3_slot_token_count(instrs, n);
   for (unsigned i = 0; i < entry_count; ++i) {
      if (entry[i].token >= AGX3_SLOT_NUM_TOKENS ||
          entry[i].slot < 1 || entry[i].slot > nslots)
         return false;
      if (entry[i].token >= token_count)
         token_count = entry[i].token + 1u;
   }

   for (unsigned i = 0; i < live_count; ++i) {
      const struct agx3_slot_live_range *range = &live[i];
      if (!range->count || !range->slot || range->slot > nslots ||
          range->first >= AGX3_SLOT_NUM_TOKENS ||
          range->count > AGX3_SLOT_NUM_TOKENS - range->first)
         return false;
      if (range->first + range->count > token_count)
         token_count = range->first + range->count;
   }

   /* reg_slot[r] = slot (1..nslots) of the async op whose result currently
    * lives in half-register r, valid iff reg_gen[r] == slot_gen[reg_slot[r]],
    * or 0 if r was last written by a synchronous op (or never). */
   uint8_t reg_slot[AGX3_SLOT_NUM_TOKENS];
   uint16_t reg_gen[AGX3_SLOT_NUM_TOKENS];
   uint16_t slot_gen[9] = {0, 1, 1, 1, 1, 1, 1, 1, 1};
   unsigned slot_writer_count[9] = {0};
   memset(reg_slot, 0, token_count * sizeof(reg_slot[0]));
   uint16_t pending = 0;
   for (unsigned i = 0; i < entry_count; ++i) {
      /* One token cannot name two independently pending generations. */
      if (reg_slot[entry[i].token])
         return false;
      slot_set_writer(reg_slot, reg_gen, slot_gen, slot_writer_count,
                      entry[i].token, entry[i].slot);
      pending |= (uint16_t)(1u << (entry[i].slot - 1));
   }
   for (unsigned i = 0; i < live_count; ++i) {
      const struct agx3_slot_live_range *range = &live[i];
      for (unsigned r = range->first; r < range->first + range->count; ++r) {
         if (reg_slot[r])
            return false;
         slot_set_writer(reg_slot, reg_gen, slot_gen, slot_writer_count, r,
                         range->slot);
      }
      pending |= (uint16_t)(1u << (range->slot - 1));
   }
   uint16_t entry_pending = pending;

   /* reader_slots[r] = the slots of the in-flight stores that latch
    * half-register r as payload. A store has no destination, so the result
    * bookkeeping above never covers its payload: the transform feedback copy
    * program reuses one payload quad for all sixteen captured varyings, and
    * the next varying's setup raced the outstanding store.
    *
    * reader_list tracks the tokens with nonzero reader_slots[r] so retiring
    * slots does not scan token_count; if more than 64 distinct tokens are
    * latched simultaneously, retirement falls back to a full scan until all
    * active reader slots clear. */
   uint16_t reader_slots[AGX3_SLOT_NUM_TOKENS];
   uint16_t reader_list[64];
   unsigned reader_list_n = 0;
   bool reader_list_overflow = false;
   memset(reader_slots, 0, token_count * sizeof(reader_slots[0]));
   uint16_t active_reader_slots = 0;

   /* slot_waited[s] = the slot has been synchronised since it was last
    * allocated, so later consumers need not wait it again. */
   bool slot_waited[9] = {false};

   /* slot_reissuable[s] = the slot's outstanding generation was issued by an
    * instruction that reissues live slots itself. */
   bool slot_reissuable[9] = {false};
   uint8_t slot_join_class[9] = {0};
   uint8_t slot_join_count[9] = {0};

   unsigned next = 0; /* round-robin cursor into 0..nslots-1 */

   for (unsigned i = 0; i < n; ++i) {
      const struct agx3_slot_instr *I = &instrs[i];
      if (I->nr_srcs > AGX3_SLOT_MAX_REGS || I->nr_dsts > AGX3_SLOT_MAX_REGS)
         return false;
      if (I->preferred_slot > nslots || (!I->is_async && I->preferred_slot))
         return false;
      if (I->coalesce_src > I->nr_srcs || (!I->is_async && I->coalesce_src))
         return false;
      if ((I->join_class && (!I->is_async || !I->join_limit || I->join_limit > 8)) ||
          (!I->join_class && (I->join_limit || I->prefer_join)))
         return false;
      const uint16_t valid_slots = (uint16_t)((1u << nslots) - 1u);
      if ((I->nr_srcs < 32 && (I->late_src_mask >> I->nr_srcs)) ||
          (!I->is_async && I->late_src_mask))
         return false;
      if ((I->transient_slots & ~valid_slots) ||
          (I->preferred_slot &&
           (I->transient_slots & (1u << (I->preferred_slot - 1)))))
         return false;

      uint16_t wait_mask = 0;
      if (I->drain) {
         wait_mask = pending;
         for (unsigned slot = 1; slot <= nslots; ++slot)
            if (pending & (1u << (slot - 1)))
               slot_waited[slot] = true;
      }
      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         const uint16_t r = I->src[s];
         if (r >= AGX3_SLOT_NUM_TOKENS)
            return false;
         const uint8_t slot = slot_reg_current(reg_slot, reg_gen, slot_gen, r);
         if (slot != 0 && !slot_waited[slot]) {
            wait_mask |= (uint16_t)(1u << (slot - 1));
            slot_waited[slot] = true;
         }
      }

      /*
       * Write-after-write. An async result that is still in flight for this
       * register lands LATER than this instruction, so a synchronous write
       * here is silently overwritten by the load when it retires -- and any
       * consumer that waits the slot then reads the loaded word instead of
       * the computed one. Mesa's own AGX2 pass checks this case explicitly
       * (agx_insert_waits.c, "Check for write-after-write"); omitting it here
       * produced no diagnostic, just wrong values on hardware.
       *
       * This runs before the bookkeeping below, so reg_slot[] still names the
       * previous writer rather than this instruction.
       */
      for (unsigned d = 0; d < I->nr_dsts; ++d) {
         const uint16_t r = I->dst[d];
         if (r >= AGX3_SLOT_NUM_TOKENS)
            return false;
         const uint8_t slot = slot_reg_current(reg_slot, reg_gen, slot_gen, r);
         if (slot != 0 && !slot_waited[slot]) {
            wait_mask |= (uint16_t)(1u << (slot - 1));
            slot_waited[slot] = true;
         }

         /* Write-after-read: retire every async op still reading this
          * register before redefining it. */
         uint16_t readers = reader_slots[r];
         while (readers) {
            const unsigned bit = (unsigned)(readers & (~readers + 1u));
            const uint8_t reader = (uint8_t)(__builtin_ctz(bit) + 1);
            readers &= (uint16_t)~bit;
            if (!slot_waited[reader]) {
               wait_mask |= (uint16_t)bit;
               slot_waited[reader] = true;
            }
         }
      }

      /* A compound instruction may use a scoreboard slot internally, for
       * example PACK_SMPIR/COPY producers followed by a TEX which waits for
       * them. Retire a prior generation before reusing the slot, then forget
       * its aliases: the prewait has made those GPR values synchronous and
       * the internal generation is consumed before this logical instruction
       * returns. */
      for (unsigned slot = 1; slot <= nslots; ++slot) {
         const uint16_t bit = (uint16_t)(1u << (slot - 1));
         if (!(I->transient_slots & bit))
            continue;

         const bool live = slot_writer_count[slot] != 0 ||
                           (active_reader_slots & bit) != 0;
         if (slot_writer_count[slot] != 0)
            slot_retire_writers(reg_slot, slot_gen, slot_writer_count, slot,
                                token_count);
         if (live && !slot_waited[slot])
            wait_mask |= bit;
         slot_waited[slot] = true;
         pending &= (uint16_t)~bit;
      }

      /* Every slot this instruction waited is now retired, so the registers
       * its async ops were reading are free again. */
      slot_retire_readers(reader_slots, reader_list, &reader_list_n,
                          &reader_list_overflow, &active_reader_slots,
                          wait_mask, token_count);

      if (I->is_async) {
         /*
          * A slot names one generation at a time. Issuing it while an earlier
          * generation is still outstanding for a live result leaves that
          * result's consumer waiting on the new generation instead, so it can
          * read its register before the older operation has written it. The
          * round-robin allocation therefore skips such slots, and when every
          * slot is in that state it waits the old generation out first.
          *
          * A preferred slot keeps its witnessed protocol: Apple's fragment
          * tile loads, for one, issue slot 6 again for each tile read. So
          * does an instruction that reissues live slots, such as a fragment
          * coefficient load, which has no wait field: it only skips a slot
          * whose live generation some other kind of instruction issued, and
          * when every slot is in that state it takes the next one as before.
          */
         uint8_t slot = I->preferred_slot;
         const unsigned cap =
            I->max_slot && I->max_slot < nslots ? I->max_slot : nslots;
         if (!slot) {
            if (I->coalesce_src) {
               const unsigned source = I->src[I->coalesce_src - 1];
               const uint8_t producer_slot =
                  slot_reg_current(reg_slot, reg_gen, slot_gen, source);
               if (producer_slot && producer_slot <= cap &&
                   slot_waited[producer_slot] &&
                   !(I->transient_slots & (1u << (producer_slot - 1))))
                  slot = producer_slot;
            }
            if (I->prefer_join && !slot) {
               for (unsigned t = 0; t < nslots && !slot; ++t) {
                  const uint8_t candidate = (uint8_t)((next + t) % nslots + 1);
                  if (candidate <= cap && !slot_waited[candidate] &&
                      slot_join_class[candidate] == I->join_class &&
                      slot_join_count[candidate] < I->join_limit)
                     slot = candidate;
               }
            }
            for (unsigned t = 0; t < nslots && !slot; ++t) {
               const uint8_t candidate = (uint8_t)((next + t) % nslots + 1);
               if (candidate > cap)
                  continue;
               const uint16_t cbit = (uint16_t)(1u << (candidate - 1));
               if (slot_waited[candidate] ||
                   (slot_writer_count[candidate] == 0 &&
                    !(active_reader_slots & cbit)) ||
                   (I->reissues_live_slot && slot_reissuable[candidate]) ||
                   (I->join_class && slot_join_class[candidate] == I->join_class &&
                    slot_join_count[candidate] < I->join_limit))
                  slot = candidate;
            }
            if (!slot) {
               slot = (uint8_t)(next + 1);
               if (slot > cap)
                  slot = 1;
               if (!I->reissues_live_slot) {
                  wait_mask |= (uint16_t)(1u << (slot - 1));
                  slot_waited[slot] = true;
               }
            }
            next = slot % nslots;
         }

         /* A preferred/reissuable native instruction may normally join an
          * in-region generation. The external loader's generation is not
          * part of that protocol: retire it before any slot reuse. */
         if ((entry_pending & (1u << (slot - 1))) && !slot_waited[slot]) {
            wait_mask |= (uint16_t)(1u << (slot - 1));
            slot_waited[slot] = true;
         }

         /* A joined queue must complete before an incompatible producer
          * replaces it. In particular, an SCS completion or a texture
          * publication must not inherit a pending device-read generation. */
         if (!slot_waited[slot] && slot_join_count[slot] &&
             ((slot_join_class[slot] != I->join_class) ||
              (I->join_class && slot_join_count[slot] >= I->join_limit))) {
            wait_mask |= (uint16_t)(1u << (slot - 1));
            slot_waited[slot] = true;
         }

         out[i].wait_mask = wait_mask;
         out[i].slot = slot;
         entry_pending &= (uint16_t)~wait_mask;
         pending &= (uint16_t)~wait_mask;
         pending |= (uint16_t)(1u << (slot - 1));

         /* Allocation can add a wait when every generation is occupied.
          * Retire its payload readers before recording this instruction. */
         slot_retire_readers(reader_slots, reader_list, &reader_list_n,
                             &reader_list_overflow, &active_reader_slots,
                             wait_mask, token_count);

         /* Reusing an already-synchronized slot starts a new dependency
          * generation. Retire the old generation's register aliases before
          * publishing the new result, otherwise an unrelated later write can
          * consume the new generation's wait and hide it from its consumer. */
         const bool new_join = slot_waited[slot] ||
                               slot_join_class[slot] != I->join_class;
         if (slot_waited[slot] && slot_writer_count[slot] != 0) {
            slot_retire_writers(reg_slot, slot_gen, slot_writer_count, slot,
                                token_count);
         }
         slot_waited[slot] = false; /* freshly outstanding */
         slot_reissuable[slot] = I->reissues_live_slot;
         slot_join_count[slot] = I->join_class ?
            (new_join ? 1 : slot_join_count[slot] + 1) : 0;
         slot_join_class[slot] = I->join_class;
         for (unsigned d = 0; d < I->nr_dsts; ++d) {
            if (I->dst[d] >= AGX3_SLOT_NUM_TOKENS)
               return false;
            slot_set_writer(reg_slot, reg_gen, slot_gen, slot_writer_count,
                            I->dst[d], slot);
         }
         if ((I->latches_srcs || I->late_src_mask) && I->nr_srcs) {
            const uint16_t bit = (uint16_t)(1u << (slot - 1));
            active_reader_slots |= bit;
            for (unsigned s = 0; s < I->nr_srcs; ++s) {
               if (I->latches_srcs || (I->late_src_mask & (UINT32_C(1) << s))) {
                  const uint16_t r = I->src[s];
                  if (reader_slots[r] == 0) {
                     if (reader_list_n <
                         sizeof(reader_list) / sizeof(reader_list[0]))
                        reader_list[reader_list_n++] = r;
                     else
                        reader_list_overflow = true;
                  }
                  reader_slots[r] |= bit;
               }
            }
         }
      } else {
         out[i].wait_mask = wait_mask;
         out[i].slot = 0;
         entry_pending &= (uint16_t)~wait_mask;
         pending &= (uint16_t)~wait_mask;
         for (unsigned d = 0; d < I->nr_dsts; ++d) {
            if (I->dst[d] >= AGX3_SLOT_NUM_TOKENS)
               return false;
            slot_clear_writer(reg_slot, reg_gen, slot_gen, slot_writer_count,
                              I->dst[d]);
         }
      }
   }

   return true;
}

bool
agx3_slot_assign_with_entry(const struct agx3_slot_instr *instrs, unsigned n,
                            unsigned nslots, const struct agx3_slot_entry *entry,
                            unsigned entry_count, struct agx3_slot_result *out)
{
   return agx3_slot_assign_initial(instrs, n, nslots, entry, entry_count,
                                   NULL, 0, out);
}

bool
agx3_slot_assign_with_live(const struct agx3_slot_instr *instrs, unsigned n,
                           unsigned nslots,
                           const struct agx3_slot_live_range *live,
                           unsigned live_count, struct agx3_slot_result *out)
{
   return agx3_slot_assign_initial(instrs, n, nslots, NULL, 0,
                                   live, live_count, out);
}

bool
agx3_slot_assign(const struct agx3_slot_instr *instrs, unsigned n,
                 unsigned nslots, struct agx3_slot_result *out)
{
   return agx3_slot_assign_with_entry(instrs, n, nslots, NULL, 0, out);
}
