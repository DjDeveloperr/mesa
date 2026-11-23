/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"

/*
 * Eliminate tautological bit-freshness tests on monotonically built 32-bit
 * loop-carried bitmasks (shared AGX3B + AGX3C).
 *
 * Matches a counted loop with a 32-bit induction variable `iv` starting at 0,
 * stepping by constant S in {1, 2, 4}, bounded by `iv < K` with K <= 32 (and
 * K + S - 1 <= 32 so `iv + off < 32` for every 0 <= off < S), and a 32-bit
 * loop-carried bitmask `mask` initialized to 0 whose ONLY updates along the
 * backedge chain are conditional or unconditional ORs with `1u << (iv + off)`
 * at strictly increasing offsets `0 <= off < S`.
 *
 * By induction on the iteration number, before each step `off`, `mask` has no
 * bits >= `iv + off` set, so bit `iv + off` is provably 0:
 *   - `(mask | (1u << p)) != mask`  ->  true
 *   - `(mask | (1u << p)) == mask`  ->  false
 *   - `(mask & (1u << p)) == 0`     ->  true
 *   - `(mask & (1u << p)) != 0`     ->  false
 *
 * Folding these conditions removes redundant divergent branches inside the
 * loop body once nir_opt_dead_cf() runs.
 */

#define MAX_CHAIN_STEPS 8

static nir_def *
chase_movs(nir_def *def)
{
   while (def && nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_def_as_alu(def);
      if (alu->op == nir_op_mov && def->num_components == 1 &&
          alu->src[0].src.ssa->num_components == 1) {
         def = alu->src[0].src.ssa;
         continue;
      }
      break;
   }
   return def;
}

static bool
is_const_u32(nir_def *def, uint32_t *val_out)
{
   def = chase_movs(def);
   if (!def || def->num_components != 1 || def->bit_size != 32 ||
       nir_def_instr(def)->type != nir_instr_type_load_const)
      return false;
   *val_out = nir_instr_as_load_const(nir_def_instr(def))->value[0].u32;
   return true;
}

static nir_def *
phi_src_for_block(nir_phi_instr *phi, nir_block *pred)
{
   nir_foreach_phi_src(src, phi) {
      if (src->pred == pred)
         return src->src.ssa;
   }
   return NULL;
}

/* Match `iv + off` or `(iv + off) & 31` where `off` is a constant uint32. */
static bool
match_iv_plus_const(nir_def *def, nir_def *iv, uint32_t *off_out)
{
   def = chase_movs(def);
   if (def == iv) {
      *off_out = 0;
      return true;
   }
   if (nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_def_as_alu(def);

   if (alu->op == nir_op_iand) {
      for (unsigned i = 0; i < 2; ++i) {
         uint32_t c = 0;
         if (is_const_u32(alu->src[i].src.ssa, &c) && c == 31u)
            return match_iv_plus_const(alu->src[1 - i].src.ssa, iv, off_out);
      }
      return false;
   }

   if (alu->op == nir_op_iadd) {
      for (unsigned i = 0; i < 2; ++i) {
         uint32_t c = 0;
         if (chase_movs(alu->src[i].src.ssa) == iv &&
             is_const_u32(alu->src[1 - i].src.ssa, &c)) {
            *off_out = c;
            return true;
         }
      }
   }

   return false;
}

/* Match `1u << (iv + off)` (optionally masked by `& 31`). */
static bool
match_bit_for_iv(nir_def *def, nir_def *iv, uint32_t *off_out)
{
   def = chase_movs(def);
   if (nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_def_as_alu(def);
   if (alu->op != nir_op_ishl)
      return false;
   uint32_t one = 0;
   if (!is_const_u32(alu->src[0].src.ssa, &one) || one != 1u)
      return false;
   return match_iv_plus_const(alu->src[1].src.ssa, iv, off_out);
}

/* Match `ior(prev, bit)` where `bit == 1u << (iv + off)`. */
static bool
match_or_iv_bit(nir_def *def, nir_def *iv, nir_def **prev_out,
                nir_def **bit_out, nir_def **ior_out, uint32_t *off_out)
{
   def = chase_movs(def);
   if (nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_def_as_alu(def);
   if (alu->op != nir_op_ior)
      return false;

   for (unsigned i = 0; i < 2; ++i) {
      uint32_t off = 0;
      nir_def *bit = alu->src[i].src.ssa;
      if (match_bit_for_iv(bit, iv, &off)) {
         *prev_out = chase_movs(alu->src[1 - i].src.ssa);
         *bit_out = bit;
         *ior_out = def;
         *off_out = off;
         return true;
      }
   }
   return false;
}

/* Match one monotone update step producing `val` from an earlier mask `prev`:
 *   - `ior(prev, bit)`
 *   - `bcsel(c, ior(prev, bit), prev)` (either arm order)
 *   - 2-input `phi(arm0, arm1)` where one arm is `prev` and the other arm
 *     recursively matches a single monotone update from `prev` (e.g., an `if`
 *     containing `bcsel(cmp, ior(prev, bit), prev)`).
 */
static bool
match_monotone_step(nir_def *val, nir_def *iv, nir_def **prev_out,
                    nir_def **bit_out, nir_def **ior_out, uint32_t *off_out,
                    unsigned depth)
{
   if (depth > 4 || !val)
      return false;
   val = chase_movs(val);

   if (match_or_iv_bit(val, iv, prev_out, bit_out, ior_out, off_out))
      return true;

   nir_instr *instr = nir_def_instr(val);
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op != nir_op_bcsel)
         return false;
      nir_def *t = chase_movs(alu->src[1].src.ssa);
      nir_def *e = chase_movs(alu->src[2].src.ssa);
      nir_def *p = NULL, *bit = NULL, *ior_d = NULL;
      uint32_t off = 0;
      if (match_or_iv_bit(t, iv, &p, &bit, &ior_d, &off) && p == e) {
         *prev_out = p;
         *bit_out = bit;
         *ior_out = ior_d;
         *off_out = off;
         return true;
      }
      if (match_or_iv_bit(e, iv, &p, &bit, &ior_d, &off) && p == t) {
         *prev_out = p;
         *bit_out = bit;
         *ior_out = ior_d;
         *off_out = off;
         return true;
      }
      return false;
   }

   if (instr->type == nir_instr_type_phi) {
      nir_phi_instr *phi = nir_instr_as_phi(instr);
      nir_def *arms[2] = {NULL, NULL};
      unsigned n = 0;
      nir_foreach_phi_src(s, phi) {
         if (n >= 2)
            return false;
         arms[n++] = chase_movs(s->src.ssa);
      }
      if (n != 2)
         return false;

      for (unsigned i = 0; i < 2; ++i) {
         nir_def *p = NULL, *bit = NULL, *ior_d = NULL;
         uint32_t off = 0;
         if (match_monotone_step(arms[i], iv, &p, &bit, &ior_d, &off,
                                 depth + 1) &&
             p == arms[1 - i]) {
            *prev_out = p;
            *bit_out = bit;
            *ior_out = ior_d;
            *off_out = off;
            return true;
         }
      }
   }

   return false;
}

struct fresh_step {
   nir_def *prev_mask;
   nir_def *bit;
   nir_def *ior_def;
   uint32_t off;
};

static bool
rewrite_fresh_tests(nir_builder *b, const struct fresh_step *st)
{
   bool progress = false;

   /* Rewrite `(prev | bit) != prev` -> true and `(prev | bit) == prev` -> false. */
   nir_foreach_use_safe(src, st->ior_def) {
      nir_instr *user = nir_src_use_instr(src);
      if (user->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *alu = nir_instr_as_alu(user);
      if (alu->op != nir_op_ine && alu->op != nir_op_ieq)
         continue;
      nir_def *s0 = chase_movs(alu->src[0].src.ssa);
      nir_def *s1 = chase_movs(alu->src[1].src.ssa);
      if ((s0 == st->ior_def && s1 == st->prev_mask) ||
          (s1 == st->ior_def && s0 == st->prev_mask)) {
         b->cursor = nir_before_instr(user);
         nir_def *repl =
            (alu->op == nir_op_ine) ? nir_imm_true(b) : nir_imm_false(b);
         nir_def_rewrite_uses(&alu->def, repl);
         progress = true;
      }
   }

   /* Rewrite `(prev & bit) == 0` -> true and `(prev & bit) != 0` -> false. */
   nir_foreach_use_safe(src, st->bit) {
      nir_instr *user = nir_src_use_instr(src);
      if (user->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *and_alu = nir_instr_as_alu(user);
      if (and_alu->op != nir_op_iand)
         continue;
      nir_def *s0 = chase_movs(and_alu->src[0].src.ssa);
      nir_def *s1 = chase_movs(and_alu->src[1].src.ssa);
      if (!((s0 == st->prev_mask && s1 == st->bit) ||
            (s1 == st->prev_mask && s0 == st->bit)))
         continue;

      nir_foreach_use_safe(csrc, &and_alu->def) {
         nir_instr *cuser = nir_src_use_instr(csrc);
         if (cuser->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *cmp = nir_instr_as_alu(cuser);
         if (cmp->op != nir_op_ieq && cmp->op != nir_op_ine)
            continue;
         uint32_t z = 1;
         nir_def *other =
            (chase_movs(cmp->src[0].src.ssa) == &and_alu->def)
               ? cmp->src[1].src.ssa
               : cmp->src[0].src.ssa;
         if (!is_const_u32(other, &z) || z != 0)
            continue;
         b->cursor = nir_before_instr(cuser);
         nir_def *repl =
            (cmp->op == nir_op_ieq) ? nir_imm_true(b) : nir_imm_false(b);
         nir_def_rewrite_uses(&cmp->def, repl);
         progress = true;
      }
   }

   return progress;
}

static bool
optimize_loop(nir_builder *b, nir_loop *loop)
{
   if (nir_loop_has_continue_construct(loop))
      return false;

   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
   nir_block *hdr = nir_loop_first_block(loop);
   nir_block *latch = nir_loop_last_block(loop);
   if (util_dynarray_num_elements(&hdr->predecessors, nir_block *) != 2)
      return false;

   /* Find top-tested exit `if (iv >= K) break;` with constant K <= 32. */
   nir_cf_node *after_hdr = nir_cf_node_next(&hdr->cf_node);
   if (!after_hdr || after_hdr->type != nir_cf_node_if)
      return false;
   nir_if *exit_if = nir_cf_node_as_if(after_hdr);
   nir_block *etb = nir_if_first_then_block(exit_if);
   nir_block *eeb = nir_if_first_else_block(exit_if);
   if (etb != nir_if_last_then_block(exit_if) ||
       eeb != nir_if_last_else_block(exit_if) ||
       !exec_list_is_empty(&eeb->instr_list))
      return false;
   nir_instr *brk = nir_block_last_instr(etb);
   if (!brk || brk->type != nir_instr_type_jump ||
       nir_instr_as_jump(brk)->type != nir_jump_break ||
       nir_block_first_instr(etb) != brk)
      return false;

   nir_def *ec = chase_movs(exit_if->condition.ssa);
   if (nir_def_instr(ec)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *ecmp = nir_def_as_alu(ec);
   if (ecmp->op != nir_op_ige && ecmp->op != nir_op_uge)
      return false;

   nir_def *iv_def = chase_movs(ecmp->src[0].src.ssa);
   uint32_t bound_k = 0;
   if (!is_const_u32(ecmp->src[1].src.ssa, &bound_k) ||
       bound_k == 0 || bound_k > 32)
      return false;

   /* Verify iv_def is a header phi initialized to 0 with constant step S in
    * {1, 2, 4}, and that every in-body bit position `iv + off < 32`. */
   if (nir_def_instr(iv_def)->type != nir_instr_type_phi ||
       nir_def_block(iv_def) != hdr)
      return false;
   nir_phi_instr *iv_phi = nir_instr_as_phi(nir_def_instr(iv_def));

   uint32_t iv_init = 1;
   if (!is_const_u32(phi_src_for_block(iv_phi, pre), &iv_init) || iv_init != 0)
      return false;

   nir_def *iv_next = phi_src_for_block(iv_phi, latch);
   uint32_t step_s = 0;
   if (!iv_next || !match_iv_plus_const(iv_next, iv_def, &step_s))
      return false;
   if (step_s != 1 && step_s != 2 && step_s != 4)
      return false;
   /* Largest possible `iv` entering the body is `((bound_k - 1) / step_s) * step_s`.
    * Ensure `max_iv + (step_s - 1) < 32` so bit indices never wrap mod 32. */
   uint32_t max_iv = ((bound_k - 1) / step_s) * step_s;
   if (max_iv + (step_s - 1) >= 32)
      return false;

   bool progress = false;

   /* Inspect each other 32-bit phi initialized to 0 for a monotone bitmask chain. */
   nir_foreach_phi(mask_phi, hdr) {
      if (mask_phi == iv_phi || mask_phi->def.num_components != 1 ||
          mask_phi->def.bit_size != 32)
         continue;
      uint32_t m_init = 1;
      if (!is_const_u32(phi_src_for_block(mask_phi, pre), &m_init) ||
          m_init != 0)
         continue;

      nir_def *cur = phi_src_for_block(mask_phi, latch);
      if (!cur)
         continue;

      struct fresh_step rev[MAX_CHAIN_STEPS];
      unsigned nsteps = 0;
      bool chain_ok = true;
      while (chase_movs(cur) != &mask_phi->def) {
         if (nsteps >= MAX_CHAIN_STEPS) {
            chain_ok = false;
            break;
         }
         nir_def *prev = NULL, *bit = NULL, *ior_d = NULL;
         uint32_t off = 0;
         if (!match_monotone_step(cur, iv_def, &prev, &bit, &ior_d, &off, 0)) {
            chain_ok = false;
            break;
         }
         rev[nsteps++] = (struct fresh_step){
            .prev_mask = prev,
            .bit = bit,
            .ior_def = ior_d,
            .off = off,
         };
         cur = prev;
      }

      if (!chain_ok || nsteps == 0)
         continue;

      /* Verify offsets strictly increase in execution order and stay < step_s
       * so each iteration sets strictly higher bit positions than all earlier
       * steps and all earlier iterations. */
      uint32_t min_fresh = 0;
      bool order_ok = true;
      for (unsigned i = 0; i < nsteps; ++i) {
         const struct fresh_step *st = &rev[nsteps - 1 - i];
         if (st->off < min_fresh || st->off >= step_s) {
            order_ok = false;
            break;
         }
         min_fresh = st->off + 1;
      }
      if (!order_ok)
         continue;

      for (unsigned i = 0; i < nsteps; ++i)
         progress |= rewrite_fresh_tests(b, &rev[nsteps - 1 - i]);
   }

   return progress;
}

static bool
scan_cf_list(nir_builder *b, struct exec_list *cf_list)
{
   bool progress = false;
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= scan_cf_list(b, &nif->then_list);
         progress |= scan_cf_list(b, &nif->else_list);
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= optimize_loop(b, loop);
         progress |= scan_cf_list(b, &loop->body);
      }
   }
   return progress;
}

bool
agx_nir_opt_monotonic_bitmask(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);
      bool func_progress = scan_cf_list(&b, &impl->body);
      progress |= nir_progress(func_progress, impl, nir_metadata_control_flow);
   }
   return progress;
}
