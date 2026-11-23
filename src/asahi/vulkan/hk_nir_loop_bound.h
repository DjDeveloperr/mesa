/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <string.h>
#include "compiler/nir/nir.h"
#include "hk_pack_loop_bound.h"

struct hk_nir_loop_param_binding {
   nir_loop *loop;
   enum hk_pack_loop_param param;
};

#define HK_NIR_LOOP_MAX_BINDINGS 16u

struct hk_nir_loop_binding_list {
   struct hk_nir_loop_param_binding entries[HK_NIR_LOOP_MAX_BINDINGS];
   unsigned count;
   bool overflow;
};

static inline void
hk_nir_loop_binding_add(struct hk_nir_loop_binding_list *list,
                        nir_loop *loop, enum hk_pack_loop_param param)
{
   if (!list || !loop)
      return;
   if (list->count >= HK_NIR_LOOP_MAX_BINDINGS) {
      list->overflow = true;
      return;
   }
   list->entries[list->count++] = (struct hk_nir_loop_param_binding){
      .loop = loop,
      .param = param,
   };
}

struct hk_nir_lb_builder {
   struct hk_pack_loop_bound *out;
   const struct hk_nir_loop_param_binding *bindings;
   unsigned num_bindings;
   bool failed;
};

static inline int
hk_nir_lb_emit(struct hk_nir_lb_builder *b, struct hk_pack_loop_op op)
{
   if (b->failed)
      return -1;
   for (uint8_t i = 0; i < b->out->count; ++i) {
      const struct hk_pack_loop_op *e = &b->out->ops[i];
      if (e->kind == op.kind && e->lhs == op.lhs && e->rhs == op.rhs &&
          e->imm == op.imm)
         return (int)i;
   }
   if (b->out->count >= HK_PACK_LOOP_MAX_OPS) {
      b->failed = true;
      return -1;
   }
   const int idx = (int)b->out->count++;
   b->out->ops[idx] = op;
   return idx;
}

static inline int
hk_nir_lb_const(struct hk_nir_lb_builder *b, uint32_t imm)
{
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_CONST, .imm = imm,
   });
}

static inline int
hk_nir_lb_param(struct hk_nir_lb_builder *b, enum hk_pack_loop_param param)
{
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_PARAM, .imm = (uint32_t)param,
   });
}

static inline bool
hk_nir_lb_is_const(const struct hk_nir_lb_builder *b, int idx, uint32_t *imm_out)
{
   if (idx < 0 || b->failed)
      return false;
   const struct hk_pack_loop_op op = b->out->ops[idx];
   if (op.kind != HK_PACK_LOOP_OP_CONST)
      return false;
   if (imm_out)
      *imm_out = op.imm;
   return true;
}

static inline int
hk_nir_lb_add(struct hk_nir_lb_builder *b, int lhs, int rhs)
{
   if (b->failed || lhs < 0 || rhs < 0)
      return -1;
   uint32_t a = 0, c = 0;
   const bool ca = hk_nir_lb_is_const(b, lhs, &a);
   const bool cc = hk_nir_lb_is_const(b, rhs, &c);
   if (ca && a == 0)
      return rhs;
   if (cc && c == 0)
      return lhs;
   if (ca && cc && (uint64_t)a + (uint64_t)c <= UINT32_MAX)
      return hk_nir_lb_const(b, a + c);
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_ADD, .lhs = (uint8_t)lhs, .rhs = (uint8_t)rhs,
   });
}

static inline int
hk_nir_lb_mul(struct hk_nir_lb_builder *b, int lhs, int rhs)
{
   if (b->failed || lhs < 0 || rhs < 0)
      return -1;
   uint32_t a = 0, c = 0;
   const bool ca = hk_nir_lb_is_const(b, lhs, &a);
   const bool cc = hk_nir_lb_is_const(b, rhs, &c);
   if ((ca && a == 0) || (cc && c == 0))
      return hk_nir_lb_const(b, 0);
   if (ca && a == 1)
      return rhs;
   if (cc && c == 1)
      return lhs;
   if (ca && cc && (uint64_t)a * (uint64_t)c <= UINT32_MAX)
      return hk_nir_lb_const(b, a * c);
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_MUL, .lhs = (uint8_t)lhs, .rhs = (uint8_t)rhs,
   });
}

static inline int
hk_nir_lb_max(struct hk_nir_lb_builder *b, int lhs, int rhs)
{
   if (b->failed || lhs < 0 || rhs < 0)
      return -1;
   if (lhs == rhs)
      return lhs;
   uint32_t a = 0, c = 0;
   const bool ca = hk_nir_lb_is_const(b, lhs, &a);
   const bool cc = hk_nir_lb_is_const(b, rhs, &c);
   if (ca && a == 0)
      return rhs;
   if (cc && c == 0)
      return lhs;
   if (ca && cc)
      return hk_nir_lb_const(b, a >= c ? a : c);
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_MAX, .lhs = (uint8_t)lhs, .rhs = (uint8_t)rhs,
   });
}

static inline bool
hk_nir_lb_is_simple_break_if(nir_if *nif)
{
   if (exec_list_is_empty(&nif->then_list) ||
       exec_list_get_head(&nif->then_list) !=
          exec_list_get_tail(&nif->then_list))
      return false;
   nir_cf_node *tn =
      exec_node_data(nir_cf_node, exec_list_get_head(&nif->then_list), node);
   if (tn->type != nir_cf_node_block)
      return false;
   nir_block *tb = nir_cf_node_as_block(tn);
   if (exec_list_is_empty(&tb->instr_list) ||
       exec_list_get_head(&tb->instr_list) !=
          exec_list_get_tail(&tb->instr_list))
      return false;
   nir_instr *ti =
      exec_node_data(nir_instr, exec_list_get_head(&tb->instr_list), node);
   if (ti->type != nir_instr_type_jump ||
       nir_instr_as_jump(ti)->type != nir_jump_break)
      return false;

   if (exec_list_is_empty(&nif->else_list) ||
       exec_list_get_head(&nif->else_list) !=
          exec_list_get_tail(&nif->else_list))
      return false;
   nir_cf_node *en =
      exec_node_data(nir_cf_node, exec_list_get_head(&nif->else_list), node);
   if (en->type != nir_cf_node_block)
      return false;
   return exec_list_is_empty(&nir_cf_node_as_block(en)->instr_list);
}

static inline int
hk_nir_lb_loop_trips(struct hk_nir_lb_builder *b, nir_loop *loop)
{
   nir_cf_node *prev_cf = nir_cf_node_prev(&loop->cf_node);
   if (!prev_cf || prev_cf->type != nir_cf_node_block) {
      b->failed = true;
      return -1;
   }
   nir_block *pre = nir_cf_node_as_block(prev_cf);
   if (exec_list_is_empty(&loop->body)) {
      b->failed = true;
      return -1;
   }
   nir_cf_node *first_cf =
      exec_node_data(nir_cf_node, exec_list_get_head(&loop->body), node);
   if (first_cf->type != nir_cf_node_block) {
      b->failed = true;
      return -1;
   }
   nir_block *hdr = nir_cf_node_as_block(first_cf);
   nir_cf_node *second_cf = nir_cf_node_next(first_cf);
   if (!second_cf || second_cf->type != nir_cf_node_if) {
      b->failed = true;
      return -1;
   }
   nir_if *brk_if = nir_cf_node_as_if(second_cf);
   if (!hk_nir_lb_is_simple_break_if(brk_if)) {
      b->failed = true;
      return -1;
   }

   /* If the break condition is a disjunction (A || B), breaking occurs at least
    * as soon as either disjunct becomes true, so proving a trip bound from any
    * disjunct is a sound upper bound. */
   const nir_def *cands[2] = { brk_if->condition.ssa, NULL };
   unsigned num_cands = 1;
   if (cands[0] && nir_def_instr(cands[0])->type == nir_instr_type_alu) {
      nir_alu_instr *top = nir_instr_as_alu(nir_def_instr(cands[0]));
      if (top->op == nir_op_ior) {
         cands[0] = top->src[0].src.ssa;
         cands[1] = top->src[1].src.ssa;
         num_cands = 2;
      }
   }

   bool found_canonical = false;
   bool found_const = false;
   uint64_t best_const_trips = UINT64_MAX;
   for (unsigned ci = 0; ci < num_cands; ++ci) {
      const nir_def *cond_def = cands[ci];
      if (!cond_def || nir_def_instr(cond_def)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *cmp = nir_instr_as_alu(nir_def_instr(cond_def));
      if (cmp->op != nir_op_ige && cmp->op != nir_op_uge)
         continue;

      const nir_def *iv_def = cmp->src[0].src.ssa;
      if (!iv_def || iv_def->num_components != 1 || iv_def->bit_size != 32 ||
          nir_def_instr(iv_def)->type != nir_instr_type_phi ||
          nir_def_instr(iv_def)->block != hdr)
         continue;

      nir_phi_instr *iv = nir_instr_as_phi(nir_def_instr(iv_def));
      const nir_src *init_src = NULL;
      const nir_src *back_src = NULL;
      unsigned src_count = 0;
      nir_foreach_phi_src(ps, iv) {
         src_count++;
         if (ps->pred == pre)
            init_src = &ps->src;
         else
            back_src = &ps->src;
      }
      if (src_count != 2 || !init_src || !back_src ||
          nir_def_instr(back_src->ssa)->type != nir_instr_type_alu)
         continue;

      nir_alu_instr *step_alu = nir_instr_as_alu(nir_def_instr(back_src->ssa));
      if (step_alu->op != nir_op_iadd)
         continue;
      const nir_src *step_src = NULL;
      if (step_alu->src[0].src.ssa == iv_def)
         step_src = &step_alu->src[1].src;
      else if (step_alu->src[1].src.ssa == iv_def)
         step_src = &step_alu->src[0].src;
      else
         continue;

      found_canonical = true;
      const nir_src *limit_src = &cmp->src[1].src;
      if (nir_src_is_const(*init_src) && nir_src_is_const(*step_src) &&
          nir_src_is_const(*limit_src)) {
         uint64_t trips = 0;
         bool ok = true;
         if (cmp->op == nir_op_ige) {
            const int64_t init_i = nir_src_as_int(*init_src);
            const int64_t step_i = nir_src_as_int(*step_src);
            const int64_t limit_i = nir_src_as_int(*limit_src);
            if (step_i <= 0)
               ok = false;
            else if (limit_i > init_i)
               trips = ((uint64_t)(limit_i - init_i) + (uint64_t)step_i - 1ull) / (uint64_t)step_i;
         } else {
            const uint64_t init_u = nir_src_as_uint(*init_src);
            const uint64_t step_u = nir_src_as_uint(*step_src);
            const uint64_t limit_u = nir_src_as_uint(*limit_src);
            if (step_u == 0)
               ok = false;
            else if (limit_u > init_u) {
               const uint64_t span = limit_u - init_u;
               if (span > UINT64_MAX - (step_u - 1ull))
                  ok = false;
               else
                  trips = (span + step_u - 1ull) / step_u;
            }
         }
         if (ok && trips <= UINT32_MAX && trips < best_const_trips) {
            best_const_trips = trips;
            found_const = true;
         }
      }
   }

   if (!found_canonical) {
      b->failed = true;
      return -1;
   }

   /* Check explicit dynamic parameter binding first (structure already verified). */
   for (unsigned i = 0; i < b->num_bindings; ++i) {
      if (b->bindings[i].loop == loop)
         return hk_nir_lb_param(b, b->bindings[i].param);
   }

   if (!found_const) {
      b->failed = true;
      return -1;
   }
   return hk_nir_lb_const(b, (uint32_t)best_const_trips);
}

static inline int
hk_nir_lb_walk_list(struct hk_nir_lb_builder *b, const struct exec_list *cf_list)
{
   int total = hk_nir_lb_const(b, 0);
   foreach_list_typed(nir_cf_node, cf, node, cf_list) {
      if (b->failed)
         return -1;
      if (cf->type == nir_cf_node_block) {
         continue;
      } else if (cf->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(cf);
         const int t = hk_nir_lb_walk_list(b, &nif->then_list);
         const int e = hk_nir_lb_walk_list(b, &nif->else_list);
         total = hk_nir_lb_add(b, total, hk_nir_lb_max(b, t, e));
      } else if (cf->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(cf);
         const int trips = hk_nir_lb_loop_trips(b, loop);
         const int body = hk_nir_lb_walk_list(b, &loop->body);
         const int one = hk_nir_lb_const(b, 1);
         const int per_iter = hk_nir_lb_add(b, one, body);
         const int loop_edges = hk_nir_lb_mul(b, trips, per_iter);
         total = hk_nir_lb_add(b, total, loop_edges);
      } else {
         b->failed = true;
         return -1;
      }
   }
   return total;
}

/* Analyze the emitted NIR control-flow tree and populate `out` with an
 * expression computing worst-case (total loop back-edges + 1). Fails closed
 * (out->valid = false) if any loop cannot be bounded statically or via a
 * verified dynamic parameter binding. */
static inline bool
hk_nir_compute_pack_loop_bound(nir_shader *nir,
                               const struct hk_nir_loop_param_binding *bindings,
                               unsigned num_bindings,
                               struct hk_pack_loop_bound *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));
   if (!nir)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint((nir_shader *)nir);
   if (!impl)
      return false;

   struct hk_nir_lb_builder b = {
      .out = out,
      .bindings = bindings,
      .num_bindings = num_bindings,
      .failed = false,
   };

   const int edges = hk_nir_lb_walk_list(&b, &impl->body);
   const int one = hk_nir_lb_const(&b, 1);
   const int bound = hk_nir_lb_add(&b, edges, one);
   if (b.failed || bound < 0 || out->count == 0) {
      memset(out, 0, sizeof(*out));
      return false;
   }
   /* Ensure the final expression value sits at ops[count - 1]. */
   if (bound != (int)out->count - 1) {
      if (out->count >= HK_PACK_LOOP_MAX_OPS) {
         memset(out, 0, sizeof(*out));
         return false;
      }
      out->ops[out->count] = out->ops[bound];
      out->count++;
   }
   out->valid = true;
   return true;
}
