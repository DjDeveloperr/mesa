/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_dynarray.h"


static bool
mux_block_in_loop(nir_block *block, nir_loop *loop)
{
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
      if (node == &loop->cf_node)
         return true;
   }
   return false;
}

static bool
mux_extract_split(nir_def *cond, nir_def **idx_out, int64_t *split_out,
                  bool *lt_is_true_out, bool *is_eq_out)
{
   if (cond->bit_size != 1 || cond->num_components != 1)
      return false;
   nir_instr *ci = nir_def_instr(cond);
   if (ci->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(ci);
   if (nir_op_infos[alu->op].num_inputs != 2)
      return false;

   for (unsigned s = 0; s < 2; ++s) {
      if (alu->src[s].src.ssa->bit_size != 32 ||
          alu->src[s].src.ssa->num_components != 1 ||
          alu->src[s].swizzle[0] != 0)
         return false;
   }

   nir_scalar a = nir_scalar_resolved(alu->src[0].src.ssa, 0);
   nir_scalar b = nir_scalar_resolved(alu->src[1].src.ssa, 0);
   bool a_const = nir_scalar_is_const(a);
   bool b_const = nir_scalar_is_const(b);
   if (a_const == b_const)
      return false;

   nir_def *idx = a_const ? b.def : a.def;
   int64_t k = a_const ? nir_scalar_as_int(a) : nir_scalar_as_int(b);
   bool idx_left = !a_const;

   *idx_out = idx;
   *is_eq_out = false;

   switch (alu->op) {
   case nir_op_ilt:
   case nir_op_ult:
      if (idx_left) {
         *split_out = k;
         *lt_is_true_out = true;
      } else {
         *split_out = k + 1;
         *lt_is_true_out = false;
      }
      return true;
   case nir_op_ige:
   case nir_op_uge:
      if (idx_left) {
         *split_out = k;
         *lt_is_true_out = false;
      } else {
         *split_out = k + 1;
         *lt_is_true_out = true;
      }
      return true;
   case nir_op_ieq:
   case nir_op_ine:
      *is_eq_out = true;
      *split_out = k;
      *lt_is_true_out = (alu->op == nir_op_ieq);
      return true;
   default:
      return false;
   }
}

static unsigned
mux_index_bound(nir_def *idx)
{
   if (idx->bit_size != 32 || idx->num_components != 1)
      return 0;
   nir_instr *instr = nir_def_instr(idx);
   if (instr->type != nir_instr_type_alu)
      return 0;
   nir_alu_instr *alu = nir_instr_as_alu(instr);

   if (alu->op == nir_op_iand) {
      for (unsigned s = 0; s < 2; ++s) {
         nir_scalar c = nir_scalar_resolved(alu->src[s].src.ssa, alu->src[s].swizzle[0]);
         if (!nir_scalar_is_const(c))
            continue;
         uint32_t mask = nir_scalar_as_uint(c);
         if (mask == 7 || mask == 15 || mask == 31)
            return mask + 1;
      }
   } else if (alu->op == nir_op_umod) {
      nir_scalar c = nir_scalar_resolved(alu->src[1].src.ssa, alu->src[1].swizzle[0]);
      if (nir_scalar_is_const(c)) {
         uint32_t m = nir_scalar_as_uint(c);
         if (m == 8 || m == 16 || m == 32)
            return m;
      }
   } else if (alu->op == nir_op_umin) {
      for (unsigned s = 0; s < 2; ++s) {
         nir_scalar c = nir_scalar_resolved(alu->src[s].src.ssa, alu->src[s].swizzle[0]);
         if (!nir_scalar_is_const(c))
            continue;
         uint32_t m = nir_scalar_as_uint(c);
         if (m == 7 || m == 15 || m == 31)
            return m + 1;
      }
   } else if (alu->op == nir_op_ubitfield_extract) {
      nir_scalar bits = nir_scalar_resolved(alu->src[2].src.ssa, alu->src[2].swizzle[0]);
      if (nir_scalar_is_const(bits)) {
         uint32_t b = nir_scalar_as_uint(bits);
         if (b >= 3 && b <= 5)
            return 1u << b;
      }
   }

   return 0;
}

static bool
mux_leaf_invariant(nir_def *def, nir_loop *loop)
{
   if (def->bit_size != 8 || def->num_components != 1)
      return false;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const)
      return true;
   return !mux_block_in_loop(instr->block, loop);
}

static bool
mux_resolve_range(nir_def *cur, nir_def *idx, nir_block *root_block,
                  nir_loop *loop, int64_t lo, int64_t hi,
                  nir_def **leaves, unsigned *internal_nodes, bool is_root)
{
   if (hi <= lo)
      return false;

   if (mux_leaf_invariant(cur, loop)) {
      for (int64_t i = lo; i < hi; ++i)
         leaves[i] = cur;
      return true;
   }

   if (hi - lo == 1)
      return false;

   nir_instr *instr = nir_def_instr(cur);
   if (instr->type != nir_instr_type_alu || instr->block != root_block)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_bcsel || cur->bit_size != 8 || cur->num_components != 1)
      return false;

   if (!is_root && !list_is_singular(&cur->uses))
      return false;

   for (unsigned s = 0; s < 3; ++s) {
      if (alu->src[s].src.ssa->num_components != 1 || alu->src[s].swizzle[0] != 0)
         return false;
   }

   nir_def *cur_idx = NULL;
   int64_t k = 0;
   bool lt_is_true = false;
   bool is_eq = false;
   if (!mux_extract_split(alu->src[0].src.ssa, &cur_idx, &k, &lt_is_true, &is_eq))
      return false;
   if (cur_idx != idx)
      return false;

   nir_def *t_val = alu->src[1].src.ssa;
   nir_def *f_val = alu->src[2].src.ssa;
   (*internal_nodes)++;

   if (is_eq) {
      if (hi - lo != 2)
         return false;
      nir_def *eq_val = lt_is_true ? t_val : f_val;
      nir_def *ne_val = lt_is_true ? f_val : t_val;
      if (k == lo) {
         return mux_resolve_range(eq_val, idx, root_block, loop, lo, lo + 1,
                                  leaves, internal_nodes, false) &&
                mux_resolve_range(ne_val, idx, root_block, loop, lo + 1, hi,
                                  leaves, internal_nodes, false);
      } else if (k == lo + 1) {
         return mux_resolve_range(ne_val, idx, root_block, loop, lo, lo + 1,
                                  leaves, internal_nodes, false) &&
                mux_resolve_range(eq_val, idx, root_block, loop, lo + 1, hi,
                                  leaves, internal_nodes, false);
      }
      return false;
   }

   if (k <= lo || k >= hi)
      return false;

   nir_def *lo_val = lt_is_true ? t_val : f_val;
   nir_def *hi_val = lt_is_true ? f_val : t_val;
   return mux_resolve_range(lo_val, idx, root_block, loop, lo, k,
                            leaves, internal_nodes, false) &&
          mux_resolve_range(hi_val, idx, root_block, loop, k, hi,
                            leaves, internal_nodes, false);
}

static nir_def *
mux_build_word_select(nir_builder *b, nir_def *sel_idx, unsigned step,
                      nir_def **words, unsigned lo, unsigned hi)
{
   if (hi - lo == 1)
      return words[lo];
   unsigned mid = lo + (hi - lo) / 2;
   nir_def *left = mux_build_word_select(b, sel_idx, step, words, lo, mid);
   nir_def *right = mux_build_word_select(b, sel_idx, step, words, mid, hi);
   nir_def *cond = nir_ilt_imm(b, sel_idx, (int32_t)(mid * step));
   return nir_bcsel(b, cond, left, right);
}

/* Estimate peak live pre-pack leaves + partial/complete words at preheader
 * entry and through preheader instructions for contiguous vs strided packing. */
static bool
mux_prefer_strided_layout(nir_block *pre, nir_def **leaves, unsigned n)
{
   unsigned nwords = n / 4;
   unsigned contig_entry_packed = 0;
   unsigned strided_entry_packed = 0;

   for (unsigned w = 0; w < nwords; ++w) {
      bool all_pre_c = true;
      bool all_pre_s = true;
      for (unsigned c = 0; c < 4; ++c) {
         nir_instr *ic = nir_def_instr(leaves[4 * w + c]);
         nir_instr *is = nir_def_instr(leaves[c * nwords + w]);
         if (ic->block == pre && ic->type != nir_instr_type_load_const)
            all_pre_c = false;
         if (is->block == pre && is->type != nir_instr_type_load_const)
            all_pre_s = false;
      }
      if (all_pre_c)
         contig_entry_packed++;
      if (all_pre_s)
         strided_entry_packed++;
   }

   /* If strided grouping lets us immediately pack 4 pre-existing leaves into a
    * word at preheader entry while contiguous grouping cannot, prefer strided
    * so those 4 values collapse to 1 register before the preheader body runs. */
   return strided_entry_packed > contig_entry_packed;
}

static nir_cursor
mux_cursor_after_pair(nir_block *pre, nir_def *d0, nir_def *d1)
{
   nir_instr *i0 = nir_def_instr(d0);
   nir_instr *i1 = nir_def_instr(d1);
   nir_instr *latest = NULL;
   nir_foreach_instr(pi, pre) {
      if (pi == i0 || pi == i1)
         latest = pi;
   }
   return latest ? nir_after_instr(latest) : nir_after_phis(pre);
}

static bool
pack_mux_trees_in_cf_list(nir_function_impl *impl, struct exec_list *cf_list,
                          nir_loop *enclosing_loop)
{
   bool progress = false;

   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= pack_mux_trees_in_cf_list(impl, &nif->then_list, enclosing_loop);
         progress |= pack_mux_trees_in_cf_list(impl, &nif->else_list, enclosing_loop);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= pack_mux_trees_in_cf_list(impl, &loop->body, loop);
         break;
      }
      case nir_cf_node_block: {
         if (!enclosing_loop)
            break;
         nir_cf_node *prev = nir_cf_node_prev(&enclosing_loop->cf_node);
         if (!prev || prev->type != nir_cf_node_block)
            break;
         nir_block *pre = nir_cf_node_as_block(prev);
         nir_block *block = nir_cf_node_as_block(node);

         nir_foreach_instr_reverse_safe(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *root = nir_instr_as_alu(instr);
            if (root->op != nir_op_bcsel || root->def.bit_size != 8 ||
                root->def.num_components != 1 || nir_def_is_unused(&root->def))
               continue;

            nir_def *idx = NULL;
            int64_t k = 0;
            bool lt_is_true = false;
            bool is_eq = false;
            if (!mux_extract_split(root->src[0].src.ssa, &idx, &k, &lt_is_true, &is_eq))
               continue;

            unsigned n = mux_index_bound(idx);
            if (n < 8 || n > 32)
               continue;

            nir_def *leaves[32] = {0};
            unsigned internal_nodes = 0;
            if (!mux_resolve_range(&root->def, idx, block, enclosing_loop,
                                   0, n, leaves, &internal_nodes, true))
               continue;

            const unsigned nwords = n / 4;
            /* The packed lookup emits (nwords - 1) word selects plus 3 fixed
             * extraction ops (word index, byte bit-shift, ubitfield_extract).
             * Require strictly more original in-loop selects than that cost. */
            const unsigned packed_in_loop_ops = (nwords - 1) + 3;
            if (internal_nodes <= packed_in_loop_ops)
               continue;

            /* Count distinct non-constant leaves defined outside the loop. */
            unsigned distinct = 0;
            nir_def *seen[32];
            for (unsigned i = 0; i < n; ++i) {
               if (nir_def_instr(leaves[i])->type == nir_instr_type_load_const)
                  continue;
               bool dup = false;
               for (unsigned j = 0; j < distinct; ++j) {
                  if (seen[j] == leaves[i]) {
                     dup = true;
                     break;
                  }
               }
               if (!dup)
                  seen[distinct++] = leaves[i];
            }
            /* Each packed 32-bit word holds 2 16-bit half-registers across the
             * loop, whereas each distinct runtime 8-bit leaf holds 1 half. Only
             * pack when distinct runtime leaves strictly exceed 2 * nwords. */
            const unsigned packed_halves = nwords * 2;
            if (distinct <= packed_halves)
               continue;

            /* Pack every 4 loop-invariant bytes into one 32-bit word in the
             * loop preheader, placing each pack right after the latest defining
             * instruction of its operands in `pre` so the narrow source bytes
             * can die immediately instead of overlapping across the entire
             * preheader. */
            nir_builder b = nir_builder_create(impl);
            bool strided = mux_prefer_strided_layout(pre, leaves, n);
            nir_def *words[8];
            for (unsigned w = 0; w < nwords; ++w) {
               nir_def *bytes[4];
               for (unsigned c = 0; c < 4; ++c) {
                  nir_def *leaf = strided ? leaves[c * nwords + w]
                                          : leaves[4 * w + c];
                  nir_instr *li = nir_def_instr(leaf);
                  if (li->type == nir_instr_type_load_const) {
                     b.cursor = nir_after_phis(pre);
                     uint8_t val = nir_instr_as_load_const(li)->value[0].u8;
                     bytes[c] = nir_imm_int(&b, val);
                  } else {
                     b.cursor = (li->block == pre) ? nir_after_instr(li)
                                                   : nir_after_phis(pre);
                     bytes[c] = nir_u2u32(&b, leaf);
                  }
               }
               b.cursor = mux_cursor_after_pair(pre, bytes[0], bytes[1]);
               nir_def *acc = nir_imadshl_agx(&b, bytes[1], nir_imm_int(&b, 256),
                                              bytes[0], nir_imm_int(&b, 0));
               b.cursor = mux_cursor_after_pair(pre, acc, bytes[2]);
               acc = nir_imadshl_agx(&b, bytes[2], nir_imm_int(&b, 65536),
                                     acc, nir_imm_int(&b, 0));
               b.cursor = mux_cursor_after_pair(pre, acc, bytes[3]);
               words[w] = nir_imadshl_agx(&b, bytes[3], nir_imm_int(&b, 16777216),
                                          acc, nir_imm_int(&b, 0));
            }

            /* Peel a redundant power-of-two mask (e.g. (i & 15) & 3 -> i & 3)
             * when extracting bit subfields of the index. */
            nir_def *raw_idx = idx;
            nir_instr *idx_instr = nir_def_instr(idx);
            if (idx_instr->type == nir_instr_type_alu) {
               nir_alu_instr *ia = nir_instr_as_alu(idx_instr);
               if (ia->op == nir_op_iand) {
                  for (unsigned s = 0; s < 2; ++s) {
                     nir_scalar c = nir_scalar_resolved(ia->src[s].src.ssa,
                                                        ia->src[s].swizzle[0]);
                     if (nir_scalar_is_const(c) &&
                         nir_scalar_as_uint(c) == n - 1 &&
                         ia->src[1 - s].src.ssa->num_components == 1 &&
                         ia->src[1 - s].swizzle[0] == 0) {
                        raw_idx = ia->src[1 - s].src.ssa;
                        break;
                     }
                  }
               }
            }

            b.cursor = nir_before_instr(&root->instr);
            nir_def *word;
            nir_def *bit_off;
            if (strided) {
               nir_def *word_idx = nir_iand_imm(&b, raw_idx, nwords - 1);
               word = mux_build_word_select(&b, word_idx, 1, words, 0, nwords);
               nir_def *hi_bits = nir_iand_imm(&b, raw_idx, 3u * nwords);
               unsigned shift = (nwords == 2) ? 2 : (nwords == 4) ? 1 : 0;
               bit_off = shift ? nir_imadshl_agx(&b, nir_imm_int(&b, 0),
                                                 nir_imm_int(&b, 1), hi_bits,
                                                 nir_imm_int(&b, shift))
                               : hi_bits;
            } else {
               word = mux_build_word_select(&b, idx, 4, words, 0, nwords);
               nir_def *byte_lane = nir_iand_imm(&b, raw_idx, 3);
               bit_off = nir_imadshl_agx(&b, nir_imm_int(&b, 0),
                                         nir_imm_int(&b, 1), byte_lane,
                                         nir_imm_int(&b, 3));
            }
            nir_def *byte32 = nir_ubitfield_extract(&b, word, bit_off,
                                                    nir_imm_int(&b, 8));

            bool need_u8 = false;
            nir_foreach_use_safe(use, &root->def) {
               nir_instr *ui = nir_src_use_instr(use);
               if (ui->type == nir_instr_type_alu) {
                  nir_alu_instr *ualu = nir_instr_as_alu(ui);
                  if (ualu->op == nir_op_u2u32 && ualu->def.num_components == 1 &&
                      ualu->src[0].swizzle[0] == 0) {
                     nir_def_replace(&ualu->def, byte32);
                     continue;
                  }
               }
               need_u8 = true;
            }
            if (need_u8 || !list_is_empty(&root->def.uses)) {
               nir_def *byte8 = nir_u2u8(&b, byte32);
               nir_def_replace(&root->def, byte8);
            } else {
               nir_instr_remove(&root->instr);

               /* If the extracted byte feeds a loop-invariant signed difference
                * `inv_cond ? -(c - byte) : (c - byte)` with c in [0, 255],
                * pre-complement the packed words and c in the preheader using
                * `(255 - b) - (255 - c) == c - b`, collapsing the in-loop
                * negation + select to a single `isub(byte, c_eff)`. */
               if (list_is_singular(&byte32->uses)) {
                  nir_src *buse = list_first_entry(&byte32->uses, nir_src, use_link);
                  if (!nir_src_is_if(buse) &&
                      nir_src_use_instr(buse)->type == nir_instr_type_alu) {
                     nir_alu_instr *sub = nir_instr_as_alu(nir_src_use_instr(buse));
                     if (sub->op == nir_op_isub && sub->instr.block == block &&
                         sub->def.num_components == 1 && sub->def.bit_size == 32 &&
                         sub->src[0].swizzle[0] == 0 && sub->src[1].swizzle[0] == 0) {
                        bool byte_is_rhs = (sub->src[1].src.ssa == byte32);
                        nir_def *c = byte_is_rhs ? sub->src[0].src.ssa
                                                 : sub->src[1].src.ssa;
                        bool c_u8 = false;
                        nir_instr *ci = nir_def_instr(c);
                        if (ci->type == nir_instr_type_load_const) {
                           c_u8 = nir_instr_as_load_const(ci)->value[0].u32 <= 255;
                        } else if (ci->type == nir_instr_type_alu) {
                           nir_alu_instr *ca = nir_instr_as_alu(ci);
                           if (ca->op == nir_op_u2u32 &&
                               ca->src[0].src.ssa->bit_size == 8)
                              c_u8 = true;
                        }
                        if (c_u8 && !mux_block_in_loop(ci->block, enclosing_loop)) {
                           nir_alu_instr *neg = NULL;
                           nir_alu_instr *sel = NULL;
                           unsigned sub_uses = 0;
                           nir_foreach_use_including_if(su, &sub->def) {
                              sub_uses++;
                              if (nir_src_is_if(su) ||
                                  nir_src_use_instr(su)->type != nir_instr_type_alu) {
                                 sub_uses = 99;
                                 break;
                              }
                              nir_alu_instr *ua = nir_instr_as_alu(nir_src_use_instr(su));
                              if (ua->op == nir_op_ineg && ua->instr.block == block &&
                                  list_is_singular(&ua->def.uses))
                                 neg = ua;
                              else if (ua->op == nir_op_bcsel && ua->instr.block == block)
                                 sel = ua;
                           }
                           if (sub_uses == 2 && neg && sel &&
                               sel->src[0].swizzle[0] == 0 &&
                               sel->src[1].swizzle[0] == 0 &&
                               sel->src[2].swizzle[0] == 0 &&
                               ((sel->src[1].src.ssa == &neg->def &&
                                 sel->src[2].src.ssa == &sub->def) ||
                                (sel->src[1].src.ssa == &sub->def &&
                                 sel->src[2].src.ssa == &neg->def))) {
                              nir_def *inv_cond = sel->src[0].src.ssa;
                              if (!mux_block_in_loop(nir_def_instr(inv_cond)->block,
                                                     enclosing_loop)) {
                                 bool neg_is_true = (sel->src[1].src.ssa == &neg->def);
                                 bool flip_when_true = (byte_is_rhs != neg_is_true);
                                 b.cursor = nir_after_block_before_jump(pre);
                                 for (unsigned w = 0; w < nwords; ++w) {
                                    nir_def *inv_w = nir_inot(&b, words[w]);
                                    nir_def *w_eff = flip_when_true
                                       ? nir_bcsel(&b, inv_cond, inv_w, words[w])
                                       : nir_bcsel(&b, inv_cond, words[w], inv_w);
                                    nir_foreach_use_safe(wu, words[w]) {
                                       if (mux_block_in_loop(nir_src_get_block(wu),
                                                             enclosing_loop))
                                          nir_src_rewrite(wu, w_eff);
                                    }
                                 }
                                 nir_def *inv_c = nir_isub(&b, nir_imm_int(&b, 255), c);
                                 nir_def *c_eff = flip_when_true
                                    ? nir_bcsel(&b, inv_cond, inv_c, c)
                                    : nir_bcsel(&b, inv_cond, c, inv_c);
                                 b.cursor = nir_before_instr(&sel->instr);
                                 nir_def *new_diff = nir_isub(&b, byte32, c_eff);
                                 nir_def_replace(&sel->def, new_diff);
                                 nir_instr_remove(&neg->instr);
                                 nir_instr_remove(&sub->instr);
                              }
                           }
                        }
                     }
                  }
               }
            }

            progress = true;
         }
         break;
      }
      default:
         break;
      }
   }

   return progress;
}

/* Fold canonicalized 32-bit iadd(a, ineg(b)) back into native isub. */
static bool
fold_iadd_ineg_to_isub(nir_function_impl *impl)
{
   bool progress = false;
   nir_builder b = nir_builder_create(impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *add = nir_instr_as_alu(instr);
         if (add->op != nir_op_iadd || add->def.bit_size != 32 ||
             add->def.num_components != 1)
            continue;

         for (unsigned s = 0; s < 2; ++s) {
            nir_def *neg_def = add->src[s].src.ssa;
            nir_def *other = add->src[1 - s].src.ssa;
            if (add->src[s].swizzle[0] != 0 || add->src[1 - s].swizzle[0] != 0)
               continue;
            if (nir_def_instr(other)->type == nir_instr_type_load_const)
               continue;
            nir_instr *ni = nir_def_instr(neg_def);
            if (ni->type != nir_instr_type_alu || !list_is_singular(&neg_def->uses))
               continue;
            nir_alu_instr *neg = nir_instr_as_alu(ni);
            if (neg->op != nir_op_ineg || neg->src[0].swizzle[0] != 0 ||
                neg->src[0].src.ssa->bit_size != 32 ||
                neg->src[0].src.ssa->num_components != 1)
               continue;

            b.cursor = nir_before_instr(&add->instr);
            nir_def *sub = nir_isub(&b, other, neg->src[0].src.ssa);
            nir_def_replace(&add->def, sub);
            nir_instr_remove(&neg->instr);
            progress = true;
            break;
         }
      }
   }
   return progress;
}

/*
 * Pack loop-invariant 8-bit bcsel mux trees inside loops into 32-bit words in
 * the loop preheader, replacing an N-way byte decision tree with an (N/4)-way
 * 32-bit word select and a single byte extract. This shrinks live-across-loop
 * scalar 8-bit register pressure (where each 8-bit value otherwise occupies a
 * full 16-bit half-register slot on AGX) and replaces up to 15 selects per
 * loop iteration with 3 selects plus one bitfield extract.
 */
bool
agx_nir_opt_pack_narrow_mux_tree(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      bool impl_progress = fold_iadd_ineg_to_isub(impl);
      impl_progress |= pack_mux_trees_in_cf_list(impl, &impl->body, NULL);
      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }

   return progress;
}
