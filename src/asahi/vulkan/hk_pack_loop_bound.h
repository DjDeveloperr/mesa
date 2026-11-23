/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

enum hk_pack_loop_param {
   HK_PACK_LOOP_PARAM_IM2COL_KH = 0,
   HK_PACK_LOOP_PARAM_DW_KH,
   HK_PACK_LOOP_PARAM_DW_KW,
   HK_PACK_LOOP_PARAM_GEMM_K_ITERS,
   HK_PACK_LOOP_PARAM_PAD_H,
   HK_PACK_LOOP_PARAM_PAD_W,
   HK_PACK_LOOP_PARAM_PAD_ZLP_ITERS,
   HK_PACK_LOOP_PARAM_COUNT,
};

enum hk_pack_loop_op_kind {
   HK_PACK_LOOP_OP_CONST = 0,
   HK_PACK_LOOP_OP_PARAM,
   HK_PACK_LOOP_OP_ADD,
   HK_PACK_LOOP_OP_MUL,
   HK_PACK_LOOP_OP_MAX,
};

struct hk_pack_loop_op {
   uint8_t kind;
   uint8_t lhs;
   uint8_t rhs;
   uint32_t imm;
};

#define HK_PACK_LOOP_MAX_OPS 64u

/* Post-order expression for worst-case (total loop back-edges + 1), derived
 * directly from the emitted NIR control-flow tree. valid == false means at
 * least one loop could not be bounded statically, forcing pack_loops = true. */
struct hk_pack_loop_bound {
   bool valid;
   uint8_t count;
   struct hk_pack_loop_op ops[HK_PACK_LOOP_MAX_OPS];
};

static inline bool
hk_pack_eval_loop_bound(const struct hk_pack_loop_bound *lb,
                        const uint64_t params[HK_PACK_LOOP_PARAM_COUNT],
                        uint64_t *bound_out)
{
   *bound_out = 0;
   if (!lb || !lb->valid || lb->count == 0 || lb->count > HK_PACK_LOOP_MAX_OPS)
      return false;

   uint64_t vals[HK_PACK_LOOP_MAX_OPS];
   for (uint8_t i = 0; i < lb->count; ++i) {
      const struct hk_pack_loop_op op = lb->ops[i];
      uint64_t v = 0;
      switch (op.kind) {
      case HK_PACK_LOOP_OP_CONST:
         v = op.imm;
         break;
      case HK_PACK_LOOP_OP_PARAM:
         if (!params || op.imm >= HK_PACK_LOOP_PARAM_COUNT)
            return false;
         v = params[op.imm];
         break;
      case HK_PACK_LOOP_OP_ADD:
         if (op.lhs >= i || op.rhs >= i ||
             __builtin_add_overflow(vals[op.lhs], vals[op.rhs], &v))
            return false;
         break;
      case HK_PACK_LOOP_OP_MUL:
         if (op.lhs >= i || op.rhs >= i ||
             __builtin_mul_overflow(vals[op.lhs], vals[op.rhs], &v))
            return false;
         break;
      case HK_PACK_LOOP_OP_MAX:
         if (op.lhs >= i || op.rhs >= i)
            return false;
         v = vals[op.lhs] >= vals[op.rhs] ? vals[op.lhs] : vals[op.rhs];
         break;
      default:
         return false;
      }
      vals[i] = v;
   }

   const uint64_t res = vals[lb->count - 1];
   if (res == 0)
      return false;
   *bound_out = res;
   return true;
}
