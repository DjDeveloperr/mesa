/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Host unit tests for hk_ray_tracing_core launch group decomposition and the
 * 2D 8x4 / 1D 32x1 invocation-id mapping shared with hk_ray_tracing_pipeline.c
 * and libagx/draws.cl.
 */

#include "hk_ray_tracing_core.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
check_groups(uint32_t w, uint32_t h, uint32_t d,
             uint32_t exp_x, uint32_t exp_y, uint32_t exp_z)
{
   uint32_t out[3] = {0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu};
   hk_rt_launch_group_count(w, h, d, out);
   assert(out[0] == exp_x);
   assert(out[1] == exp_y);
   assert(out[2] == exp_z);
}

/* Verify that every (x, y, z) in [0, w) x [0, h) x [0, d) is visited exactly
 * once by the shader's 8x4 (h > 1) or 32x1 (h <= 1) mapping over the
 * threadgroup grid produced by hk_rt_launch_group_count(). */
static void
check_bijection(uint32_t w, uint32_t h, uint32_t d)
{
   uint32_t groups[3];
   hk_rt_launch_group_count(w, h, d, groups);

   const size_t total = (size_t)w * (size_t)h * (size_t)d;
   uint8_t *seen = calloc(total ? total : 1, 1);
   assert(seen != NULL);

   const bool tile_2d = h > 1;
   for (uint32_t gz = 0; gz < groups[2]; ++gz) {
      for (uint32_t gy = 0; gy < groups[1]; ++gy) {
         for (uint32_t gx = 0; gx < groups[0]; ++gx) {
            for (uint32_t lid = 0; lid < HK_RT_WORKGROUP_SIZE_X; ++lid) {
               const uint32_t x = tile_2d
                  ? ((gx << HK_RT_TILE_2D_X_LOG2) +
                     (lid & (HK_RT_TILE_2D_X - 1)))
                  : ((gx << HK_RT_WORKGROUP_SIZE_X_LOG2) + lid);
               const uint32_t y = tile_2d
                  ? ((gy << HK_RT_TILE_2D_Y_LOG2) +
                     (lid >> HK_RT_TILE_2D_X_LOG2))
                  : gy;
               const uint32_t z = gz;

               if (x >= w || y >= h || z >= d)
                  continue;

               const size_t idx =
                  ((size_t)z * (size_t)h + (size_t)y) * (size_t)w + (size_t)x;
               assert(seen[idx] == 0);
               seen[idx] = 1;
            }
         }
      }
   }

   for (size_t i = 0; i < total; ++i)
      assert(seen[i] == 1);

   free(seen);
}

int
main(void)
{
   /* Zero dimensions produce zero workgroups on the zero axis. */
   check_groups(0, 0, 0, 0, 0, 0);
   check_groups(0, 1080, 1, 0, 270, 1);
   check_groups(1920, 0, 1, 60, 0, 1);
   check_groups(1920, 1080, 0, 240, 270, 0);

   /* 1D launches (height == 1) use flat 32x1 strips. */
   check_groups(1, 1, 1, 1, 1, 1);
   check_groups(31, 1, 1, 1, 1, 1);
   check_groups(32, 1, 1, 1, 1, 1);
   check_groups(33, 1, 1, 2, 1, 1);
   check_groups(100, 1, 7, 4, 1, 7);

   /* 2D/3D launches (height > 1) use 8x4 tiles. */
   check_groups(1, 2, 1, 1, 1, 1);
   check_groups(8, 4, 1, 1, 1, 1);
   check_groups(9, 5, 1, 2, 2, 1);
   check_groups(17, 9, 3, 3, 3, 3);
   check_groups(1920, 1080, 1, 240, 270, 1);
   check_groups(2560, 1440, 1, 320, 360, 1);

   /* Top-of-range non-wrapping arithmetic. */
   check_groups(0xffffffffu, 1, 1, 0x08000000u, 1, 1);
   check_groups(0xffffffffu, 0xffffffffu, 1, 0x20000000u, 0x40000000u, 1);
   check_groups(0xffffffe0u, 1, 1, 0x07ffffffu, 1, 1);
   check_groups(0xfffffff8u, 0xfffffffcu, 2, 0x1fffffffu, 0x3fffffffu, 2);

   /* Exact bijective coverage of shader launch IDs across 1D, 2D, and 3D. */
   check_bijection(0, 5, 1);
   check_bijection(5, 0, 1);
   check_bijection(1, 1, 1);
   check_bijection(31, 1, 1);
   check_bijection(32, 1, 1);
   check_bijection(65, 1, 3);
   check_bijection(1, 2, 1);
   check_bijection(5, 3, 1);
   check_bijection(8, 4, 1);
   check_bijection(9, 5, 1);
   check_bijection(17, 9, 3);
   check_bijection(64, 36, 2);

   printf("HK_RAY_TRACING_CORE_PASS\n");
   return 0;
}
