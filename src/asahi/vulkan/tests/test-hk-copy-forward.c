/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>

#include "hk_copy_forward.h"

#define CHECK(cond)                                                            \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__,      \
                 #cond);                                                       \
         abort();                                                              \
      }                                                                        \
   } while (0)

static struct hk_copy_forward_range
mk_range(uint32_t handle, uint64_t offset, uint64_t address, uint64_t size)
{
   return (struct hk_copy_forward_range){
      .handle = handle,
      .offset = offset,
      .address = address,
      .size = size,
   };
}

int
main(void)
{
   struct hk_copy_forward_state st;
   uint64_t elided[HK_COPY_FORWARD_MAX_ENTRIES];
   uint32_t elided_n = 0;
   uint64_t fwd_va = 0;

   const struct hk_copy_forward_range buf_a = mk_range(1, 0, 0x100000, 4096);
   const struct hk_copy_forward_range buf_b = mk_range(2, 0, 0x200000, 4096);
   const struct hk_copy_forward_range buf_c = mk_range(3, 0, 0x300000, 4096);

   /* 1. Positive: copy(A -> B), readonly read(B) forwarded to A,
    * full overwrite(B) proves copy dead and elides it. */
   hk_copy_forward_reset(&st);
   CHECK(hk_copy_forward_record_copy(&st, &buf_a, &buf_b, 101, elided, &elided_n));
   CHECK(elided_n == 0);
   CHECK(hk_copy_forward_resolve_readonly_read(&st, &buf_b, &fwd_va));
   CHECK(fwd_va == buf_a.address);
   hk_copy_forward_note_write(&st, &buf_b, true, elided, &elided_n);
   CHECK(elided_n == 1 && elided[0] == 101);
   CHECK(st.active_count == 0);

   /* 2. Positive: sub-range read inside B forwards to matching offset in A. */
   hk_copy_forward_reset(&st);
   CHECK(hk_copy_forward_record_copy(&st, &buf_a, &buf_b, 102, elided, &elided_n));
   struct hk_copy_forward_range sub_b = mk_range(2, 256, 0x200100, 512);
   CHECK(hk_copy_forward_resolve_readonly_read(&st, &sub_b, &fwd_va));
   CHECK(fwd_va == buf_a.address + 256);
   hk_copy_forward_note_write(&st, &buf_b, true, elided, &elided_n);
   CHECK(elided_n == 1 && elided[0] == 102);

   /* 3. Positive: multiple consecutive consumers reading B all forward to A. */
   hk_copy_forward_reset(&st);
   CHECK(hk_copy_forward_record_copy(&st, &buf_a, &buf_b, 103, elided, &elided_n));
   CHECK(hk_copy_forward_resolve_readonly_read(&st, &buf_b, &fwd_va));
   CHECK(fwd_va == buf_a.address);
   CHECK(hk_copy_forward_resolve_readonly_read(&st, &sub_b, &fwd_va));
   CHECK(fwd_va == buf_a.address + 256);
   hk_copy_forward_note_write(&st, &buf_b, true, elided, &elided_n);
   CHECK(elided_n == 1 && elided[0] == 103);

   /* 4. Positive: chained scratch reuse (copy A->B, read B, copy C->B,
    * read B, overwrite B) elides both copies in order. */
   hk_copy_forward_reset(&st);
   CHECK(hk_copy_forward_record_copy(&st, &buf_a, &buf_b, 201, elided, &elided_n));
   CHECK(elided_n == 0);
   CHECK(hk_copy_forward_resolve_readonly_read(&st, &buf_b, &fwd_va));
   CHECK(fwd_va == buf_a.address);
   CHECK(hk_copy_forward_record_copy(&st, &buf_c, &buf_b, 202, elided, &elided_n));
   CHECK(elided_n == 1 && elided[0] == 201);
   CHECK(hk_copy_forward_resolve_readonly_read(&st, &buf_b, &fwd_va));
   CHECK(fwd_va == buf_c.address);
   hk_copy_forward_note_write(&st, &buf_b, true, elided, &elided_n);
   CHECK(elided_n == 1 && elided[0] == 202);

   /* 5. Positive: mutation of src AFTER consumer already forwarded still
    * allows elision once dst is fully overwritten with no unforwarded reads. */
   hk_copy_forward_reset(&st);
   CHECK(hk_copy_forward_record_copy(&st, &buf_a, &buf_b, 203, elided, &elided_n));
   CHECK(hk_copy_forward_resolve_readonly_read(&st, &buf_b, &fwd_va));
   CHECK(fwd_va == buf_a.address);
   hk_copy_forward_note_write(&st, &buf_a, true, elided, &elided_n);
   CHECK(elided_n == 0);
   hk_copy_forward_note_write(&st, &buf_b, true, elided, &elided_n);
   CHECK(elided_n == 1 && elided[0] == 203);

   /* 6. Negative: dst live-out at CB end (invalidate_all) never elides copy. */
   hk_copy_forward_reset(&st);
   CHECK(hk_copy_forward_record_copy(&st, &buf_a, &buf_b, 301, elided, &elided_n));
   CHECK(hk_copy_forward_resolve_readonly_read(&st, &buf_b, &fwd_va));
   hk_copy_forward_invalidate_all(&st);
   CHECK(st.active_count == 0);
   hk_copy_forward_note_write(&st, &buf_b, true, elided, &elided_n);
   CHECK(elided_n == 0);

   /* 7. Negative: intervening write to src BEFORE consumer reads dst blocks
    * forwarding and prevents later elision. */
   hk_copy_forward_reset(&st);
   CHECK(hk_copy_forward_record_copy(&st, &buf_a, &buf_b, 302, elided, &elided_n));
   hk_copy_forward_note_write(&st, &buf_a, true, elided, &elided_n);
   CHECK(!hk_copy_forward_resolve_readonly_read(&st, &buf_b, &fwd_va));
   hk_copy_forward_note_write(&st, &buf_b, true, elided, &elided_n);
   CHECK(elided_n == 0);

   /* 8. Negative: partial overwrite of dst does not elide copy. */
   hk_copy_forward_reset(&st);
   CHECK(hk_copy_forward_record_copy(&st, &buf_a, &buf_b, 303, elided, &elided_n));
   CHECK(hk_copy_forward_resolve_readonly_read(&st, &buf_b, &fwd_va));
   struct hk_copy_forward_range half_b = mk_range(2, 0, 0x200000, 2048);
   hk_copy_forward_note_write(&st, &half_b, true, elided, &elided_n);
   CHECK(elided_n == 0);
   /* Subsequent full write cannot retroactively elide already-deactivated copy. */
   hk_copy_forward_note_write(&st, &buf_b, true, elided, &elided_n);
   CHECK(elided_n == 0);

   /* 9. Negative: read-write consumer of dst prevents elision. */
   hk_copy_forward_reset(&st);
   CHECK(hk_copy_forward_record_copy(&st, &buf_a, &buf_b, 304, elided, &elided_n));
   hk_copy_forward_note_readwrite(&st, &buf_b);
   hk_copy_forward_note_write(&st, &buf_b, true, elided, &elided_n);
   CHECK(elided_n == 0);

   /* 10. Negative: overlapping VA, overlapping BO with distinct VA, and
    * exact self-copy are rejected. */
   hk_copy_forward_reset(&st);
   struct hk_copy_forward_range overlap_va = mk_range(4, 0, 0x100800, 4096);
   CHECK(!hk_copy_forward_record_copy(&st, &buf_a, &overlap_va, 401, elided, &elided_n));
   struct hk_copy_forward_range alias_bo = mk_range(1, 512, 0x500000, 4096);
   CHECK(!hk_copy_forward_record_copy(&st, &buf_a, &alias_bo, 402, elided, &elided_n));
   CHECK(!hk_copy_forward_record_copy(&st, &buf_a, &buf_a, 403, elided, &elided_n));

   /* 11. Negative: straddling read extending past dst end blocks forwarding
    * and prevents elision. */
   hk_copy_forward_reset(&st);
   CHECK(hk_copy_forward_record_copy(&st, &buf_a, &buf_b, 404, elided, &elided_n));
   struct hk_copy_forward_range straddle = mk_range(2, 4080, 0x200ff0, 64);
   CHECK(!hk_copy_forward_resolve_readonly_read(&st, &straddle, &fwd_va));
   hk_copy_forward_note_write(&st, &buf_b, true, elided, &elided_n);
   CHECK(elided_n == 0);

   /* 12. Negative: overflow wraparound and zero-size ranges rejected. */
   hk_copy_forward_reset(&st);
   struct hk_copy_forward_range zero_r = mk_range(1, 0, 0x100000, 0);
   struct hk_copy_forward_range wrap_r = mk_range(1, 0, UINT64_MAX - 8, 32);
   CHECK(!hk_copy_forward_range_valid(&zero_r));
   CHECK(!hk_copy_forward_range_valid(&wrap_r));
   CHECK(!hk_copy_forward_record_copy(&st, &zero_r, &zero_r, 501, elided, &elided_n));
   CHECK(!hk_copy_forward_record_copy(&st, &wrap_r, &buf_b, 502, elided, &elided_n));

   return 0;
}
