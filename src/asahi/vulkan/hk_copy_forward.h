/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Command-buffer-level dead-scratch buffer copy forwarding tracker.
 *
 * Tracks pending non-overlapping pure buffer copies (src -> dst) recorded
 * inside a primary command buffer:
 *  - A read-only consumer reading a sub-range contained in `dst` while `src`
 *    remains unmodified can forward its descriptor address directly to the
 *    matching offset inside `src`.
 *  - When a later write-only command completely overwrites `dst` before any
 *    un-forwarded read of `dst` and before command buffer end, the tracked
 *    copy is provably dead and its caller-defined `tag` is reported elidable.
 *  - Any partial overwrite, read-write access, intervening mutation of `src`,
 *    aliased BO/VA overlap, or untrackable barrier/pass conservatively
 *    prevents forwarding or elision.
 */

#define HK_COPY_FORWARD_MAX_ENTRIES 8

struct hk_copy_forward_range {
   /* Non-zero BO handle for alias checks across distinct VA mappings; 0 if
    * unavailable (falls back to VA-only checks). */
   uint32_t handle;
   /* Byte offset within the BO identified by `handle`. */
   uint64_t offset;
   /* GPU virtual address of the start of the range. */
   uint64_t address;
   /* Byte length of the range. */
   uint64_t size;
};

struct hk_copy_forward_entry {
   bool active;
   bool src_mutated;
   uint32_t forwarded_reads;
   uint32_t unforwarded_reads;
   uint64_t tag;
   struct hk_copy_forward_range src;
   struct hk_copy_forward_range dst;
};

struct hk_copy_forward_state {
   struct hk_copy_forward_entry entries[HK_COPY_FORWARD_MAX_ENTRIES];
   uint32_t active_count;
};

void hk_copy_forward_reset(struct hk_copy_forward_state *state);

void hk_copy_forward_invalidate_all(struct hk_copy_forward_state *state);

bool hk_copy_forward_range_valid(const struct hk_copy_forward_range *r);

bool hk_copy_forward_ranges_overlap(const struct hk_copy_forward_range *a,
                                    const struct hk_copy_forward_range *b);

bool hk_copy_forward_range_contains(const struct hk_copy_forward_range *outer,
                                    const struct hk_copy_forward_range *inner);

/* Record a pure non-overlapping buffer copy `src -> dst` identified by `tag`.
 * First applies the full write-only overwrite to `dst` (which may prove an
 * earlier pending copy targeting `dst` dead and emit its tag into
 * `elided_tags`/`elided_count`). Returns true if the new copy was tracked. */
bool hk_copy_forward_record_copy(struct hk_copy_forward_state *state,
                                 const struct hk_copy_forward_range *src,
                                 const struct hk_copy_forward_range *dst,
                                 uint64_t tag,
                                 uint64_t *elided_tags,
                                 uint32_t *elided_count);

/* Attempt to forward a read-only consumer of `read_range` to the source of an
 * active unmodified copy whose destination contains `read_range`.
 * On success, writes the forwarded GPU VA to `*forwarded_va_out` and returns
 * true. If `read_range` overlaps an active copy's destination without being
 * forwardable, records an un-forwarded read on that entry so it cannot later
 * be elided. */
bool hk_copy_forward_resolve_readonly_read(
   struct hk_copy_forward_state *state,
   const struct hk_copy_forward_range *read_range,
   uint64_t *forwarded_va_out);

/* Record a read-write access to `range`: prevents elision of any active copy
 * whose destination overlaps `range`, invalidates forwarding from that
 * destination, and marks `src_mutated` on any copy whose source overlaps it. */
void hk_copy_forward_note_readwrite(struct hk_copy_forward_state *state,
                                    const struct hk_copy_forward_range *range);

/* Record a buffer write to `write_range`. If `write_only` is true and
 * `write_range` completely covers an active copy's destination with zero
 * un-forwarded reads, appends that copy's `tag` to `elided_tags` and removes
 * the entry. Partial overwrites deactivate forwarding without elision. Any
 * overlap with an entry's `src` marks `src_mutated`. */
void hk_copy_forward_note_write(struct hk_copy_forward_state *state,
                                const struct hk_copy_forward_range *write_range,
                                bool write_only,
                                uint64_t *elided_tags,
                                uint32_t *elided_count);

#ifdef __cplusplus
}
#endif
