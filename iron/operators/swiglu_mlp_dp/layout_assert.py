#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-free budget assertions for the fused-SwiGLU-MLP layout options.

Encodes the hardware budget table for this design's weight-channel layout options -- including
the MemTile-staged option B (smaller tiles, deeper L1 fifo) -- as executable checks, so a layout
change can be validated BEFORE anything is compiled or run.

Budgets and where they come from (all authoritative, not estimated):
  * compute-tile L1          65536 B   AIETargetModel.h `getLocalMemorySize()` (AIE2P)
  * MemTile                  0x80000   `getMemTileSize()`
  * BDs                      48 memtile / 16 others   `getNumBDs(col,row)`
  * locks                    64 memtile / 16 others   `getNumLocks(col,row)`; max value 63
  * program memory           0x4000    `getProgramMemorySize()`
  * shim, per tile & direction <= 16 simultaneously active BDs, and **a fill's OBJECT count
    counts against it** -- this is the rule that killed the qkv misc widening (measured:
    "Too many simultaneously active buffer descriptors on tile (0,0)") and forces per-object
    fills in the first place.
  * one BD moves <= 16383 words (64 KB); a wrap dimension is <= 1023 (mlir-aie
    verifyStridesWraps, the reason `_split_run` exists).
  * objectfifo transfers are counted in OBJECTS: every fill/drain must move a whole number of
    objects (a fractional delivery desyncs the accounting -- measured ERT_CMD_STATE_TIMEOUT).

Usage:  python layout_assert.py            # checks every config below and prints a table
        python layout_assert.py --json     # machine-readable
"""
import argparse
import json

L1_BYTES = 65536
MEMTILE_BYTES = 0x00080000
BD_MEMTILE, BD_OTHER = 48, 16
# The shim's BD-ID pool is per CHANNEL and is consumed by the FIFO DEPTHS mapped onto it --
# MEASURED on the option-B (MemTile-staged) build: a stage depth of 14 or 16 fails
# with `'aie.dma_bd' op Allocator exhausted available BD IDs (maximum 24 available for
# channel 0)`, while 12 builds and runs.  With weight depth 4 + misc 2 + out 2 that pins the
# MemTile stage at ~12 tiles -- i.e. this pool, not the MemTile's 512 KB, is what bounds how
# "deep" the staged buffer can be.
BD_ID_SHIM = 24
LOCK_MEMTILE, LOCK_OTHER = 64, 16
PROGRAM_MEM = 0x4000
BD_LEN_WORDS_MAX = 16383
WRAP_MAX = 1023
# Measured on this tree: a fully unrolled per-head/per-tile loop costs ~143 B of .text per
# mv/rope call site (swiglu 4B: 96 sites = 16896 B > 16384 -> program-memory overflow).
TEXT_PER_CALL_SITE = 143


def wtile_units(tsi, d, group=128):
    """Weight-tile size in bf16 ELEMENTS (bytes/2): [tsi rows x d u8 | tsi x d/128 bf16]."""
    return (tsi * (d + (d // group) * 2) + 63) // 64 * 32


def config_shipped():
    """The configuration shipped in the 4B chain today (QKV/attn aside: MLP_XDNA arm)."""
    return dict(
        name="shipped (TSI=4, depth=2, misc object D-wide)",
        d=2560, ff=9728, ff_pad=10240, n=8, tsi=4, depth=2,
        misc_obj=None,        # D-wide
        weight_memtile=False, memtile_blocks=0, memtile_block_bytes=0, stage_depth=0,
        text_call_sites=96,   # 3 matrices x 32 tiles/col ... measured 96 at tsi=8; see note
    )


def config_proposed_b():
    """Option B of the note: MemTile-staged weight channel + TSI=2 + fifo depth 4."""
    return dict(
        name="proposed B (TSI=2, depth=4, MemTile staging)",
        d=2560, ff=9728, ff_pad=10240, n=8, tsi=2, depth=4,
        misc_obj=None,
        weight_memtile=True, memtile_blocks=4, memtile_block_bytes=32 * 1024, stage_depth=8,
        text_call_sites=96,
    )


def config_proposed_a():
    """Option A of the note: shipped L1/layout, only the TaskGroup structure changes."""
    c = config_shipped()
    c["name"] = "proposed A (per-chunk phase interleave, no budget change)"
    return c


def evaluate(c):
    """Return a list of (axis, measured, budget, ok, detail) for one configuration."""
    d, ff, ffp, n = c["d"], c["ff"], c["ff_pad"], c["n"]
    tsi = c["tsi"]
    rows = []
    chunked = (ff % d) != 0
    ff_pc = ffp // n if chunked else ff // n
    d_pc = d // n
    wu = wtile_units(tsi, d)
    mo = c["misc_obj"] or d

    # ---- whole objects / divisibility -------------------------------------------------
    rows.append(("object integrity (D % misc object)",
                 mo, d, d % mo == 0,
                 f"misc object {mo} divides D {d}; fractional deliveries are forbidden"))
    rows.append(("object integrity (FF/N % TSI)",
                 tsi, ff_pc, ff_pc % tsi == 0,
                 f"tile rows {tsi} divide gate/up rows per core {ff_pc}"))
    rows.append(("object integrity (D/N % TSI)",
                 tsi, d_pc, d_pc % tsi == 0, f"tile rows {tsi} divide down rows per core {d_pc}"))

    # ---- tile alignment (load_v) ------------------------------------------------------
    tile_bytes = wu * 2
    rows.append(("tile byte alignment (64 B)", tile_bytes % 64, 64, tile_bytes % 64 == 0,
                 f"one {tsi}-row tile is {tile_bytes} B; a mid-cache-line tile base is the "
                 f"documented load_v footgun"))

    # ---- L1 ---------------------------------------------------------------------------
    misc_bytes = 2 * (mo * 2)                       # depth 2
    weight_bytes = c["depth"] * (wu * 2)
    out_bytes = 2 * (d_pc * 2)                      # depth 2
    if chunked:
        persistent = 2 * (d * 2) + 2 * (ff_pc * 2) + (d_pc * 2)
    else:
        persistent = 2 * (d * 2) + (ff * 2) + 2 * (ff_pc * 2) + (d_pc * 2)
    stack = 0x1000 if chunked else 0x800
    total = misc_bytes + weight_bytes + out_bytes + persistent + stack
    rows.append(("L1 total (misc+weight+out+persistent+stack)", total, L1_BYTES,
                 total <= L1_BYTES,
                 f"misc={misc_bytes} weight={weight_bytes} out={out_bytes} "
                 f"persistent={persistent} stack={stack}"))

    # ---- shim channels: PER DIRECTION, not summed -------------------------------------
    # The design's own accounting ("misc(1) + weight(N) = 9 at N=8", against
    # get_shim_dma_limit() = 16) is the INPUT side; the output side carries N drains.  Summing
    # the two directions is wrong and would "fail" the shipped config, which places fine.
    shim_in = 1 + n
    shim_out = n
    rows.append(("shim input channels (misc 1 + weight N)", shim_in, 16, shim_in <= 16,
                 "the device-wide limit that stops this design at N=8 and not N=16"))
    rows.append(("shim output channels (out N)", shim_out, 16, shim_out <= 16,
                 "the second direction; independent of the input budget"))

    n_w_tiles = ff_pc // tsi
    objs_tg1 = 3 * (d // mo) + n_w_tiles          # per column: misc fills + ONE matrix's tiles
    bd_info = (f"fill/object counts for the heaviest TaskGroup: misc {d // mo} obj/fill x3, "
               f"weight {n_w_tiles} obj/fill.  NOT asserted: this tree has two conflicting "
               f"measurements of how aiecc bills them (qkv's misc channel needed per-object "
               f"fills at 11 objects, while this design's 320-object weight fill places fine), "
               f"so treat any threshold here as unverified until aiecc says otherwise.")
    rows.append(("BD billing (informational)", objs_tg1, BD_OTHER, True, bd_info))

    # ---- BD length / wrap -------------------------------------------------------------
    bd_words_ok = (d * 2 // 4) <= BD_LEN_WORDS_MAX
    rows.append(("single BD length (32-bit words)", d * 2 // 4, BD_LEN_WORDS_MAX, bd_words_ok,
                 f"a D-wide bf16 run is {d * 2 // 4} words"))
    rows.append(("wrap dimension", min(d, n_w_tiles), WRAP_MAX, min(d, n_w_tiles) <= WRAP_MAX,
                 "mlir-aie verifyStridesWraps caps a wrap dim at 1023"))

    # ---- MemTile staging (option B) ---------------------------------------------------
    if c["weight_memtile"]:
        mt_bytes = c["memtile_blocks"] * c["memtile_block_bytes"]
        rows.append(("MemTile capacity", mt_bytes, MEMTILE_BYTES, mt_bytes <= MEMTILE_BYTES,
                     f"{c['memtile_blocks']} in-flight blocks x {c['memtile_block_bytes']} B"))
        mt_bds = -(-c["memtile_block_bytes"] // (32 * 1024)) + 2   # data BDs + 2 for bookkeeping
        rows.append(("MemTile BDs", mt_bds, BD_MEMTILE, mt_bds <= BD_MEMTILE, "48 available"))
        # The stage's own depth, measured: 8/10/12 build (14/16 -> "Allocator exhausted
        # available BD IDs (maximum 24 available for channel 0)").  misc 2 + out 2 + weight
        # depth + stage depth share that pool.
        sd = c.get("stage_depth", 0) or 0
        bd_ids = sd + c["depth"] + 2 + 2
        rows.append(("shim BD-ID pool (fifo depths sum)", bd_ids, BD_ID_SHIM,
                     bd_ids <= BD_ID_SHIM,
                     "stage + weight + misc(2) + out(2); measured 12 builds, 14 does not"))
        mt_locks = c["memtile_blocks"] * 2 + 2
        rows.append(("MemTile locks", mt_locks, LOCK_MEMTILE, mt_locks <= LOCK_MEMTILE,
                     "64 available, max value 63"))

    # ---- program memory ---------------------------------------------------------------
    # Measured, not modelled: the shipped 4B build is 9728-10992 B of .text against 16384 B
    # (67%), and tile size does NOT change it because the per-tile loop is emitted once.  The
    # failure mode this tree actually hit (qkv_head_dp, and the swiglu arm at tsi=8) is a
    # SMALL constant loop with a big body getting fully unrolled at ~143 B per call site.
    text_measured = 10992
    rows.append(("program memory (.text, measured shipped build)", text_measured, PROGRAM_MEM,
                 text_measured <= PROGRAM_MEM,
                 f"measured on the 4B build; the overflow risk is a small constant loop with a "
                 f"big body being fully unrolled (~{TEXT_PER_CALL_SITE} B per call site), which "
                 f"is what forced qkv_head_dp's runtime tile bound"))

    return rows


CONFIGS = [config_shipped(), config_proposed_a(), config_proposed_b()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = {}
    for c in CONFIGS:
        rows = evaluate(c)
        out[c["name"]] = [
            dict(axis=a, measured=m, budget=b, ok=bool(ok), detail=d) for a, m, b, ok, d in rows
        ]
        if not args.json:
            print(f"\n=== {c['name']} ===")
            for axis, m, b, ok, detail in rows:
                print(f"  [{'PASS' if ok else 'FAIL'}] {axis}: {m} vs {b}")
                if not ok:
                    print(f"         -> {detail}")
        n_fail = sum(1 for *_, ok, _ in rows if not ok)
        if not args.json:
            print(f"  -> {'ALL ASSERTIONS PASS' if n_fail == 0 else f'{n_fail} FAILING AXIS/ES'}")

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print("\nKey reading: the shipped config passes every axis; option B (MemTile staging + "
              "TSI=2 + depth=4) also passes every ASSERT -- but it was MEASURED to be ~6% SLOWER "
              "on device (36.0 vs 38.4 GB/s, same-session interleaved A/B), because the shim's "
              "24-entry BD-ID pool (consumed by the fifo depths) caps the stage at ~12 tiles, so "
              "the deep buffer the plan relied on never materialises.  Budgets passing is not "
              "evidence of a win; see the note's section 6.")


if __name__ == "__main__":
    main()
