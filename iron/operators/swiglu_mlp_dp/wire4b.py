# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire layout for the CHUNKED-DOWN arm of swiglu_mlp_dp (Qwen3-4B shapes).

The 0.6B wire is "one row per weight row": [m*K u8 | m*(K/128) bf16 scales] per (column, tile)
block, and the gate/up rows and the down rows are reconciled by the shared-tile identity
TSI_GU*WROW_D == TSI_D*WROW_FF -- which needs FF/D to be an integer.  At 4B (9728/2560 = 3.8) the
model's FF is therefore PADDED to a whole multiple of D (FF_PAD = 4*D = 10240) in the GATHER
domain, and the down projection is CHUNKED along K:

  * every wire row is one D-wide K-chunk row: [k u8 | (k/BLOCK) bf16 scales], k = D for gate/up
    AND for every down chunk (uniform, no short tail);
  * the gate/up buffers are column-major [column c][tile j], carry the model's REAL FF rows (no
    zero-weight pad rows any more) and are in the ROW ORDER `wire_layout` returns -- not natural
    order.  See `wire_layout`'s docstring: the gathered gh is core-major with ff_pad//cols slots
    per core, and Wd's 128-column scale granularity forces every 128-slot block of gh to carry ONE
    model scale-group, which a natural contiguous row assignment cannot satisfy at 4B
    (ff//cols = 1216 = 9.5 groups);
  * the down buffer is TILE-BLOCKED chunk-major [chunk ci][column c][tile j], because that is the
    order the core walks it (chunk-outer, tile-inner) and the order the runtime's tg3 fills issue;
    its K dimension IS the padded FF (`ff=ff_pad`), and its COLUMNS follow `wire_layout`'s
    wd_cols (the 64 zero slots each core's gathered slice carries are the wire's pad columns);
  * every tile block is rounded up to a 64 B multiple (4 rows * 2600 B = 10400 -> 10432; 8 rows is
    exact at 20800), because a tile base landing mid-cache-line is the load_v footgun.  The pad
    lives inside the block and the runtime's per-column fill is one contiguous copy of the whole
    column, counted in TILE blocks.  The kernel reads only its [u8 | scales] prefix, so the pad is
    transferred but never touched.

`pack_*` write that layout and `unpack_*` are their exact inverses -- the CPU reference gate uses
the latter, and the pack/unpack round-trip self-test below pins the index math.  The caller is
responsible for handing `pack_*` matrices already re-ordered by `wire_layout`.

Scale convention: `sc_bits` is a uint16 array of BF16 bit patterns with shape (M, K//128), i.e.
exactly what `tensor.to(torch.bfloat16).view(torch.uint16).numpy()` produces -- the same domain the
0.6B arm's packer feeds the kernel.
"""

import numpy as np


def wire_layout(cols: int, ff: int, ff_pad: int, tsi: int, group: int = 128):
    """The ROW/COLUMN ORDER the chunked wire's gate/up and down matrices must be built in.

    Why an order at all.  At 4B the all-gathered gh is core-major: ff_pad//cols (1280) slots per
    core, filled by that core's emit rounds, which for a real-row gate/up wire gives
    `gh[c*1280 + q] = product(model row c*1216 + q)` and ZERO for the last (ff_pad-ff)//cols (64)
    slots of each core (the L1 tail the core zeroes -- design_ours_int8.py).  Wd multiplies gh by
    COLUMNS and reads its scales per `group` (128) consecutive columns, so every 128-slot block of
    gh must carry the weights of exactly ONE model scale-group.  ff//cols model rows per core is
    NOT a whole number of groups at 4B (1216 = 9.5*128), so no contiguous natural assignment can
    satisfy that: the wire gives each core `full_groups` whole model groups plus a `tail_rows`-row
    chunk taken from the MODEL'S END, and the cores take those chunks in order.

    At 4B/8 columns: 9 whole groups (1152 rows) + 64 rows from the model tail per core; the 8 tail
    chunks cover model rows 9216..9727 exactly; each core's last 128-slot gh block is 64 model
    columns (all inside one model group, so one scale is still exact) + 64 zero slots.

    Returns
      gu_rows[q]          : model row the q-th gate/up wire row holds (wire order, column-major);
                            a PERMUTATION of range(ff)
      wd_cols[k]          : model column the k-th Wd wire column holds, or -1 for a zero column
      wd_block_group[b]   : model scale-group the b-th group-aligned block of Wd wire columns
                            takes its scales from
    """
    n = cols
    r = ff_pad // n
    r_gu = ff // n
    assert ff_pad % n == 0 and ff % n == 0, (ff_pad, ff, n)
    assert r % group == 0, f"gh slots per core ({r}) must be group-aligned"
    full_groups = r_gu // group
    tail_rows = r_gu - full_groups * group
    tail_start = n * full_groups * group
    assert tail_start + n * tail_rows == ff, (tail_start, n * tail_rows, ff)
    assert (full_groups * group) % tsi == 0 and tail_rows % tsi == 0, (
        f"both runs of a core's rows must divide TSI_GU={tsi} (tiles are {tsi} rows)")
    for c in range(n):
        start = tail_start + c * tail_rows
        assert start % group + tail_rows <= group, (
            f"core {c}'s tail chunk must lie inside ONE model group (else its gh block needs two "
            f"scales, which the wire cannot express)")
    gu_rows = np.empty(n * r_gu, dtype=np.int64)
    for c in range(n):
        dst = c * r_gu
        run = full_groups * group
        gu_rows[dst:dst + run] = np.arange(c * run, c * run + run)
        gu_rows[dst + run:dst + r_gu] = np.arange(
            tail_start + c * tail_rows, tail_start + (c + 1) * tail_rows)
        # a core's gh blocks are laid out below: `full_groups` whole groups, then the tail chunk
        # followed by the zero slots its own L1 tail provides.
    assert sorted(gu_rows.tolist()) == list(range(ff)), "gate/up rows are not a permutation"
    wd_cols = np.full(ff_pad, -1, dtype=np.int64)
    wd_block_group = np.empty(ff_pad // group, dtype=np.int64)
    for c in range(n):
        for t in range(r // group):
            b = c * (r // group) + t
            slot = c * r + t * group
            if t < full_groups:
                m0 = (c * full_groups + t) * group
                n_cols = group
                wd_block_group[b] = c * full_groups + t
            else:
                m0 = tail_start + c * tail_rows
                n_cols = tail_rows
                wd_block_group[b] = m0 // group
            wd_cols[slot:slot + n_cols] = np.arange(m0, m0 + n_cols)
    assert sorted(wd_cols[wd_cols >= 0].tolist()) == list(range(ff)), (
        "down columns are not a permutation of the model columns")
    for b, g in enumerate(wd_block_group):
        blk = wd_cols[b * group:(b + 1) * group]
        blk = blk[blk >= 0]
        assert len(blk) == 0 or (blk >= g * group).all() and (blk < (g + 1) * group).all(), (
            f"wire block {b} carries columns from outside model group {g}")
    # The crux: gh slot k is PRODUCED by gate/up wire row (c, j) and CONSUMED with Wd's column k,
    # so the two indexings must name the same model row -- otherwise the down matvec multiplies a
    # weight by a different row's activation, which is exactly the 0.156-cosine failure.
    for c in range(n):
        for j in range(r_gu):
            assert wd_cols[c * r + j] == gu_rows[c * r_gu + j], (
                f"gh slot {c * r + j} is produced from model row {gu_rows[c * r_gu + j]} but "
                f"consumed with model column {wd_cols[c * r + j]}")
    return gu_rows, wd_cols, wd_block_group


def pack_gu(u8: np.ndarray, sc_bits: np.ndarray, *, cols: int, tsi: int, d: int, ff: int,
            wtile_bytes: int) -> np.ndarray:
    """Wg (or Wu): [cols][tile j] blocks; column c owns FF/cols rows, each row k = D."""
    rpc = ff // cols
    n_tiles = rpc // tsi
    out = np.zeros(cols * n_tiles * wtile_bytes, np.uint8)
    for c in range(cols):
        for j in range(n_tiles):
            r0 = c * rpc + j * tsi
            blk = out[(c * n_tiles + j) * wtile_bytes:][:wtile_bytes]
            n = tsi * d
            blk[:n] = np.ascontiguousarray(u8[r0:r0 + tsi, :d]).reshape(-1)
            s = np.ascontiguousarray(sc_bits[r0:r0 + tsi, :d // 128]).reshape(-1)
            blk[n:n + s.size * 2] = s.view(np.uint8)
    return out


def pack_down(u8: np.ndarray, sc_bits: np.ndarray, *, cols: int, tsi: int, d: int,
              chunk_k: list, wtile_bytes: int) -> np.ndarray:
    """Wd: [chunk ci][column c][tile j] blocks; column c owns D/cols rows, chunk ci spans K=chunk_k[ci]."""
    dpc = d // cols
    n_tiles = dpc // tsi
    out = np.zeros(len(chunk_k) * cols * n_tiles * wtile_bytes, np.uint8)
    koff = 0
    for ci, k in enumerate(chunk_k):
        for c in range(cols):
            for j in range(n_tiles):
                r0 = c * dpc + j * tsi
                blk = out[((ci * cols + c) * n_tiles + j) * wtile_bytes:][:wtile_bytes]
                n = tsi * k
                blk[:n] = np.ascontiguousarray(u8[r0:r0 + tsi, koff:koff + k]).reshape(-1)
                s = np.ascontiguousarray(
                    sc_bits[r0:r0 + tsi, koff // 128:(koff + k) // 128]).reshape(-1)
                blk[n:n + s.size * 2] = s.view(np.uint8)
        koff += k
    return out


def unpack_gu(packed: np.ndarray, *, cols: int, tsi: int, d: int, ff: int,
              wtile_bytes: int) -> tuple:
    """Inverse of pack_gu -> ((FF, D) u8, (FF, D//128) uint16)."""
    rpc = ff // cols
    n_tiles = rpc // tsi
    u8 = np.zeros((ff, d), np.uint8)
    sc = np.zeros((ff, d // 128), np.uint16)
    for c in range(cols):
        for j in range(n_tiles):
            r0 = c * rpc + j * tsi
            blk = packed[(c * n_tiles + j) * wtile_bytes:][:wtile_bytes]
            n = tsi * d
            u8[r0:r0 + tsi, :d] = blk[:n].reshape(tsi, d)
            sc[r0:r0 + tsi, :d // 128] = blk[n:n + tsi * (d // 128) * 2].view(np.uint16).reshape(
                tsi, d // 128)
    return u8, sc


def unpack_down(packed: np.ndarray, *, cols: int, tsi: int, d: int, chunk_k: list,
                wtile_bytes: int) -> tuple:
    """Inverse of pack_down -> ((D, sum(chunk_k)) u8, (D, sum(chunk_k)//128) uint16)."""
    dpc = d // cols
    n_tiles = dpc // tsi
    ktot = sum(chunk_k)
    u8 = np.zeros((d, ktot), np.uint8)
    sc = np.zeros((d, ktot // 128), np.uint16)
    koff = 0
    for ci, k in enumerate(chunk_k):
        for c in range(cols):
            for j in range(n_tiles):
                r0 = c * dpc + j * tsi
                blk = packed[((ci * cols + c) * n_tiles + j) * wtile_bytes:][:wtile_bytes]
                n = tsi * k
                u8[r0:r0 + tsi, koff:koff + k] = blk[:n].reshape(tsi, k)
                sc[r0:r0 + tsi, koff // 128:(koff + k) // 128] = (
                    blk[n:n + tsi * (k // 128) * 2].view(np.uint16).reshape(tsi, k // 128))
        koff += k
    return u8, sc


def _selftest():
    rng = np.random.default_rng(0)
    D, FF_REAL, FF, COLS, TSI = 2560, 9728, 10240, 8, 4
    wtile = (TSI * (D + (D // 128) * 2) + 63) // 64 * 64         # 10432 (4 rows, padded)
    assert wtile % 64 == 0, wtile
    chunk_k = [D] * (FF // D)                                     # [2560] * 4, uniform
    pad_rows = FF - FF_REAL                                       # the gather pad (down K only)

    # gate/up: REAL rows only (the core zeroes its unwritten g/u tail instead)
    u8g = rng.integers(0, 256, (FF_REAL, D), dtype=np.uint8)
    scg = rng.integers(0, 65535, (FF_REAL, D // 128), dtype=np.uint16)
    p = pack_gu(u8g, scg, cols=COLS, tsi=TSI, d=D, ff=FF_REAL, wtile_bytes=wtile)
    assert p.size == COLS * (FF_REAL // COLS // TSI) * wtile, p.size
    u8r, scr = unpack_gu(p, cols=COLS, tsi=TSI, d=D, ff=FF_REAL, wtile_bytes=wtile)
    assert np.array_equal(u8r, u8g) and np.array_equal(scr, scg), "pack_gu/unpack_gu mismatch"

    # down: K is the PADDED FF, pad columns carry zero weights (u8=128)
    u8d = rng.integers(0, 256, (D, FF), dtype=np.uint8)
    scd = rng.integers(0, 65535, (D, FF // 128), dtype=np.uint16)
    u8d[:, FF_REAL:] = 128
    pd = pack_down(u8d, scd, cols=COLS, tsi=TSI, d=D, chunk_k=chunk_k, wtile_bytes=wtile)
    assert pd.size == len(chunk_k) * COLS * (D // COLS // TSI) * wtile, pd.size
    u8dr, scdr = unpack_down(pd, cols=COLS, tsi=TSI, d=D, chunk_k=chunk_k, wtile_bytes=wtile)
    assert np.array_equal(u8dr, u8d) and np.array_equal(scdr, scd), "pack_down/unpack_down mismatch"

    print(f"wire4b selftest OK  wtile_bytes={wtile}  chunk_k={chunk_k}  "
          f"gather pad rows (down K only)={pad_rows}")

    # wire_layout: the order the caller must build the three matrices in (see its docstring)
    gu_rows, wd_cols, wd_blk = wire_layout(COLS, FF_REAL, FF, TSI, 128)
    assert sorted(gu_rows.tolist()) == list(range(FF_REAL)), "gu rows are not a permutation"
    assert (wd_cols >= 0).sum() == FF_REAL, "down wire must carry every model column once"
    assert (wd_cols < 0).sum() == FF - FF_REAL, "down wire pads are the gather pad"
    assert sorted(wd_cols[wd_cols >= 0].tolist()) == list(range(FF_REAL))
    assert len(wd_blk) == FF // 128, len(wd_blk)
    print(f"  wire_layout: {gu_rows.size} gate/up rows (a permutation of 0..{FF_REAL - 1}), "
          f"{int((wd_cols >= 0).sum())} model down cols + {int((wd_cols < 0).sum())} zero cols, "
          f"{len(wd_blk)} group-aligned scale blocks")
    print(f"  Wg bytes = {p.size}  Wd bytes = {pd.size}")


if __name__ == "__main__":
    _selftest()
