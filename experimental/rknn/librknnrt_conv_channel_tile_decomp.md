# librknnrt.so Conv ChannelTile decompilation

Source binary: `experimental/rknn/librknnrt.so`

Command style used:

```sh
rabin2 -zz experimental/rknn/librknnrt.so | rg 'min_weight_banks|ChannelTile'
r2 -q -e scr.color=false -e bin.relocs.apply=true -c 'aaa' -c 'axt @ 0x60f388' -c q experimental/rknn/librknnrt.so
r2 -q -e scr.color=false -e bin.relocs.apply=true -c aa -c 's 0x00387e3c' -c af -c pdf -c q experimental/rknn/librknnrt.so
```

## Target string and xrefs

The string:

```text
0x0060f388 min_weight_banks > 3 && m_Target == RKNNTarget::RKNPU_F2, do conv with ChannelTile
```

has two code xrefs:

```text
fcn.00387e3c @ 0x0038831c
fcn.00388a18 @ 0x00388e84
```

Both functions implement the same high-level decision: compute a minimum number of
weight CBUF banks for a proposed conv tile, then switch to ChannelTile when that
number exceeds the target-specific bank threshold.

The logging/plumbing callees used on the diagnostic path are not conv tiling
logic:

```text
fcn.000ff6e0  stream/string construction
fcn.000dae38  append text/integer to stream
fcn.000dfbd0  emit stream/log
operator new/delete and abort
```

They are excluded from the tiling pseudo-code below except where a branch emits
one of the ChannelTile messages.

## Related tiling-string xrefs

After starting from the F2 ChannelTile string, the other conv tiling/reuse
strings in `librknnrt.so` lead to a second layer-level tiling path and a set of
diagnostic helpers. These are the useful xrefs:

| String address | String | Xref function | Role |
| --- | --- | --- | --- |
| `0x00607178` | `CNA feature group0: %d` | `fcn.00101be8` | Register-task/resource-group dump. Confirms the runtime tracks CNA feature/weight/CSC groups, but it is not the planner. |
| `0x0060c768` | `failed to update data and weight reuse!` | `fcn.00307198`, `fcn.00312328` | Tile-table reuse updater. These functions validate/update data/weight reuse bits after a split plan is built. |
| `0x0062cc50` | `banks num invalid, input_bank_num: %d, weight_bank_num: %d` | `fcn.005c6b50` | Large workload/layer config function. Validates bank allocation and reuse strategy while emitting NPU workload configuration. |
| `0x0062cc90` | `Invalid reuse strategy!` | `fcn.005c6b50` | Same workload config validator. |
| `0x0062ce00` | `Conv min_weight_banks > 3, OutputName : %s` | `fcn.005d4ac0` | Layer-level conv tiler. This is the broader function that builds split/Y/K tile records and logs when a conv crosses the F2-style weight-bank threshold. |
| `0x0062ce50` | `MC K Tile Failed, min kernel step %d, but get %d` | `fcn.005d4ac0` | Same layer-level conv tiler; reports failed multi-core K/kernel tiling. |
| `0x0062ce30` | `failed to tile argb mode layer!` | `fcn.005d4ac0` | Same layer-level conv tiler; ARGB-mode tile generation failure path. |
| `0x0062cf58` | `Unknown split Method -> %d` | `fcn.005f1998` | Abort helper for invalid split-method enum. |
| `0x0062cf78` | tile-table header with `xstart`, `ystart`, `kstart`, `data reuse`, `weight reuse`, `mc_treat_by_*` | `fcn.005f1cd0` | Tile-table dump/formatter. Shows the actual per-tile fields RKNN uses after planning. |
| `0x0062d040` | `illegal tiling method %d` | `fcn.005f38f0` | Conv K/X split composer. Emits fatal diagnostics when a split method cannot be represented. |
| `0x0062d298` | `X tile buffer overflow!` | `fcn.005f51c0` | Top-level conv Y/K/multicore tiler. Builds nested tile vectors and emits X/Y/MC diagnostics. |
| `0x0062d508` | `Generate Y config crash!` | `fcn.005f51c0` | Same top-level conv tiler; reports failed Y tile configuration. |
| `0x0062d7b8` | `Illegal mc Y type` | `fcn.005f51c0` | Same top-level conv tiler; validates multi-core Y treatment. |
| `0x0062d8e8` | `Illegal mc K type` | `fcn.005f51c0` | Same top-level conv tiler; validates multi-core K treatment. |

The important separation is:

- `fcn.00387e3c` and `fcn.00388a18` answer: should this conv switch to
  ChannelTile because bank pressure is too high?
- `fcn.005d4ac0`, `fcn.005f38f0`, and `fcn.005f51c0` answer: after a
  conv is being tiled, how are Y/K/X split records chosen, checked, nested,
  and mapped to multi-core treatment?
- `fcn.00307198`, `fcn.00312328`, `fcn.005c6b50`, and `fcn.005f1cd0` answer:
  how are data/weight reuse flags, bank counts, and tile records validated or
  printed?

## Function inventory

The stripped binary does not expose useful C++ names for the compiler internals,
so the coverage set below is built from `rabin2 -zz` string xrefs and direct
callee inspection around the conv tiling cluster. These are the functions that
participate in conv tiling rather than generic logging, allocation, or vector
bookkeeping:

| Function | Size | Role | Decompilation status |
| --- | ---: | --- | --- |
| `fcn.00387e3c` | `1436` | Main ChannelTile predicate. Computes candidate channel/K tile, minimum data tile, minimum weight banks, then checks target thresholds. | Full pseudo-code below. |
| `fcn.00388a18` | `1248` | Fixed-16 sibling ChannelTile predicate. Same threshold table with a narrower candidate setup. | Full pseudo-code below. |
| `fcn.00384ed0` | `628` | Minimum weight-bank estimator used by both ChannelTile predicates and planner paths. | Full pseudo-code below. |
| `fcn.00387530` | not separately sized in this note | Base tile-shape selector. Chooses atomic count/step pair from shape vectors and target limits. | Full pseudo-code below. |
| `fcn.00383338` | not separately sized in this note | Minimum data-tile estimator. Converts spatial span and aligned channel count into CBUF data pressure. | Full pseudo-code below. |
| `fcn.003873e8` | not separately sized in this note | Channel tile clamp/tuner. Rounds requested channel tiles and halves until legal. | Full pseudo-code below. |
| `fcn.00385988` | `224` | Weight/K split feasibility helper. Converts spare banks and K groups into a legal K step or `-1`. | Full pseudo-code below. |
| `fcn.00384ba0`, `fcn.00384d48`, `fcn.00389078`, `fcn.000e1100`, `fcn.000f4758` | small | Atomic-width limit selectors used by the estimators/tuners. | Selector pseudo-code below. |
| `fcn.005d4ac0` | `11792` | Layer-level conv split planner. Builds Y/K tile records and logs F2 `min_weight_banks > 3`. | High-level pseudo-code below. |
| `fcn.005f38f0` | `6344` | K/X split composer used from `fcn.005f51c0`. Produces per-split vectors and rejects illegal tiling methods. | High-level pseudo-code below. |
| `fcn.005f51c0` | `18488` | Top-level conv Y/K/multicore tiler. Calls `fcn.005f38f0`, validates X/Y/MC split state, builds nested vectors. | High-level pseudo-code below. |
| `fcn.00307198`, `fcn.00312328` | not separately sized in this note | Data/weight reuse update passes over finished tile lists. | Shared pseudo-code below. |
| `fcn.005c6b50` | not separately sized in this note | Workload bank/reuse field validator and register-task config helper. | Relevant checks below. |
| `fcn.005f1cd0` | not separately sized in this note | Tile-table dump helper for `xstart`, `ystart`, `kstart`, reuse, and MC fields. | Formatter pseudo-code below. |
| `fcn.005f1998` | not separately sized in this note | Invalid split-method abort helper. | Small pseudo-code below. |

Functions such as `fcn.000ff6e0`, `fcn.000dae38`, `fcn.000dfbd0`,
`operator new/delete`, `memcpy`, `memmove`, and vector constructors/destructors
appear in these functions but are plumbing, not conv tiling logic.

## Target tags

The target is stored as a 32-bit tag at offset `+0x0` of the conv tiler object.
The code compares the tag as an integer, which reads as reversed ASCII on LE:

```c
enum RKNNTargetTag {
    RKNPU_F2 = 0x46495247, // bytes "GRIF"
    RKNPU_F3 = 0x46495248, // bytes "HRIF"
    RKNPU_W2 = 0x57494e46, // bytes "FNIW"
    RKNPU_W1 = 0x57494e45, // bytes "ENIW"
};
```

## Inferred object fields

The stripped binary gives offsets but no field names. These names are inferred
from how each value is used by the tiling math.

```c
struct ConvTileCtx {
    uint32_t target;              // +0x00, RKNNTargetTag
    int32_t  atomic_c_total;      // +0x28
    int32_t  limit_4_a;           // +0x2c
    int32_t  limit_4_b;           // +0x30
    int32_t  limit_8_a;           // +0x34
    int32_t  limit_8_b;           // +0x38
    int32_t  limit_16_a;          // +0x3c
    int32_t  limit_16_b;          // +0x40
    int32_t  limit_32_a;          // +0x44
    int32_t  limit_32_b;          // +0x48
    int32_t  limit2_4;            // +0x4c
    int32_t  limit2_8;            // +0x50
    int32_t  limit2_16_or_fmt9;   // +0x54
    int32_t  limit2_32;           // +0x58
    int32_t  banks_for_normal;    // +0x60, for most targets
    int32_t  bank_granularity;    // +0x64
    int32_t  atomic_c_hw;         // +0x68, frequently shifted by 3
    int32_t  banks_for_f2;        // +0xa0, used for F2 target
    int32_t  special_align;       // +0x220, used by fcn.00387530 format==9 path
};
```

## fcn.00387e3c: main conv ChannelTile test

This is the first and larger xref function. It computes a candidate K/channel
tile, derives data/weight bank pressure, then returns `1` when ChannelTile should
be used.

Key addresses:

```text
0x00388150 call fcn.00384ed0      ; returns min_weight_banks in w0
0x00388158 cmp w0, 3              ; F2 threshold
0x00388180 branch to F2 log
0x00388188 cmp w0, 7              ; F3/W1 threshold
0x00388198 branch to F3 log
0x0038819c cmp w0, 8
0x003881a0 min_weight_banks == 8 skips W2
0x003881b0 branch to W2 log
0x003881c0 branch to W1 log
```

Pseudo-code:

```c
bool should_use_channel_tile_main(
    ConvTileCtx *ctx,
    int dims0[4],        // first shape vector copied into a small vector
    int arg2_flag,
    int dims1[2],
    bool early_disable,
    bool swap_dims,
    bool use_alt_shape,
    bool force_fallback)
{
    if (early_disable)
        return false;

    int in0;
    int in1;
    int out_h;
    int out_w;
    int stride_like;
    int fmt_or_dtype;

    if (swap_dims) {
        in0 = dims0[1];
        in1 = dims0[0];
    } else {
        in0 = dims0[0];
        in1 = dims0[1];
    }

    out_h = dims0[2];
    out_w = dims0[3];
    stride_like = dims1[0];
    fmt_or_dtype = dims1[1];

    TileShape tile_shape = select_tile_shape(ctx, arg2_flag);
    int tile_count = tile_shape.count;
    if (tile_count == 0)
        abort_bad_tiling(tile_count);

    int atomic = ctx->atomic_c_total;
    int atom_groups;
    if (ctx->target == 0 && tile_count == 4)
        atom_groups = div_round_up_signed(atomic, 16);
    else
        atom_groups = div_round_up_signed(atomic, 8) / tile_count;

    int c_step = tile_count * 8;
    int shape_limit = pick_limit_b_by_atomic(ctx, c_step, fmt_or_dtype);

    int aligned_tile_count = round_up_to_multiple(tile_count + in1, atom_groups);

    // Programs/checks an initial data-vs-weight split into local vectors.
    check_base_tiling(
        ctx,
        /*mode=*/0,
        /*start=*/0,
        /*shape_vec=*/dims0,
        /*scratch_vec=*/local_vec,
        /*out=*/&tile_shape,
        /*atomic_step=*/c_step,
        /*unknown=*/0,
        /*alt=*/swap_dims,
        /*final=*/false);

    int total_atomic_tiles = ctx->atomic_c_total / c_step;
    int data_span_tiles = (ctx->atomic_c_hw * 8) / c_step;
    int aligned_start = round_up_to_multiple(in1, total_atomic_tiles);

    int modulo_base;
    if (c_step == 8)
        modulo_base = ctx->limit_8_a;
    else if (c_step == 16)
        modulo_base = ctx->limit_16_a;
    else if (c_step == 32)
        modulo_base = ctx->limit_32_a;
    else if (c_step == 64)
        modulo_base = -1;
    else
        abort_bad_tiling(c_step);

    int aligned_for_limit = round_up_to_multiple(aligned_start, total_atomic_tiles);
    int split_residue = aligned_for_limit % modulo_base;
    if (split_residue != 0) {
        int reduced = choose_power2_reduced_tile(aligned_for_limit, split_residue,
                                                 modulo_base);
        aligned_for_limit = aligned_for_limit - split_residue + reduced;
    }

    int bank_budget = (ctx->target == RKNPU_F2)
        ? ctx->banks_for_f2
        : ctx->banks_for_normal;

    int min_data_tile = estimate_min_data_tile(ctx,
        (out_w - 1) * stride_like + 1,
        aligned_tile_count,
        c_step);

    int min_weight_banks = estimate_min_weight_banks(
        ctx,
        /*mode=*/0,
        /*tile spatial dims=*/dims0[2], dims0[3],
        /*aligned input tile=*/aligned_for_limit,
        /*k tile=*/in0,
        /*tile_count=*/tile_count,
        /*alt=*/use_alt_shape,
        /*force=*/force_fallback,
        /*first=*/true);

    int remaining_bank_space =
        (bank_budget - min_weight_banks) * ctx->bank_granularity / min_data_tile;

    if (min_weight_banks > 3 && ctx->target == RKNPU_F2) {
        log("min_weight_banks > 3 && m_Target == RKNNTarget::RKNPU_F2, do conv with ChannelTile");
        return true;
    }

    if (min_weight_banks > 7 && ctx->target == RKNPU_F3) {
        log("min_weight_banks > 7 && m_Target == RKNNTarget::RKNPU_F3, do conv with ChannelTile");
        return true;
    }

    if (min_weight_banks > 8 && ctx->target == RKNPU_W2) {
        log("min_weight_banks > 8 && m_Target == RKNNTarget::RKNPU_W2, do conv with ChannelTile");
        return true;
    }

    if (min_weight_banks > 7 && ctx->target == RKNPU_W1) {
        log("min_weight_banks > 7 && m_Target == RKNNTarget::RKNPU_W1, do conv with ChannelTile");
        return true;
    }

    // After the ChannelTile diagnostic block, the function continues with a
    // viability loop. It halves the selected channel limit until the remaining
    // bank space can hold the required data tile. If it cannot, it returns true.
    int channel_limit = shape_limit;
    if (arg2_flag == 0x0a || arg2_flag == 0x10)
        atom_groups = channel_limit;

    while (in0 < channel_limit)
        channel_limit >>= 1;

    if (out_h_tile_requirement(out_h, stride_like) > remaining_bank_space)
        return true;

    return false;
}
```

## fcn.00388a18: sibling ChannelTile test

This function is a compact sibling. It uses a fixed `c_step = 16`/`tile_count = 2`
path and then applies the same target threshold table.

Key addresses:

```text
0x00388bf8 call fcn.00384ed0      ; returns min_weight_banks in w0
0x00388c04 cmp w0, 3
0x00388c24 branch to F2 log
0x00388c2c cmp w0, 7
0x00388c3c branch to F3 log
0x00388c44 min_weight_banks == 8 skips W2
0x00388c54 branch to W2 log
0x00388c64 branch to W1 log
```

Pseudo-code:

```c
bool should_use_channel_tile_fixed16(
    ConvTileCtx *ctx,
    int dims0[4],
    int dims1[2],
    bool swap_dims,
    bool alt,
    bool final)
{
    int k_tile = swap_dims ? dims0[1] : dims0[0];
    int c_start = swap_dims ? dims0[0] : dims0[1];
    int out_h = dims0[2];
    int out_w = dims0[3];

    check_base_tiling(ctx, 0, swap_dims, local_vec, out_vec,
                      /*atomic_step=*/16, 0, 0, false);

    int atomic_tiles = div_round_up_signed(ctx->atomic_c_total, 16);
    int aligned_c = round_up_to_multiple(c_start, atomic_tiles);
    int limit_mod = ctx->limit_16_a;
    int residue = aligned_c % limit_mod;
    if (residue != 0)
        aligned_c = aligned_c - residue + choose_power2_reduced_tile(aligned_c, residue,
                                                                     limit_mod);

    int bank_budget = (ctx->target == RKNPU_F2)
        ? ctx->banks_for_f2
        : ctx->banks_for_normal;

    int min_data_tile = estimate_min_data_tile(ctx, (out_w - 1) * dims1[0] + 1,
                                               aligned_c, 16);

    int min_weight_banks = estimate_min_weight_banks(
        ctx, 0, dims0[2], dims0[3], aligned_c, k_tile,
        /*tile_count=*/2, alt, final, true);

    int remaining_bank_space =
        (bank_budget - min_weight_banks) * ctx->bank_granularity / min_data_tile;

    if (min_weight_banks > 3 && ctx->target == RKNPU_F2)
        return log_channel_tile_and_true(F2);
    if (min_weight_banks > 7 && ctx->target == RKNPU_F3)
        return log_channel_tile_and_true(F3);
    if (min_weight_banks > 8 && ctx->target == RKNPU_W2)
        return log_channel_tile_and_true(W2);
    if (min_weight_banks > 7 && ctx->target == RKNPU_W1)
        return log_channel_tile_and_true(W1);

    int channel_limit = ctx->limit_16_b;
    while (k_tile < channel_limit)
        channel_limit >>= 1;

    if (out_h_tile_requirement(out_h, dims1[0]) > remaining_bank_space)
        return true;

    return false;
}
```

## fcn.00384ed0: estimate minimum weight banks

This is the core helper called immediately before the target threshold checks.

Inputs by calling convention:

```text
x0 ctx
w1 mode/flag
w2 tile spatial dim A
w3 tile spatial dim B
w4 aligned channel/count
w5 k tile
w6 tile_count or forced bank units
w7 bool alt
stack+0x70 bool force_2
stack+0x78 bool force_small
stack+0x80 format/dtype
```

Pseudo-code:

```c
int estimate_min_weight_banks(
    ConvTileCtx *ctx,
    bool mode,
    int tile_a,
    int tile_b,
    int aligned_channels,
    int k_tile,
    int forced_units,
    bool alt,
    bool force_2,
    bool force_small,
    int fmt_or_dtype)
{
    int element_step = 4;
    int tile_count = 2;

    // For some target tags or flags the routine uses 4 banks instead of 2.
    if (is_non_f2_like_target(ctx->target) || force_2 || force_small)
        tile_count = 4;

    if (forced_units != 0)
        element_step = forced_units * 8;

    if (fmt_or_dtype == 9) {
        element_step = normalize_fmt9_step(fmt_or_dtype);
        tile_count = pick_limit_b_by_atomic(ctx, element_step, fmt_or_dtype);
    } else {
        tile_count = pick_limit_b_by_atomic(ctx, element_step, fmt_or_dtype);
    }

    if (alt)
        tile_count = 2;

    if (!mode && force_2) {
        k_tile = 1;
        tile_count = 1;
    } else if (mode) {
        k_tile = 2;
        tile_count = 2;
    }

    int weight_bytes =
        tile_a * tile_count * tile_b * aligned_channels * element_step;
    int weight_banks = ceil_div_signed(weight_bytes, 8);

    if (ctx->target != 0) {
        int cacc_banks = ceil_div_signed(weight_banks, ctx->bank_granularity);
        int k_groups = ceil_div_signed(k_tile, tile_count);
        int base = ceil_div_signed(ctx->banks_for_normal + cacc_banks - 1,
                                   ctx->banks_for_normal);

        if (k_groups != 1 && base > 1) {
            int rem = cacc_banks % ctx->banks_for_normal;
            if (rem != 0 && ctx->banks_for_normal <= rem * k_groups)
                base++;
        }

        return base;
    }

    int total = ctx->bank_granularity * ctx->banks_for_normal;
    return ceil_div_signed(weight_banks + total - 1, total);
}
```

## fcn.00387530: check/select base tiling shape

This helper writes two integers to the output pointer passed in `x4`.

Pseudo-code:

```c
struct TileShape {
    int count;
    int step;
};

int check_base_tiling(
    ConvTileCtx *ctx,
    bool mode,
    int fmt_or_dtype,
    int shape_vec[],
    TileShape *out,
    int atomic_step,
    bool use_shape_vec,
    bool alt)
{
    int atomic_c_hw = ctx->atomic_c_hw * 8;
    int step = atomic_c_hw / atomic_step;

    int vec_len = vector_len(shape_vec);
    if (vec_len == 4 && !use_shape_vec) {
        if (ctx->target == RKNPU_F2 || ctx->target == RKNPU_W1 ||
            ctx->target == RKNPU_F3 || ctx->target == TAG_TREH ||
            ctx->target == TAG_TREI) {
            out->count = ctx->atomic_c_total / atomic_step;
            out->step = step;
            return 0;
        }
    }

    if (vec_len == 2 || use_shape_vec) {
        int candidate = choose_shape_candidate(shape_vec, use_shape_vec,
                                               atomic_step, alt);
        int hw_limit = get_hw_limit(ctx, atomic_step, alt);
        if (candidate > hw_limit)
            candidate = hw_limit;

        if (fmt_or_dtype == 9)
            candidate = round_up_to_multiple(candidate, ctx->special_align);
        else if (!mode)
            candidate = round_up_to_multiple(candidate,
                                             get_channel_align(ctx, atomic_step, alt));

        out->count = ctx->atomic_c_total / atomic_step;
        out->step = candidate;
        return 0;
    }

    out->count = 0;
    out->step = step;
    return 0;
}
```

## fcn.00383338: estimate minimum data tile

This helper is called immediately before `fcn.00384ed0` in the main function.
Its return value is used to convert spare CBUF bank capacity into a spatial/data
tile budget:

```text
0x00388118 call fcn.00383338
0x00388168 spare_banks * bank_granularity / returned_min_data_tile
```

Call shape from `fcn.00387e3c`:

```c
min_data_tile = fcn_00383338(
    ctx,
    (out_w - 1) * stride_like + 1,
    aligned_tile_count,
    c_step);
```

Pseudo-code:

```c
int estimate_min_data_tile(ConvTileCtx *ctx,
                           int spatial_span,
                           int aligned_tile_count,
                           int atomic_step)
{
    int atomic_groups = div_round_up_signed(ctx->atomic_c_total, 8);

    if (is_small_bank_target(ctx->target)) {
        int hw_atoms = ctx->atomic_c_hw;
        int hw_bytes = hw_atoms * 8;
        int channels_per_atomic_step = hw_bytes / atomic_step;
        int groups_per_hw_atom = hw_atoms / atomic_groups;

        int channel_group = aligned_tile_count / channels_per_atomic_step;
        int residue = aligned_tile_count % channels_per_atomic_step;
        int min_data = channel_group * spatial_span;

        if (groups_per_hw_atom == 4) {
            if (residue == channels_per_atomic_step)
                return min_data + spatial_span;
            if (residue == channels_per_atomic_step * 2)
                return min_data + ceil(spatial_span * 0.5);
            if (residue == channels_per_atomic_step * 3)
                return min_data + ceil(spatial_span * 0.25);
            return min_data;
        }

        if (groups_per_hw_atom == 2) {
            if (residue == channels_per_atomic_step)
                return min_data + ceil(spatial_span * 0.5);
            return min_data;
        }

        // Unknown ratio path emits a diagnostic and returns the partial result.
        log_bad_split(groups_per_hw_atom);
        return min_data;
    }

    if (ctx->target == 0 && aligned_tile_count <= 4)
        return specialized_small_target_estimate(ctx, spatial_span,
                                                 aligned_tile_count, atomic_step);

    // Normal fallback visible at 0x00383394..0x003833b4.
    int hw_bytes = ctx->atomic_c_hw * 8;
    int channel_scale = ctx->atomic_c_hw / atomic_groups;
    int channels_per_step = hw_bytes / atomic_step;
    int groups = aligned_tile_count / channels_per_step;

    return groups * spatial_span;
}
```

## fcn.003873e8: clamp/tune channel tile size

This helper uses `fcn.000f4758` and `fcn.00389078` to find a legal tile count
for a channel/atomic step. It halves the candidate until it fits the computed
weight/data pressure.

Pseudo-code:

```c
int tune_channel_tile(
    ConvTileCtx *ctx,
    bool use_target_limit,
    int requested,
    int atomic_step,
    int fmt_or_dtype,
    int special_fmt)
{
    int base = ctx->atomic_c_total / atomic_step;
    int rounded = round_up_to_multiple(requested, base);

    if (special_fmt == 9) {
        int normalized_step = normalize_fmt_to_atomic(special_fmt);
        return pick_limit_a_by_atomic(ctx, normalized_step, fmt_or_dtype);
    }

    if (use_target_limit) {
        if (is_f2_like_bank_target(ctx->target)) {
            int limit = pick_limit2_by_atomic(ctx, atomic_step, fmt_or_dtype);
            if (rounded < limit)
                return rounded;
        }

        return pick_limit2_by_atomic(ctx, atomic_step, fmt_or_dtype);
    }

    int limit = pick_limit_a_by_atomic(ctx, atomic_step, fmt_or_dtype);
    int half;
    do {
        half = limit / 2;
        if (rounded > half)
            return limit;

        int atom_group = ctx->atomic_c_hw / div_round_up_signed(atomic_step, 8);
        int quarterish = div_round_up_signed(atom_group, 4);
        if (limit <= quarterish)
            return rounded;

        limit = half;
    } while (true);
}
```

## fcn.00385988: compute legal K/weight split step

This helper is called from the layer-level planner paths before the
`MC K Tile Failed` diagnostic:

```text
0x005d7178 call 0x00385988
0x005f4bf4 call 0x00385988
```

It combines the selected bank budget, target bank granularity, and requested K
work into a legal K step. The helper returns `-1` when the required split is
below the minimum representable step.

Inputs inferred from instruction use:

```text
x0 ctx
w1 available_or_requested_k_units
w2 divisor/spatial_or_group_scale
w3 forced_units; if non-zero, atomic_step = forced_units * 8, else 4
```

Pseudo-code:

```c
int compute_legal_k_split_step(ConvTileCtx *ctx,
                               int requested_units,
                               int divisor,
                               int forced_units)
{
    int atomic_step = forced_units ? forced_units * 8 : 4;

    // Convert available bank units into the same scale used by the B-limit
    // table. ctx->bank_granularity is +0x64; the following word is the
    // hardware atomic value used by other estimators.
    int scaled = ctx->bank_granularity * ctx->atomic_c_hw;
    scaled = (scaled * requested_units) / divisor;

    int limit = pick_limit_b_by_atomic(ctx, atomic_step, 0);

    if (scaled >= limit) {
        int groups = scaled / limit;
        return groups * limit;
    }

    if (limit < scaled * 2) {
        // The request fits only as a half-limit chunk. This is the branch that
        // lets the planner repair K split sizes without dropping below the
        // hardware minimum.
        return (scaled / (limit / 2)) * (limit / 2);
    }

    return -1;
}
```

In the large tilers this function is the bridge between a high-level K/output
channel split request and the target's atomic-width limit table. A negative
return is treated as an illegal or failed K tile and eventually reaches the MC K
failure logging path.

## Small selector helpers

### fcn.00384ba0

Selects an "A" limit by atomic width:

```c
int pick_limit_a_by_atomic(ConvTileCtx *ctx, int atomic_step, int fmt)
{
    if (fmt == 9 || atomic_step == 16)
        return ctx->limit_16_a;   // +0x3c
    if (atomic_step == 4)
        return ctx->limit_4_a;    // +0x2c
    if (atomic_step == 8)
        return ctx->limit_8_a;    // +0x34
    if (atomic_step == 32)
        return ctx->limit_32_a;   // +0x44
    abort_bad_tiling(atomic_step);
}
```

### fcn.00384d48

Selects a "B" limit by atomic width:

```c
int pick_limit_b_by_atomic(ConvTileCtx *ctx, int atomic_step, int fmt)
{
    if (fmt == 9 || atomic_step == 16)
        return ctx->limit_16_b;   // +0x40
    if (atomic_step == 4)
        return ctx->limit_4_b;    // +0x30
    if (atomic_step == 8)
        return ctx->limit_8_b;    // +0x38
    if (atomic_step == 32)
        return ctx->limit_32_b;   // +0x48
    if (atomic_step == 64)
        return -1;
    abort_bad_tiling(atomic_step);
}
```

### fcn.00389078

Selects a second limit table used by `fcn.003873e8`:

```c
int pick_limit2_by_atomic(ConvTileCtx *ctx, int atomic_step, int fmt)
{
    if (fmt == 9 || atomic_step == 16)
        return ctx->limit2_16_or_fmt9; // +0x54
    if (atomic_step == 4)
        return ctx->limit2_4;          // +0x4c
    if (atomic_step == 8)
        return ctx->limit2_8;          // +0x50
    if (atomic_step == 32)
        return ctx->limit2_32;         // +0x58
    abort_bad_tiling(atomic_step);
}
```

### fcn.000e1100 and fcn.000f4758

These are duplicated selector-style helpers from another module. They choose
similar limit values by `atomic_step` and format, with special cases for
format values `2..7` and `9`.

Important behavior:

```c
int normalize_atomic_for_format_e1100(ConvTileCtx *ctx,
                                      int atomic_step,
                                      int fmt,
                                      int fmt2)
{
    if (fmt2 == 0) {
        if (atomic_step == 4)  return ctx->limit_4_b;
        if (atomic_step == 8)  return ctx->limit_8_b;
        if (atomic_step == 16) return ctx->limit_16_b;
        if (atomic_step == 32) return ctx->limit_32_b;
        if (atomic_step == 64) return -1;
    }

    if (fmt2 == 2 || fmt2 == 4)
        return 32;
    if (fmt2 == 3 || (fmt2 > 4 && fmt2 <= 7))
        return 16;

    // Unsupported format emits a diagnostic and retries with fmt2=0.
    return normalize_atomic_for_format_e1100(ctx, atomic_step, fmt, 0);
}
```

`fcn.000f4758` is the same family but reads the `*_a` limit table for the direct
atomic-width path.

## fcn.005d4ac0: layer-level conv split planner

This is the large function reached by the string:

```text
0x0062ce00 Conv min_weight_banks > 3, OutputName : %s
```

and by:

```text
0x0062ce50 MC K Tile Failed, min kernel step %d, but get %d
```

The function is much larger than the local ChannelTile predicate
(`~11 KiB`, high cyclomatic complexity). The useful decompiled behavior is that
it builds a tile vector for one conv layer, using the same lower-level bank/data
helpers seen in the direct xref functions:

```text
0x005d55a4 call 0x00383338  ; data tile estimate helper
0x005d72dc call 0x00383338  ; repeated data tile feasibility check
0x005d7178 call 0x00385988  ; bank/fit helper used before MC K fallback
```

Pseudo-code of the relevant tiling path:

```c
vector<ConvWorkTile> build_conv_work_tiles(
    ConvTileCtx *ctx,
    ConvLayer *layer,
    ShapeVec input_shape,
    ShapeVec weight_shape,
    ShapeVec output_shape,
    ConvAttrs attrs,
    TileOptions opts)
{
    // The prologue copies 4-int shape vectors into stack locals. Inferred names:
    int in_h = input_shape[2];
    int in_w = input_shape[3];
    int in_c = input_shape[1];
    int out_h = output_shape[2];
    int out_w = output_shape[3];
    int out_c = output_shape[1];
    int kernel_h = weight_shape[2];
    int kernel_w = weight_shape[3];
    int kernel_c = weight_shape[1];

    TileSeed seed = build_initial_tile_seed(ctx, input_shape, weight_shape,
                                            output_shape, attrs, opts);
    vector<ConvWorkTile> tiles;

    int split_method = choose_split_method(seed);
    if (split_method_is_illegal(split_method))
        fatal_illegal_tiling_method(layer, split_method);

    int min_weight_banks = seed.min_weight_banks;
    if (min_weight_banks > 3 && seed.emit_conv_threshold_log)
        log("Conv min_weight_banks > 3, OutputName : %s", layer->output_name);

    // Outer loop walks Y/H tiles. The function repeatedly computes how many
    // output rows fit after reserving banks for weights.
    for (int y_start = 0; y_start < out_h; ) {
        int y_step = choose_y_step_from_data_banks(ctx, seed, y_start);
        int input_bank_need = estimate_min_data_tile(ctx,
            input_span_for_y_tile(y_start, y_step, attrs),
            seed.aligned_channel_count,
            seed.atomic_step);

        // Inner loop walks K/output-channel/kernel-group tiles. The MC K branch
        // tries to keep the K step at or above the minimum legal hardware step.
        for (int k_start = 0; k_start < out_c; ) {
            int k_step = choose_k_step_from_weight_banks(ctx, seed, k_start);
            int min_k_step = minimum_legal_k_step(ctx, seed);

            if (k_step < min_k_step) {
                log("MC K Tile Failed, min kernel step %d, but get %d",
                    min_k_step, k_step);
                k_step = repair_or_fallback_k_step(ctx, seed, min_k_step);
            }

            ConvWorkTile t = {};
            t.xstart = 0;
            t.ystart = y_start;
            t.kstart = k_start;
            t.y_step = y_step;
            t.k_step = k_step;
            t.input_bank_num = input_bank_need;
            t.weight_bank_num = min_weight_banks;
            t.data_reuse = compute_initial_data_reuse(seed, y_start, k_start);
            t.weight_reuse = compute_initial_weight_reuse(seed, y_start, k_start);
            t.mc_treat_by_y_tile = seed.mc_y_tile;
            t.mc_treat_by_k_tile = seed.mc_k_tile;
            t.mc_treat_by_1c_y_tile = seed.mc_1c_y_tile;
            t.mc_treat_by_1c_k_tile = seed.mc_1c_k_tile;
            tiles.push_back(t);

            k_start += k_step;
        }

        y_start += y_step;
    }

    if (!update_data_weight_reuse(ctx, tiles))
        log_or_abort("failed to update data and weight reuse!");

    return tiles;
}
```

This function is the strongest evidence that the official runtime's ChannelTile
path is not just naive OC slicing. It builds explicit tile records with Y/K
starts, bank counts, reuse flags, and multi-core treatment fields. That matches
the experiment result: simple per-submit OC slicing works for the small proxy,
but the real `160->320 3x3` overflow likely needs the extra reuse/group
programming from this layer-level planner.

## fcn.005f38f0: K/X split vector composer

This function is reached from `fcn.005f51c0`:

```text
0x005f5d54 call 0x005f38f0
```

It is also the xref owner for the fatal `illegal tiling method` diagnostic at
`0x0062d040`. The function consumes candidate shape vectors, target limits, and
multi-core mode flags, then emits split vectors used by the outer tiler. It is
not a register emitter; it is a planner-side vector builder.

Important calls and checks:

```text
0x005f3c08 call 0x00383338  ; data-tile pressure estimate
0x005f4bf4 call 0x00385988  ; K split feasibility/repair helper
0x005f4e0c log+abort        ; illegal tiling method
0x005f5018 log path         ; extended layer-shape diagnostic
```

Pseudo-code:

```c
bool compose_kx_split_vectors(
    ConvTileCtx *ctx,
    ShapeVec *input_shape,
    ShapeVec *kernel_shape,
    vector<vector<int>> *out_split_vectors,
    vector<vector<bool>> *out_reuse_masks,
    int mc_y_type,
    int mc_k_type,
    int split_method,
    TileLimits limits,
    TileAttrs attrs)
{
    Shape4 in = normalize_shape4(input_shape);
    Shape4 kernel = normalize_shape4(kernel_shape);

    int y_span = input_span_for_candidate(in, attrs);
    int data_need = estimate_min_data_tile(ctx, y_span,
                                           attrs.aligned_channels,
                                           attrs.atomic_step);

    int bank_room_for_k = attrs.total_banks - data_need;
    int k_step = compute_legal_k_split_step(ctx,
                                            bank_room_for_k,
                                            attrs.k_divisor,
                                            attrs.forced_units);

    if (k_step <= 0)
        return false;

    switch (split_method) {
    case SPLIT_NONE:
        append_single_tile(out_split_vectors, in, kernel);
        break;
    case SPLIT_BY_Y:
        append_y_tiles(out_split_vectors, in, kernel, limits.y_step);
        break;
    case SPLIT_BY_K:
        append_k_tiles(out_split_vectors, in, kernel, k_step);
        break;
    case SPLIT_BY_Y_AND_K:
        append_yk_tiles(out_split_vectors, in, kernel, limits.y_step, k_step);
        break;
    default:
        log("Failed to config layer: '%s', illegal tiling method %d", ...);
        abort();
    }

    build_reuse_masks_for_adjacent_tiles(out_reuse_masks, *out_split_vectors,
                                         mc_y_type, mc_k_type);
    return true;
}
```

This is one of the missing "not just ChannelTile" functions. It handles the
generic Y/K/X split vector construction used by the top-level conv tiler. The
specific split-method enum values are inferred from branch shape rather than
symbol names, but the behavior is clear: choose a split method, build vectors,
and reject impossible methods before register-task configuration.

### Focused split-vector pass

The `fcn.005f51c0` caller reaches this function at `0x005f5d54`. Immediately
before the call it prepares stack-backed vector outputs at `x29+0x4a8`,
`x29+0x4c0`, and `x29+0x4d8`, plus two byte result flags at `x29+0x31e` and
`x29+0x31f`. This supports treating `fcn.005f38f0` as a tile-table composer,
not a direct register emitter.

Inside `fcn.005f38f0`, the repeated builder blocks around `0x005f4154` and
`0x005f4d5c` use three important stack slots:

| Stack slot | Current interpretation | Evidence |
| --- | --- | --- |
| `[x29+0x1188]` | Tiling-method value for this composer pass. | Loaded before comparisons with `3` and `2`; later loaded into the fatal `illegal tiling method %d` formatter at `0x005f5094`. |
| `[x29+0x1178]` | Pointer/cache for one nested `vector<vector<int>>` output, holding per-tile start/modulo indexes. | Resized/indexed as a nested vector, then written with `tile_index % method`, `tile_index % 3`, or `tile_index & 1`. |
| `[x29+0x1180]` | Pointer/cache for the paired nested `vector<vector<int>>` output, holding per-tile marker/type flags. | Resized/indexed in parallel with `0x1178`, then written with marker values `1`, `3`, or `7`. |

Observed method behavior:

- generic method value: store `tile_index % method` into the first vector and
  marker `1` into the second vector, with marker `7` on a remainder boundary;
- method `3`: store `tile_index % 3` into the first vector and marker `7` on
  the boundary case, otherwise marker `1`;
- method `2`: store `tile_index & 1` into the first vector and marker `3` or
  `1` into the second vector.

The same pattern appears twice, once looping over a `w20` tile count and once
over a `w19` tile count. These are likely two dimensions/split lists, not two
separate Python strategy families.

Reverse status: `2` and `3` are now confirmed tiling-method cases in the
composer, and `0x1178`/`0x1180` are narrowed to paired per-tile index/type
vectors. They are still not safe to name as `Y`, `K`, `1C`, `MC`, or
`ChannelTile` until the field that chooses `[x29+0x1188]` is named and the
later reuse/register updater is tied to concrete CNA/DPU/PC entries.

## fcn.005f51c0: top-level conv Y/K/multicore tiler

This is the largest conv tiling function found from the string/xref pass. It
owns these diagnostics:

```text
0x0062d298 X tile buffer overflow! Failed to config layer: '%s', Fatal Error input W too large
0x0062d508 Generate Y config crash! Failed to config layer: '%s'
0x0062d7b8 Illegal mc Y type, Failed to config layer: '%s'
0x0062d8e8 Illegal mc K type, Failed to config layer: '%s'
```

It also calls the lower-level split composer:

```text
0x005f5d54 call 0x005f38f0
```

and many of the same conv-bank helpers used by the ChannelTile predicate and
layer-level planner:

```text
0x005f5818 call 0x00388438
0x005f5848 call 0x003863d0
0x005f5884 call 0x00385558
0x005f5bbc call 0x00383338
0x005f5e50 call 0x00385a68
0x005f8f40 call 0x00385a68
```

Pseudo-code of the useful planner behavior:

```c
bool build_multicore_conv_tiles(
    ConvTileCtx *ctx,
    ConvLayer *layer,
    ShapeVec input_shape,
    ShapeVec kernel_shape,
    ShapeVec output_shape,
    ConvAttrs attrs,
    MultiCoreOptions mc,
    vector<vector<ConvWorkTile>> *per_core_tiles)
{
    Shape4 in = normalize_shape4(input_shape);
    Shape4 kernel = normalize_shape4(kernel_shape);
    Shape4 out = normalize_shape4(output_shape);

    TileLimits limits = derive_target_limits(ctx, in, kernel, out, attrs);
    if (input_width_tile_overflows(in, kernel, attrs, limits)) {
        log("X tile buffer overflow! Failed to config layer: '%s', Fatal Error input W too large", ...);
        return false;
    }

    if (!valid_mc_y_type(mc.y_type)) {
        log("Illegal mc Y type, Failed to config layer: '%s'", ...);
        return false;
    }

    if (!valid_mc_k_type(mc.k_type)) {
        log("Illegal mc K type, Failed to config layer: '%s'", ...);
        return false;
    }

    vector<vector<int>> split_vectors;
    vector<vector<bool>> reuse_masks;
    bool ok = compose_kx_split_vectors(ctx, &input_shape, &kernel_shape,
                                       &split_vectors, &reuse_masks,
                                       mc.y_type, mc.k_type,
                                       limits.split_method, limits, attrs);
    if (!ok) {
        log("Generate Y config crash! Failed to config layer: '%s'", ...);
        return false;
    }

    for (int core = 0; core < mc.core_num; core++) {
        for (SplitRecord r : split_vectors_for_core(split_vectors, core, mc)) {
            ConvWorkTile t = {};
            t.xstart = r.xstart;
            t.ystart = r.ystart;
            t.kstart = r.kstart;
            t.y_step = r.y_step;
            t.k_step = r.k_step;
            t.input_bank_num = r.input_bank_num;
            t.weight_bank_num = r.weight_bank_num;
            t.data_reuse = r.data_reuse;
            t.weight_reuse = r.weight_reuse;
            t.mc_treat_by_y_tile = mc.y_type;
            t.mc_treat_by_k_tile = mc.k_type;
            t.mc_treat_by_1c_y_tile = mc.one_core_y_type;
            t.mc_treat_by_1c_k_tile = mc.one_core_k_type;
            (*per_core_tiles)[core].push_back(t);
        }
    }

    return true;
}
```

This function is the other major "all tiling" piece. It is responsible for the
outer Y/X/multi-core legality checks and for distributing split records into
per-core nested vectors. The previous ChannelTile-only decompilation explained
when RKNN switches to ChannelTile; `fcn.005f51c0` explains the broader conv
tiling machinery that can fail on X width, Y config generation, or illegal MC
Y/K modes even when ChannelTile thresholding is not involved.

## Reuse and tile-table helpers

### fcn.00101be8: CNA group formatter

This function is the xref target for the CNA group diagnostic strings. It is a
formatter, not the planner, but it gives a concrete resource-mask layout. A
focused disassembly pass shows the function takes the mask in `w0`, prints the
raw mask first, then prints individual bits by repeatedly extracting one bit
with `and`/`ubfx` and appending the matching diagnostic string.

| Mask bit | Printed field |
| ---: | --- |
| `0` | `CNA feature group0` |
| `1` | `CNA feature group1` |
| `2` | `CNA weight  group0` |
| `3` | `CNA weight  group1` |
| `4` | `CNA csc     group0` |
| `5` | `CNA csc     group1` |
| `6` | `ACCU        group0` |
| `7` | `ACCU        group1` |
| `8` | `DPU         group0` |
| `9` | `DPU         group1` |
| `10` | `PPU         group0` |
| `11` | `PPU         group1` |
| `12` | `DMA read     error` |
| `13` | `DMA write    error` |

The bit names above come from the contiguous string block at
`0x00607178..0x00607318`, confirmed with `rabin2 -zz` and a rodata dump.

This is useful for the ChannelTile reverse trail because the local Python
OC-chain only changes weight/output addresses and optionally
`CNA_CBUF_CON0_WEIGHT_REUSE`, while RKNN has a separate feature/weight/CSC group
mask visible in diagnostics. Static xref checks for the `RKNNRegisterTask_f2`
RTTI strings (`0x00606f30`, `0x00609230`) only reached data/RTTI references, and
direct xrefs to the CNA group strings still lead to this formatter. A focused
caller search for `fcn.00101be8` itself (`r2 axt @ 0x00101be8` and
`objdump -d | rg 'bl\s+101be8|101be8'`) found no direct call sites beyond the
function body, so the formatter is likely reached indirectly or through stripped
logging/type plumbing. The programming site for this mask is still not
identified.

### fcn.00307198 and fcn.00312328

These two functions are the xrefs for:

```text
0x0060c768 failed to update data and weight reuse!
```

They are not reached by the direct `min_weight_banks > 3` string, but they are
part of the same compiled tiling pipeline. Their role is to walk the tile list
after split planning and mark which adjacent tiles may reuse data and/or
weights.

A focused follow-up pass shows these are not simple duplicates:

- `fcn.00307198` is a small single-index/list updater. It walks lists rooted at
  `ctx+0x448`, `ctx+0x460`, and `ctx->field_0x40 + {0x120,0x128,0x1e0,0x1e8,0x200}`.
- `fcn.00312328` is a counted/vectorized updater. It loops over a count and
  key/active vectors, then reaches the same list families plus additional node
  payloads at `+0x28` and `+0x38`. It also has extra RegisterTask mode and
  weight-data-type handling.

The `fcn.00312328` control shape is now narrower:

- If `ctx+0x460 == 0`, it starts from `ctx->field_0x40`; the first active path
  walks `field_0x40 + 0x120/0x128`, and alternate paths also reach
  `field_0x40 + 0x1e0/0x1e8/0x200`.
- If `ctx+0x460 != 0`, it switches to the direct context list family
  `ctx+0x440/0x448`.
- In both cases it iterates `count` entries, skips inactive entries from the
  active vector, searches a keyed tree/list by node key at `+0x20`, and then
  uses node payloads at `+0x28`/`+0x38`.
- The payload format is compact and self-describing: several call sites check a
  small record-size/header value before reading offsets such as `+4`, `+6`,
  `+8`, `+0xa`, `+0x18`, `+0x1c`, `+0x2c`, or `+0x40`. Those values become the
  register-task index or side value passed to `fcn.002efd38`,
  `fcn.002efce0`, or a RegisterTask virtual setter.
- Failure paths at `0x00312cb4`, `0x00312e0c`, `0x00312f74`,
  `0x0031326c`, `0x003132c8`, and `0x003132f4` still all emit the same
  `failed to update data and weight reuse!` diagnostic, so these branches are
  part of one broader materialization pass rather than unrelated errors.
- The nearby rodata confirms two relevant diagnostics in this region:
  `0x0060c768` is `failed to update data and weight reuse!`, while
  `0x0060cfe0` is `Unsupport weight data type!`. The latter belongs to the
  RegisterTask/weight-type guard side of `fcn.00312328`; it is not evidence that
  all reuse failures are weight-format failures.
- The vtable targets compared by these branches, including `0x103f10`,
  `0x103f18`, `0x1076a8`, `0x1076b0`, `0x107948`, `0x105d50`, `0x107930`,
  `0x107588`, `0x105828`, and `0x104180`, disassemble as tiny `mov w0, #0; ret`
  stubs or adjacent runs of those stubs. They identify method slots/default
  implementations, not the concrete register-writing code. The missing mapping
  is still from typed tile-list payloads to the indexed register-command vector.

The low-level write primitive is now known. Both update paths eventually patch
the value field of existing 64-bit register commands:

```c
uint64_t patch_reg_value(uint64_t old_entry, uint32_t value)
{
    // fcn.00100a40: bfi x0, x1, #16, #32
    return (old_entry & 0xffff00000000ffffULL) | ((uint64_t)value << 16);
}
```

This matches the local Python `E(target, reg, value)` encoding: target in bits
`48..63`, register value in bits `16..47`, and register address in bits `0..15`.

The immediate helper forms are:

```c
int update_indexed_reg_value(ctx, int index, uint32_t value)
{
    // fcn.002efd38
    vec = ctx->field_0x288;
    entry_ptr = vec->base_0x08 + vec->index_base_0x28 + index * 8;
    if (!entry_ptr)
        return -1;
    *entry_ptr = patch_reg_value(*entry_ptr, value);
    ctx->dirty_0x11 = 1;
    return 0;
}

int update_pair_record(ctx, PairRecord *record, uint32_t value)
{
    // fcn.002efce0
    if (record->side_int_ptr)
        record->side_int_ptr[1] = value;
    if (!record->entry_ptr)
        return -1;
    *record->entry_ptr = patch_reg_value(*record->entry_ptr, value);
    ctx->dirty_0x11 = 1;
    return 0;
}
```

Focused helper recheck:

- `fcn.002efd38` computes a command-vector slot from fields under `ctx+0x288`
  and the caller-provided `index * 8`. It returns `-1` if the slot is the
  invalid sentinel; otherwise it patches the encoded 64-bit entry and sets
  `ctx+0x11 = 1`.
- `fcn.002efce0` takes a two-pointer payload record. It optionally writes the
  value to the first pointer's side integer at `+4`, requires the second pointer
  to be non-null, then patches the encoded entry at the second pointer and sets
  `ctx+0x11 = 1`.
- `fcn.00100a40` is exactly `bfi x0, x1, #16, #32; ret`, so both helper paths
  mutate only the middle 32-bit value field of an already encoded command.

The important consequence is that RKNN reuse update is not equivalent to one
visible `CNA_CBUF_CON0_WEIGHT_REUSE` bit. It patches selected pre-existing
register commands through tile-list metadata. The remaining reverse task is to
name which indexed register entries these vectors point to for each split
family.

Next route for naming those entries:

- Hook or breakpoint `fcn.002efd38` and `fcn.002efce0` while running an RKNN
  model that triggers the F2 ChannelTile path.
- For each helper call, log the caller PC, helper value, command-vector index
  or pair-record pointer, old encoded command, and new encoded command.
- Decode old/new commands as the local Python driver does:
  `target = entry >> 48`, `reg = entry & 0xffff`,
  `value = (entry >> 16) & 0xffffffff`.
- Compare the patched `(target, reg)` pairs against the Python register names
  for CNA/DPU/PC. Static analysis has identified the patch mechanics, but a
  runtime trace is the direct path to naming the semantic register destinations
  for ChannelTile reuse/group state.

Minimal GDB-style trace recipe:

- Find the loaded `librknnrt.so` base in `/proc/<pid>/maps`; the helper offsets
  are `0x002efd38`, `0x002efce0`, and `0x00100a40`.
- Break at `base+0x002efd38`; log `x0`, `w1`, `w2`, `lr`, and the old/new
  command at the computed vector slot.
- Break at `base+0x002efce0`; log `x0`, `x1`, `w2`, `lr`, `pair_record[0]`,
  `pair_record[1]`, and old/new `*pair_record[1]`.
- Decode commands as `target = entry >> 48`, `reg = entry & 0xffff`,
  `value = (entry >> 16) & 0xffffffff`. The `(target, reg)` pair is the missing
  semantic destination.
- A read-only local template for this trace now exists at
  `experimental/trace_librknnrt_reuse.gdb`. It follows the same embedded-GDB-
  Python style as `experimental/capture_rknpu_submit.gdb` and logs the two reuse
  helper entry states. This host is not expected to run official RKNN or the
  vendor rknpu stack; treat this as a later user-run experiment on a suitable
  environment with an RKNN conv model that actually triggers the F2 ChannelTile
  path.
- Static TRM comparison candidates for that trace are now documented in
  `conv_tile_result_and_cleanup_plan.md`: CNA/CORE/DPU/RDMA `S_POINTER`,
  `CNA_CONV_CON2`, `CNA_CBUF_CON0`, `CNA_FEATURE_DATA_ADDR`,
  `CNA_DCOMP_ADDR0`, `CNA_WEIGHT_SIZE*`, and `CNA_DCOMP_AMOUNT*`. These are not
  proven RKNN destinations; they are the local register names to compare against
  decoded `(target, reg)` trace output.

Pseudo-code:

```c
bool update_data_weight_reuse(ConvTileCtx *ctx, vector<ConvWorkTile> &tiles)
{
    if (tiles.empty())
        return true;

    for (int i = 0; i < tiles.size(); i++) {
        ConvWorkTile *prev = (i == 0) ? NULL : &tiles[i - 1];
        ConvWorkTile *cur = &tiles[i];

        cur->data_reuse = false;
        cur->weight_reuse = false;

        if (prev != NULL) {
            // Data can be reused when the feature/X/Y window is compatible and
            // the next tile only changes K/output-channel work.
            if (same_feature_window(*prev, *cur) &&
                compatible_mc_y_treatment(*prev, *cur))
                cur->data_reuse = true;

            // Weights can be reused when the K/kernel-group window is compatible
            // and the next tile only changes Y/spatial work.
            if (same_weight_window(*prev, *cur) &&
                compatible_mc_k_treatment(*prev, *cur))
                cur->weight_reuse = true;
        }

        if (!reuse_combination_is_legal(ctx, *cur)) {
            log("failed to update data and weight reuse!");
            return false;
        }
    }

    return true;
}
```

For `conv_new_clean.py`, this is the missing class of state in the failing
PC-chain ChannelTile experiment: per-tile arithmetic and final output offsets
can be correct, while chained tiles with changing weight offsets still fail if
the reuse flags/group state are not updated like RKNN does here.

### fcn.005c6b50

This workload config function is the xref for:

```text
0x0062cc50 banks num invalid, input_bank_num: %d, weight_bank_num: %d
0x0062cc90 Invalid reuse strategy!
```

Pseudo-code for the relevant checks:

```c
void configure_workload_banks_and_reuse(Workload *wl, ConvWorkTile *tile)
{
    int input_bank_num = tile->input_bank_num;
    int weight_bank_num = tile->weight_bank_num;

    if (input_bank_num <= 0 || weight_bank_num <= 0 ||
        input_bank_num + weight_bank_num > target_cbuf_banks(wl)) {
        log("banks num invalid, input_bank_num: %d, weight_bank_num: %d",
            input_bank_num, weight_bank_num);
        fail_workload_config();
    }

    if (!valid_reuse_strategy(tile->data_reuse, tile->weight_reuse,
                              tile->mc_treat_by_y_tile,
                              tile->mc_treat_by_k_tile,
                              tile->mc_treat_by_1c_y_tile,
                              tile->mc_treat_by_1c_k_tile)) {
        log("Invalid reuse strategy!");
        fail_workload_config();
    }

    program_cbuf_bank_fields(wl, input_bank_num, weight_bank_num);
    program_reuse_fields(wl, tile);
}
```

### fcn.005f1cd0 and fcn.005f1998

`fcn.005f1cd0` prints the tile-table header:

```text
|xstart  |ystart  |kstart  | data reuse | weight reuse | mc_treat_by_y_tile | mc_treat_by_k_tile | mc_treat_by_1c_y_tile | mc_treat_by_1c_k_tile |
```

It iterates nested tile vectors and formats the fields that matter for
ChannelTile debugging. Pseudo-code:

```c
void dump_workload_tile_table(vector<vector<ConvWorkTile>> &tiles,
                              bool print_data_reuse,
                              bool print_weight_reuse,
                              bool print_mc_y,
                              bool print_mc_k,
                              bool print_mc_1c_y,
                              bool print_mc_1c_k)
{
    print(tile_table_header);

    for (int core = 0; core < tiles.size(); core++) {
        for (int i = 0; i < tiles[core].size(); i++) {
            ConvWorkTile *t = &tiles[core][i];
            print_row(t->xstart, t->ystart, t->kstart,
                      t->data_reuse, t->weight_reuse,
                      t->mc_treat_by_y_tile,
                      t->mc_treat_by_k_tile,
                      t->mc_treat_by_1c_y_tile,
                      t->mc_treat_by_1c_k_tile);
        }
    }
}
```

Focused formatter mapping:

The function arguments are stored at entry as:

| Stack slot | Source argument | Current meaning |
| --- | --- | --- |
| `[x29+0xd8]` | `x0` | Outer tile loop / core or X dimension vector. |
| `[x29+0x100]` | `x1` | Nested `ystart` vector source. |
| `[x29+0x110]` | `x2` | Nested `kstart` vector source. |
| `[x29+0xc0]` | `x4` | First reuse bitset source. |
| `[x29+0xc8]` | `x3` | Second reuse bitset source. |
| `[x29+0xbc]` | `w5` | Print/control flag copied from arg 5. |
| `[x29+0xb8]` | `w6` | Print/control flag copied from arg 6. |
| `[x29+0xb4]` | `w7` | Print/control flag copied from arg 7. |
| `[x29+0xb0]` | stack byte arg | Print/control flag copied from the first stack byte arg. |

At the row-print block around `0x005f206c`:

- `xstart` is loaded from `[x29+0xd8]` with the outer row index;
- `ystart` is loaded from `[x29+0x100]` using the current outer and inner
  vector offsets;
- `kstart` is loaded from `[x29+0x110]` using the same outer and inner offsets;
- the first reuse bool is computed by testing a bit in the bitset pointed to by
  `[x29+0xc0]`;
- the second reuse bool is computed by testing a bit in the bitset pointed to by
  `[x29+0xc8]`;
- the four MC treatment values printed after the reuse booleans are scalar
  control values copied from `w5`, `w6`, `w7`, and the stack byte arg.

The header names these two bitsets as `data reuse` and `weight reuse`, but the
current disassembly pass only proves the formatter order and bitset mechanics.
It does not yet prove which producer argument is data versus weight at the call
sites.

Call-site boundary:

- `rabin2 -zz` finds the header only at `0x0062cf78`.
- `objdump -d ... | rg '5f1cd0|005f1cd0'` finds the function body at
  `0x005f1cd0`, but no direct `bl 0x005f1cd0` caller.
- `r2 axt 0x005f1cd0` likewise did not produce a useful static call-site list.

Therefore the producer of the two reuse bitsets is not identified by ordinary
static xrefs in this pass. The next route is to trace xrefs to the header string
or the surrounding logging stream construction, or to instrument a runtime path
that enables this tile-table dump.

Current boundary for `conv_new_clean.py` cleanup:

- The formatter proves RKNN tile records carry `xstart`, `ystart`, `kstart`,
  two reuse bitsets, and four MC treatment fields.
- The reuse updater proves RKNN can patch already-built register command values
  through indexed/list metadata, using the same 64-bit command encoding as the
  Python driver.
- The CNA group formatter proves feature, weight, and CSC groups are tracked as
  separate state.
- None of these sections yet identify the exact producer field that maps a
  Python OC/channel tile to RKNN's split method, reuse bitset, CNA group mask,
  or patched register index.

So this binary evidence supports keeping the Python branches as named planner
families during a mechanical refactor. It does not support replacing the current
software-stitched OC/channel/im2col fallbacks with a chained RKNN-style
ChannelTile executor.

`fcn.005f1998` is a small fatal helper:

```c
void abort_unknown_split_method(int split_method)
{
    log("Unknown split Method -> %d", split_method);
    abort();
}
```

## Helper arithmetic

The compiler emits signed division with manual rounding. The following helper
names match the instruction patterns:

```c
static int div_round_up_signed(int x, int d)
{
    int adjusted = x + d - 1;
    return adjusted / d;
}

static int ceil_div_signed(int x, int d)
{
    int adjusted = x + d - 1;
    return adjusted / d;
}

static int round_up_to_multiple(int x, int step)
{
    return ((x + step - 1) / step) * step;
}
```

## Final decompiled ChannelTile policy

The useful policy distilled from both xref functions:

```c
bool conv_needs_channel_tile(ConvTileCtx *ctx, ConvShape shape)
{
    Candidate c = build_channel_tile_candidate(ctx, shape);

    int min_weight_banks = estimate_min_weight_banks(
        ctx,
        c.mode,
        c.tile_a,
        c.tile_b,
        c.aligned_channels,
        c.k_tile,
        c.tile_count,
        c.alt,
        c.force_2,
        c.force_small,
        c.fmt_or_dtype);

    switch (ctx->target) {
    case RKNPU_F2:
        if (min_weight_banks > 3) {
            log("min_weight_banks > 3 && m_Target == RKNNTarget::RKNPU_F2, do conv with ChannelTile");
            return true;
        }
        break;
    case RKNPU_F3:
        if (min_weight_banks > 7) {
            log("min_weight_banks > 7 && m_Target == RKNNTarget::RKNPU_F3, do conv with ChannelTile");
            return true;
        }
        break;
    case RKNPU_W2:
        if (min_weight_banks > 8) {
            log("min_weight_banks > 8 && m_Target == RKNNTarget::RKNPU_W2, do conv with ChannelTile");
            return true;
        }
        break;
    case RKNPU_W1:
        if (min_weight_banks > 7) {
            log("min_weight_banks > 7 && m_Target == RKNNTarget::RKNPU_W1, do conv with ChannelTile");
            return true;
        }
        break;
    }

    return fallback_data_tile_pressure_says_channel_tile(ctx, c);
}
```

For RK3588/F2, this means the official RKNN runtime switches conv to
ChannelTile as soon as the estimated minimum weight CBUF demand is more than
three banks.

## Completion audit

Objective restated as deliverables:

1. Start from `experimental/rknn/librknnrt_conv_channel_tile_decomp.md`.
2. Do not stop at the two ChannelTile predicate functions.
3. Find the conv tiling functions in `librknnrt.so` reachable from tiling,
   split, bank, reuse, X/Y/K, and multicore diagnostics.
4. Add decompiled or high-level pseudo-code for every function that performs
   conv tiling work, while excluding generic logging/allocation/vector plumbing.

Prompt-to-artifact checklist:

| Requirement | Evidence in this file |
| --- | --- |
| Start from `librknnrt_conv_channel_tile_decomp.md` | This file remains the single expanded artifact. |
| Include ChannelTile threshold functions | `fcn.00387e3c` and `fcn.00388a18` sections contain full predicate pseudo-code. |
| Include bank/data estimators used by those predicates | `fcn.00384ed0`, `fcn.00387530`, `fcn.00383338`, `fcn.003873e8`, and selector-helper sections are documented. |
| Include non-ChannelTile conv tiling functions | `fcn.005d4ac0`, `fcn.005f38f0`, and `fcn.005f51c0` now have planner pseudo-code and call/xref evidence. |
| Include reuse/bank validation around the tile records | `fcn.00307198`, `fcn.00312328`, `fcn.005c6b50`, and `fcn.005f1cd0` sections document reuse updates, bank checks, and tile-table output. |
| Identify remaining ChannelTile reuse/register gap | Reuse helper sections document patch mechanics and the runtime-trace route for `fcn.002efd38` / `fcn.002efce0`; `experimental/trace_librknnrt_reuse.gdb` is the local read-only template for that trace. The concrete typed-payload to `(target, reg)` destination mapping is still not known until a true F2 ChannelTile workload is traced. |
| Cover all useful tiling strings found from the binary | The "Related tiling-string xrefs" table maps `min_weight_banks`, `MC K Tile Failed`, `failed to tile argb`, `illegal tiling method`, `X tile buffer overflow`, `Generate Y config crash`, `Illegal mc Y/K type`, reuse, bank, and tile-table strings to functions. |
| Exclude non-tiling plumbing | The function inventory explicitly excludes stream/logging, allocation, memory copy, and STL vector plumbing. |

Verification commands used:

```sh
rabin2 -zz experimental/rknn/librknnrt.so | rg -i 'conv|min_weight|tile|tiling|split|reuse|cbuf|bank|xstart|ystart|kstart|feature|weight|csc|overflow'
r2 -q -e scr.color=false -e bin.relocs.apply=true -c 'aaa' \
  -c 'axt @ 0x0060f388' -c 'axt @ 0x0062ce00' -c 'axt @ 0x0062ce50' \
  -c 'axt @ 0x0062ce30' -c 'axt @ 0x0062cf58' -c 'axt @ 0x0062cf78' \
  -c 'axt @ 0x0062d040' -c 'axt @ 0x0062d298' -c 'axt @ 0x0062d508' \
  -c 'axt @ 0x0062d7b8' -c 'axt @ 0x0062d8e8' \
  -c 'axt @ 0x0060c768' -c 'axt @ 0x0062cc50' -c 'axt @ 0x0062cc90' \
  -c q experimental/rknn/librknnrt.so
r2 -q -e scr.color=false -e bin.relocs.apply=true -c 'aa' -c 's 0x00385988' -c af -c pdf -c q experimental/rknn/librknnrt.so
r2 -q -e scr.color=false -e bin.relocs.apply=true -c 'aa' -c 's 0x005f38f0' -c af -c pdf -c q experimental/rknn/librknnrt.so
r2 -q -e scr.color=false -e bin.relocs.apply=true -c 'aa' -c 's 0x005f51c0' -c af -c pdf -c q experimental/rknn/librknnrt.so
```

Residual uncertainty: this is a stripped binary, so helper names and struct field
names are inferred from instruction use, call sites, and diagnostic strings. The
coverage is complete for the conv tiling functions discovered by the tiling
string/xref pass. The remaining callees in the planner callgraphs are generic
math/vector/allocation helpers with no tiling strings of their own, so they are
treated as plumbing rather than separate conv tiling functions.

Residual blocker for `conv_new_clean.py` strategy collapse: the binary notes now
show how reuse updates patch encoded command values, but not which concrete
CNA/DPU/PC register entries are patched for a true F2 ChannelTile workload. That
requires the runtime helper trace described above before Python can safely
replace software-stitched OC/im2col fallbacks with chained RKNN-style
ChannelTile.

## Appendix: Mapping conv_new_clean.py 6 strategies to RKNN planner states

Source: `examples/kernel_6_18/conv_new_clean.py`, function `run_conv2d` (line 746).

### Decision tree (lines 757-762)

```python
grouped_serial          = is_spatial and groups > 1 and not depthwise
spatial_im2col          = is_spatial and groups == 1 and not depthwise and
                          (weight_banks > RK_CBUF_BANKS//3 or output_bytes > buffer)
spatial_oc_serial       = is_spatial and groups == 1 and not depthwise and
                          out_c > 16 and (in_c % 16 != 0 or in_c >= 16)
depthwise_spatial_tiled = depthwise and is_spatial and
                          (out_h > depthwise_tile_h or out_c > align_c)
```

When none of the above match, the general path checks:

```python
if _needs_pointwise_oc_tile_schedule(...):  # Strategy 5: OC-tiled pointwise
elif _needs_pointwise_tile_schedule(...):    # Strategy 6: PC-chain pointwise
else:                                         # Default: simple direct submit
```

### Precise 1:1 mapping

| # | Python strategy | Trigger summary | RKNN planner function | Split method | Notes |
|---|-----------------|-----------------|----------------------|--------------|-------|
| 1 | `spatial_im2col` | Large weight spatial conv that exceeds CBUF budget | `fcn.005d4ac0` (layer-level) would use ChannelTile | `SPLIT_BY_K` (via implicit im2col→1x1) | **No direct HW equivalent** – Python converts to pointwise via software im2col. RKNN programs hardware spatial conv with channel tiling instead. This is the **biggest divergence** between Python and RKNN. |
| 2 | `grouped_serial` | Grouped spatial (groups>1, spatial) | Higher-level subgraph loop, not in tiling planner | `SPLIT_NONE` per group | Groups handled above `fcn.005d4ac0` level. The planner functions (fcn.005d4ac0/fcn.005f51c0) show no group iteration. |
| 3 | `spatial_oc_serial` | Spatial, moderate weight, out_c>16, in_c odd | `fcn.00388a18` (fixed-16 ChannelTile) + `fcn.005d4ac0` | `SPLIT_BY_Y_AND_K` with c_step=16 | Row tile=32 matches `out_h_tile_requirement()` in decomp. OC tile=16 matches `c_step=16`. Independent submit, not chained. |
| 4 | `depthwise_spatial_tiled` | Depthwise spatial, large H or C | `fcn.005d4ac0` depthwise path + `fcn.00384ed0` with `mode=1` | `SPLIT_BY_Y_AND_K` (depthwise) | Channel tile=32 matches HW depthwise bank limit. `mode=1` forces `k_tile=2, tile_count=2`. |
| 5 | `_needs_pointwise_oc_tile_schedule` | Specific 1x1 shapes (96→24, 144→24/32, 192→32/16, 256→32 at known resolutions) | `fcn.00387e3c` (main ChannelTile predicate) | `SPLIT_BY_K` (ChannelTile OC-slicing) | Hardcoded shapes == F2 min_weight_banks>3 cases. OC tile size (=32 for in_c>=192, =16 for smaller) matches `tile_count * 8`. **Closest to RKNN ChannelTile**. |
| 6 | `_needs_pointwise_tile_schedule` | Same shapes as #5 but with PC-chain | `fcn.005d4ac0` + `fcn.00307198`/`fcn.00312328` (reuse update) | `SPLIT_BY_Y` with weight_reuse | **Only strategy using PC-chain + weight_reuse**. The weight_reuse flag matches `CNA_CBUF_CON0_WEIGHT_REUSE`. Output offset calculation mimics `kstart`/`ys tart` in tile records. |
| 7 | Default | Everything else (small, simple shapes) | `fcn.005f38f0` / `fcn.005f51c0` | `SPLIT_NONE` | Single-tile path. Straightforward register programming. |

### Key observations for refactor

1. **im2col (strategy 1) is the outlier**: It's the only strategy that fundamentally changes the operation type (spatial → pointwise via software). RKNN never does this – the hardware spatial path works with proper register programming.

2. **OC/Channel tiling (strategies 3, 5) share structure**: Both do row tiles + OC/Channel tiles with a tile stride of 16 or 32. The difference is output format (nc1hwc2 vs flat) and whether they use PC-chain. They could share a loop structure.

3. **PC-chain (strategy 6) is the model for chaining**: This is the closest Python strategy to RKNN's actual execution model. The weight_reuse flag, output offset accumulation, and input offset tracking match the tile record fields in `fcn.005d4ac0`.

4. **Output format divergence**: The code uses 3 output readback formats:
   - `_unpack_flat_1x1_output` (for 1x1, strategies 5-7)
   - `_unpack_nc1hwc2_output` (for spatial)
   - `_unpack_grouped_spatial_output` (for grouped)
    
   RKNN tile-table dumps show all tiles share the same `xstart/ystart/kstart` fields regardless of format. A unified NC1HWC2 format should work for all.

5. **Missing RKNN state not replicated**:
   - `data_reuse` / `weight_reuse` per-tile flags (only weight_reuse used, and only as bool)
   - CNA feature/weight/CSC group mask (decomp §1187-1198)
   - `mc_treat_by_y_tile` / `mc_treat_by_k_tile` / `mc_treat_by_1c_*` fields
   - Per-tile `input_bank_num` / `weight_bank_num` (global data_bank only)

6. **PC-chain tails**: Only strategy 6 uses the 4-QWORD PC tail properly. The other multi-tile strategies (3, 4, 5) submit each tile independently, which is less efficient than RKNN's chained approach.

## Appendix B: Tiling method dispatch (r2 static analysis of fcn.005f38f0)

Confirmed by r2 disassembly of `fcn.005f38f0` at address `0x005f4d54..0x005f4f94`:

### Method comparison at the builder blocks

The tiling method value is loaded from `[x29+0x1188]` into `w0`/`w9`. The dispatch:

```asm
0x005f4d54: ldr w0, [x29, 0x1188]    ; load tiling method
0x005f4d5c: cmp w0, 3                ; method == 3?
0x005f4d64: sdiv w4, w19, w0         ; w4 = tile_count / method
0x005f4d6c: mul  w4, w4, w0          ; w4 = (tile_count / method) * method
0x005f4d80: b.eq 0x5f4eb4            ; if method == 3, METHOD_3 path
0x005f4d84: ldr w2, [x29, 0x1188]
0x005f4d88: cmp w2, 2                ; method == 2?
0x005f4d8c: b.eq 0x5f4f34            ; if method == 2, METHOD_2 path
                                      ; else DEFAULT path
```

### Method behaviors

| Method | Index calculation | Boundary marker | Inner marker | Notes |
|--------|-------------------|-----------------|--------------|-------|
| 2 | `tile_index & 1` | 3 | 1 | Two groups (even/odd). Used for K/OC-split: tile_count=2 means 2 groups of OC. |
| 3 | `tile_index % 3` | 7 | 1 | Three groups. Used for Y+K combined split |
| other | `tile_index % method` | 7 for remainder==3, 3 for remainder==1/2 | 1 | N groups, N = method value |

### Default path details

```asm
w5 = tile_count - w4       ; remainder = tile_count % method
w8 = 3                     ; marker for remainder==2
w7 = 7                     ; marker for remainder==3
w6 = 1                     ; marker for normal tiles
w9 = tiling_method         ; stored for use in loop

loop:
    w3 = loop_index / w9       ; quotient
    w3 = w3 * w9               ; (quotient * method)
    w3 = loop_index - w3       ; loop_index % method
    str w3, [x1]               ; store index % method in first vector
    
    if loop_index < w4:        ; tiles before boundary
        str 0, [x1]            ; clear in first vector
        str 1, [x2]            ; store marker 1 in second vector
    elif w5 == 3:
        str 7, [x2]            ; boundary marker 7
    elif w5 == 2:
        str 3, [x2]            ; boundary marker 3  
    elif w5 == 1:
        ; also stores boundary marker
```

## Appendix C: Live RKNN register capture from ops_rknn (dynamic analysis)

Captured from `~/npu/ops_rknn/conv2d_dump_regs` (source: `conv2d_dump_regs.cpp`) running under GDB on the remote NPU host (192.168.192.36). The GDB script `rknn.gdb` breaks at `rknn_destroy` and runs `python3 dump.py 2` to decode the register command buffer.

### Models tested

The test program runs 4 RKNN models:
1. `conv2d_fail_1x6_3x1_5x7.rknn` (in_c=1, out_c=6, kh=3, kw=1, H=5, W=7)
2. `conv2d_fail_2x4_3x3_6x6.rknn` (in_c=2, out_c=4, 3x3, 6x6)
3. `conv2d_fail_2x4_2x2_5x5.rknn` (in_c=2, out_c=4, 2x2, 5x5)
4. `conv2d_fail_1x32_5x5_10x10.rknn` (in_c=1, out_c=32, 5x5, 10x10)

### Key finding: ChannelTile register pattern for model 4 (1→32, 5x5)

The `conv2d_fail_1x32_5x5_10x10` model generates **5 chained tasks** via PC-chain:

| Tile | WEIGHT_KERNELS | DST_BASE_ADDR | DATA_CUBE_CHANNEL | SURFACE_ADD | PC core | Notes |
|------|---------------|---------------|-------------------|-------------|---------|-------|
| 0 | 32 | 0xfffe6000 | 31 (32 ch) | 72 | main | Full conv, base output |
| 1 | **16** | 0xfffe6000 | **15 (16 ch)** | 72 | core 1 | **ChannelTile**: half OC, same output base |
| 2 | 32 | **0xfffe6480** | 31 (32 ch) | 72 | core 2 | Multicore copy, offset output |
| 3 | 32 | **0xfffe60c0** | 31 (32 ch) | 72 | core 2 | Multicore, diff output |
| 4 | 32 | **0xfffe6180** | 31 (32 ch) | 72 | core 2 | Multicore, diff output |

### Critical observations from the register capture

1. **WEIGHT_KERNELS=16 in tile 1** confirms ChannelTile is OC-splitting (16 channels instead of 32). This IS the RKNN ChannelTile.

2. **DST_BASE_ADDR** is SAME for tiles 0-1 (0xfffe6000), different for tiles 2-4. Tiles 0 and 1 write to the same output buffer -> OC channels are at different offsets within the surface. Tiles 2-4 are multicore copies writing to separate output regions.

3. **PC_REGISTER_AMOUNTS** contains:
   - `RESERVED_0`: actually **CORE INDEX** (1 or 2). Not reserved!
   - `PC_DATA_AMOUNT`: 14 register words per tile body
   So the PC tail is: `PC_BASE_ADDRESS | (core<<12) | register_amount`

4. **SURFACE_ADD is constant (72)** across all tiles → same output format
   - `72 >> 4 = 4.5` → This is `out_width_stride * ceil(align_out_c / 16)`
   - For the 1→32 5x5 model: width_stride=10 (aligned), out_atoms = 6*6=36, out_width_stride=36
   - SURF_ADD = 36 * (32/16) << 4 = 36 * 2 << 4 = 72 << 4... wait
   - 72 >> 4 = 4, at `out_width_stride=36`, out_c=32 → `max(2, 32/16) = 2`, so `36*2*16=1152`, shifted >>4 = 72. **Confirmed**.

5. **PPU registers** are programmed alongside CNA/CORE/DPU for every conv tile. The captured register set includes:
   - `REG_PPU_S_POINTER`, `REG_PPU_DATA_CUBE_IN_CHANNEL`, `REG_PPU_DATA_CUBE_OUT_CHANNEL`
   - `REG_PPU_DATA_FORMAT`, `REG_PPU_DST_BASE_ADDR`, `REG_PPU_DST_SURF_STRIDE`
   - `REG_PPU_MISC_CTRL`, `REG_PPU_OPERATION_MODE_CFG`
   - `REG_PPU_RDMA_RDMA_*` (S_POINTER, SRC_BASE_ADDR, DATA_FORMAT, CUBE_IN_CHANNEL, LINE_STRIDE, SURF_STRIDE)
   
   **Our Python driver sets NONE of these PPU registers.** This is a significant omission.
   Note: i dont think PPU is needed. just dead code

6. **PC_VERSION** values seen: `0x00020000`, `0x00010000`, `0x00000100`, `0x00550000` — these are version/doorbell handshake values.

7. **DPU_RDMA_RDMA_OPERATION_ENABLE**: value `1777180458` = `0x69e000aa` — This is an RDMA configuration value. The Python driver doesn't program this register at all.

### Complete register set per tile (14 body words + 4 PC tail)

RKNN emits these registers for each conv tile (in order):
```
1.  DPU_RDMA_OPERATION_ENABLE     (RDMA setup)
2.  PC_VERSION                     (version doorbell)
3.  CNA_CBUF_CON0                  (bank config)
4.  CNA_CONV_CON1                  (conv control 1)
5.  DPU_S_POINTER                  (ping-pong config)
6.  CNA_CONV_CON1                  (may repeat with different fields)
7.  CNA_CONV_CON2                  (feature grains)
8.  CNA_CONV_CON3                  (stride)
9.  CNA_DATA_SIZE0                 (input H/W)
10. CNA_DATA_SIZE1                 (input channels)
11. CNA_DATA_SIZE2                 (output width)
12. CNA_DATA_SIZE3                 (output atomics)
13. CNA_WEIGHT_SIZE0               (weight total)
14. CNA_WEIGHT_SIZE1               (weight per kernel)
15. CNA_WEIGHT_SIZE2               (kernel dims + count)
16. CNA_CBUF_CON0                  (may repeat bank config)
17. CNA_CBUF_CON1                  (data entries)
18-22. CNA_CVT_CON0..4             (convert scales)
23. CNA_FEATURE_DATA_ADDR          (input address)
24. CNA_DMA_CON0                   (burst length)
25. CNA_DMA_CON1                   (line stride)
26. CNA_DMA_CON2                   (surface stride)
27. CNA_FC_DATA_SIZE0              (DMA H/W)
28. CNA_FC_DATA_SIZE1              (DMA channels)
29. CNA_DCOMP_ADDR0                (weight address)
30. CNA_CVT_CON5                   (per-channel cvt mask)
31. CORE_MISC_CFG                  (misc: precision, dw, op_en)
32. CORE_DATAOUT_SIZE_0            (output H/W)
33. CORE_DATAOUT_SIZE_1            (output channel)
34. DPU_FEATURE_MODE_CFG           (burst, mode)
35. DPU_DATA_FORMAT                (precision)
36. DPU_DST_BASE_ADDR              (output address)
37. DPU_DST_SURF_STRIDE            (output stride)
38. DPU_DATA_CUBE_WIDTH            (width-1)
39. DPU_DATA_CUBE_HEIGHT           (height-1)
40. DPU_DATA_CUBE_CHANNEL          (channels)
41. DPU_BS_CFG                     (batch/norm bypass)
42. DPU_BS_OW_CFG                  (output width cfg)
43. DPU_WDMA_SIZE_0                (WDMA channel)
44. DPU_WDMA_SIZE_1                (WDMA H/W)
45. DPU_BN_CFG                     (bn bypass)
46. DPU_EW_CFG                     (ew bypass)
47. DPU_EW_CVT_SCALE_VALUE         (scale)
48. DPU_OUT_CVT_SCALE              (fp32tofp16)
49. DPU_SURFACE_ADD                (surface offset)
50-53. PPU_S_POINTER, PPU_DATA_CUBE_IN_CHANNEL, PPU_DATA_CUBE_OUT_CHANNEL,
      PPU_DATA_FORMAT, PPU_DST_BASE_ADDR, PPU_DST_SURF_STRIDE,
      PPU_MISC_CTRL, PPU_OPERATION_MODE_CFG,
      PPU_RDMA_RDMA_S_POINTER, PPU_RDMA_RDMA_SRC_BASE_ADDR,
      PPU_RDMA_RDMA_DATA_FORMAT, PPU_RDMA_RDMA_CUBE_IN_CHANNEL,
      PPU_RDMA_RDMA_SRC_LINE_STRIDE, PPU_RDMA_RDMA_SRC_SURF_STRIDE
54. PC_BASE_ADDRESS                 (next segment address)
55. PC_REGISTER_AMOUNTS             (next amount + core)
56. PC_VERSION                      (version doorbell)
57. PC_OPERATION_ENABLE             (enable + core mask)
```

Note: The tile bodies seen in the capture have exactly 14 register WORDS (qwords). The Python driver emits ~50+ registers. This difference is because RKNN uses a **sparse encoding**: it only emits registers whose values differ from the hardware default or the previous tile. The broadcast/common registers are programmed once and reused.

## Appendix D: Reuse update mechanics (static analysis)

### fcn.002efd38 (register value patcher by index)

```
Inputs: ctx(x0), index(x1), value(x2)
x3  = ctx.field_0x288      ; command vector struct
x19 = x3[0x28]             ; index base (offset array)
x21 = x3[0x08]             ; base pointer (command entry array)
x19 = x19 + index * 8      ; compute entry address
if x21 + x19 == 0:         ; sentinel check
    return -1
x0  = x21[x19]             ; load old command entry
w1  = w2                   ; new value
x0  = patch_reg_value(x0, w1) ; bfi x0, x1, #16, #32 (bits 16-47)
x21[x19] = x0              ; store back
ctx[0x11] = 1              ; set dirty flag
return 0
```

### fcn.002efce0 (pair-record updater)

```
Inputs: ctx(x0), pair_record(x1), value(x2)
x19 = pair_record[8]       ; entry_ptr
x1  = pair_record[0]       ; side_int_ptr
if x19 == 0: return;       ; no entry
if x1 != 0: side_int_ptr[4] = value  ; write side value
x20 = ctx
x0  = *entry_ptr           ; load old command
w1  = w2                   ; new value
x0  = patch_reg_value(x0, w1) ; bfi bits 16-47
*entry_ptr = x0            ; store back
ctx[0x11] = 1              ; set dirty flag
return 0
```

### fcn.00100a40 (low-level patch primitive)

```asm
bfi x0, x1, #16, #32      ; replace bits 16-47 of x0 with x1
ret
```

This matches the Python `E(target, reg, value)` encoding:
- Bits 48-63: target
- Bits 16-47: value (what gets patched)
- Bits 0-15: register address

The key insight from the reuse path: when `fcn.00307198` or `fcn.00312328` walks the tile list and sets `data_reuse`/`weight_reuse`, it calls `fcn.002efd38(ctx, index, new_value)` for each register that needs updating. The **index** identifies which register position in the flat command array (at ctx+0x288) needs patching.

This means the reuse update does NOT emit new registers. It patches existing register commands that were placed in the command vector during the initial tile setup. The patch targets are identified by their position in the command array (index), not by register address.

## Appendix E: Remaining reverse engineering gaps

As of May 2026, after this round of reverse engineering:

| Gap | Status | What's needed |
|-----|--------|---------------|
| Split-method enum values | **RESOLVED**: methods 2, 3, and generic N | Confirmed by r2 disassembly. Method 2 = two groups (K-split), method 3 = three groups (Y+K), generic = N groups |
| Register fields per tiling family | **PARTIAL**: live capture shows WEIGHT_KERNELS change for ChannelTile | Need to capture more model variants to see all register field differences |
| Reuse programming sequence for chained tiles | **PARTIALLY RESOLVED**: patch mechanics understood (bfi via index), pair-record format known | Need GDB trace on `fcn.002efd38`/`fcn.002efce0` to map index values → register addresses |
| Y/K decision tree | **MAPPED** in Appendix A | See strategy-to-RKNN mapping table |
| Output placement math | **PARTIAL**: SURFACE_ADD formula confirmed, DST offsets seen in capture | Full tile placement math needs shape sweep |
| ChannelTile vs OC tiling | **CONFIRMED IDENTICAL**: WEIGHT_KERNELS changed from 32→16, matching OC tile pattern | RKNN ChannelTile IS OC-channel splitting with PC-chain chaining |
| PPU registers | **DISCOVERED**: ~14 PPU registers programmed per tile, missing from Python driver | Need to add PPU register programming to Python conv driver |
| Reuse update → register targets | **KNOWN MECHANISM**: index-based patching into flat command array at ctx+0x288 | Need to map tile list → which command indices get patched |

### Next recommended reverse engineering steps

1. **GDB trace of fcn.002efd38 and fcn.002efce0**: Break on these functions while running a conv2d model that triggers ChannelTile. Log the caller PC, index, old value, new value, and decode the patched command.

2. **Shape sweep with ops_rknn**: Run conv2d_dump_regs with many more model shapes to see how the register programming changes (WEIGHT_KERNELS split, SURFACE_ADD, DST_BASE_ADDR, PPU registers).

3. **PPU register programming**: Add PPU_S_POINTER, PPU_DST_BASE_ADDR, PPU_DST_SURF_STRIDE, PPU_DATA_CUBE_OUT_CHANNEL and the PPU_RDMA registers to the Python driver.

4. **Reuse update validation**: Add CNA feature/weight/CSC group mask bits to the Python driver, matching the fcn.00101be8 formatter's mask layout.
