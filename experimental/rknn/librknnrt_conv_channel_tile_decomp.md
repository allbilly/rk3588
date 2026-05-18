# librknnrt.so Conv ChannelTile decompilation

Source binary: `ref/rknn/librknnrt.so`

Command style used:

```sh
rabin2 -zz ref/rknn/librknnrt.so | rg 'min_weight_banks|ChannelTile'
r2 -q -e scr.color=false -e bin.relocs.apply=true -c 'aaa' -c 'axt @ 0x60f388' -c q ref/rknn/librknnrt.so
r2 -q -e scr.color=false -e bin.relocs.apply=true -c aa -c 's 0x00387e3c' -c af -c pdf -c q ref/rknn/librknnrt.so
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
| `0x0062cf58` | `Unknown split Method -> %d` | `fcn.005f1998` | Abort helper for invalid split-method enum. |
| `0x0062cf78` | tile-table header with `xstart`, `ystart`, `kstart`, `data reuse`, `weight reuse`, `mc_treat_by_*` | `fcn.005f1cd0` | Tile-table dump/formatter. Shows the actual per-tile fields RKNN uses after planning. |
| `0x0062d040` | `illegal tiling method %d` | `fcn.005f38f0` | Fatal diagnostic for illegal tiling method. |
| `0x0062d298` | `X tile buffer overflow!` | `fcn.005f51c0` | Fatal diagnostic for input-width/X tile overflow. |

The important separation is:

- `fcn.00387e3c` and `fcn.00388a18` answer: should this conv switch to
  ChannelTile because bank pressure is too high?
- `fcn.005d4ac0` answers: after a conv is being tiled, how are Y/K split
  records chosen and checked?
- `fcn.00307198`, `fcn.00312328`, `fcn.005c6b50`, and `fcn.005f1cd0` answer:
  how are data/weight reuse flags, bank counts, and tile records validated or
  printed?

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

## Reuse and tile-table helpers

### fcn.00307198 and fcn.00312328

These two functions are the xrefs for:

```text
0x0060c768 failed to update data and weight reuse!
```

They are not reached by the direct `min_weight_banks > 3` string, but they are
part of the same compiled tiling pipeline. Their role is to walk the tile list
after split planning and mark which adjacent tiles may reuse data and/or
weights.

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
