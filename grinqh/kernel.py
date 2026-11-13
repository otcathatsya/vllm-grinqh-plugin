import os
from itertools import product
from typing import Optional, List

import torch
from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_gptq_marlin_linear
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

from .utils import unpack_pyt, BitLevel, grinqh_input_postshuffle

num_warps_options = [4, 8]
num_stages_options = [1, 2, 3, 4]
block_m_options = [8, 16, 32]
block_n_options = [8, 16, 32]
num_block_m_loops_options = [1, 4, 8, 16]

configs = []
for block_m, block_n, num_warps, num_stages, num_block_m_loops in product(
        block_m_options, block_n_options, num_warps_options, num_stages_options, num_block_m_loops_options
):
    total_elements = block_m * block_n
    if num_warps * 32 > (total_elements / 2):
        continue
    if num_stages >= 4 and block_m == 64 and block_n == 64:
        continue
    if num_warps == 16 and (block_m <= 32 or block_n <= 32):
        continue

    configs.append(
        triton.Config(
            {"BLOCK_M": block_m, "BLOCK_N": block_n, "NUM_BLOCK_M_LOOPS": num_block_m_loops},
            num_warps=num_warps,
            num_stages=num_stages,
        )
    )


@triton.jit
def grinqh_gemv_uint32_packed(
        Y, A, X, scales,
        t0: tl.constexpr, t1: tl.constexpr, t2: tl.constexpr, t3: tl.constexpr,
        N: tl.constexpr, M: tl.constexpr,
        SCALES_BLOCK_SIZE_M: tl.constexpr, SCALES_BLOCK_SIZE_N: tl.constexpr,
        MAX_BITS: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, NUM_BLOCK_M_LOOPS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    accum_dtype = tl.float32

    NUM_ZONES: tl.constexpr = MAX_BITS // 2
    # round to nearest power of 2 (up to max 4 zones) cause triton req
    PADDED_ZONES: tl.constexpr = 4 if NUM_ZONES > 2 else NUM_ZONES
    A_ROW_STRIDE: tl.constexpr = NUM_ZONES * (N // 4)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    zone_indices = tl.arange(0, PADDED_ZONES)

    safe_zone_indices = tl.where(zone_indices < NUM_ZONES, zone_indices, 0)
    offs_n_zones = safe_zone_indices[:, None] * (N // 4) + offs_n[None, :]

    acc = tl.zeros((4, 4, BLOCK_N), dtype=accum_dtype)

    # always 4 as opposed to padded ones
    neighbourhood_indices = tl.arange(0, 4)
    Y_loc_base = neighbourhood_indices[:, None] * N + offs_n[None, :]
    zone_offsets = neighbourhood_indices[:, None, None] * (N // 4)
    Y_ptr = Y_loc_base[None, :, :] + zone_offsets

    # pre-calculate as many constexprs as possible to replace dynamic tl.where selections
    BIAS_BASE: tl.constexpr = 1 << (MAX_BITS - 1)
    BIAS_T0: tl.constexpr = BIAS_BASE >> 2
    BIAS_T1: tl.constexpr = BIAS_BASE >> 4
    BIAS_T2: tl.constexpr = BIAS_BASE >> 6
    BIAS_T3: tl.constexpr = BIAS_BASE >> 8

    DELTA_T0: tl.constexpr = BIAS_T0
    DELTA_T1: tl.constexpr = BIAS_T1 - BIAS_T0
    DELTA_T2: tl.constexpr = BIAS_T2 - BIAS_T1
    DELTA_T3: tl.constexpr = BIAS_T3 - BIAS_T2

    zone_indices_bc = tl.arange(0, PADDED_ZONES)[None, :]
    precision_mask = zone_indices_bc < NUM_ZONES

    shift_dest = (3 - tl.arange(0, PADDED_ZONES)[None, :, None].to(tl.uint32)) * 2
    byte_iter = tl.arange(0, 4)[:, None, None, None]
    shift_amt = byte_iter * 8  # [0, 8, 16, 24]

    for iblockm in range(NUM_BLOCK_M_LOOPS):
        offs_m = pid_m * BLOCK_M * NUM_BLOCK_M_LOOPS + tl.arange(0, BLOCK_M) + iblockm * BLOCK_M

        # transform to (BM, 4, BN)
        A_loc = A[None] + (offs_m[:, None, None] * A_ROW_STRIDE + offs_n_zones[None, :, :])
        X_loc = X + offs_m

        x_chunk = tl.load(X_loc, mask=offs_m < M, other=0.0, eviction_policy='evict_last')

        abs_x = tl.abs(x_chunk)

        # spam tl.where to concat the threshold masks into one big (BM, 4) mask
        threshold_mask = tl.where(zone_indices_bc == 0, (abs_x > t0)[:, None],
                                  tl.where(zone_indices_bc == 1, (abs_x > t1)[:, None],
                                           tl.where(zone_indices_bc == 2, (abs_x > t2)[:, None],
                                                    (abs_x > t3)[:, None])))

        b0 = tl.where(abs_x > t0, DELTA_T0, 0) if NUM_ZONES > 0 else 0
        b1 = tl.where(abs_x > t1, DELTA_T1, 0) if NUM_ZONES > 1 else 0
        b2 = tl.where(abs_x > t2, DELTA_T2, 0) if NUM_ZONES > 2 else 0
        b3 = tl.where(abs_x > t3, DELTA_T3, 0) if NUM_ZONES > 3 else 0

        biases_per_inp = b0 + b1 + b2 + b3
        bias_total = tl.sum(biases_per_inp * x_chunk.to(accum_dtype), 0)

        zone_active_mask = threshold_mask & precision_mask

        mask_load = zone_active_mask[:, :, None] * (offs_n[None, None, :] < (N // 4)) * (offs_m[:, None, None] < M)

        a_packed = tl.load(A_loc, mask=mask_load, other=0.0, eviction_policy='evict_first')

        weights_unpacked = tl.zeros((BLOCK_M, 4, BLOCK_N), dtype=tl.uint32)

        # n = 'neighbourhood idx', since we can unpack 4x uint8s at once (uint32) and we need 16 to fill one uint32,
        # this loop covers the minimum iterations to reconstruct one 'neighbourhood' of 4(x4) completed uint8s
        for n in tl.static_range(0, 4):
            # get the same range of 2-bits in every zone (aka all values needed to reconstruct a full neighbourhood of uint8s)
            shift_src = (3 - n) * 2
            bits_isolated = (a_packed >> shift_src) & 0x03030303

            # shift it depending on zone (using dim broadcast)
            bits_positioned = bits_isolated << shift_dest

            # reduce: we can sum because there they are all offset cleanly, yielding the original uint8s
            weight_z_val = tl.sum(bits_positioned, 1)

            # basically weights_unpacked[:, z, :] = weight_z_val
            weights_unpacked = tl.where(neighbourhood_indices[None, :, None] == n,
                                        weight_z_val[:, None, :],
                                        weights_unpacked)

        # get byte [0,1,2,3] so we end up with (bytes, M, zones, N)
        w_raw = (weights_unpacked[None, :, :, :] >> shift_amt) & 0xFF
        # we cast to int8 first to recover the negative sign, shift because we need to
        # correct magnitude (it's stored in MSBs) before accumulation if <8 max bits
        w_float = w_raw.to(tl.int8) >> (8 - MAX_BITS)

        offs_n_scale = Y_ptr // SCALES_BLOCK_SIZE_N

        # get the right group of scales depending on current block and num loops
        offs_m_scale = (pid_m * BLOCK_M * NUM_BLOCK_M_LOOPS + iblockm * BLOCK_M) // SCALES_BLOCK_SIZE_M
        scale_vals = tl.load(scales + offs_m_scale * (N * 4 // SCALES_BLOCK_SIZE_N) + offs_n_scale,
                             mask=(offs_n[None, None, :] < (N // 4)), other=0.0).to(accum_dtype)

        partial_res = (bias_total + tl.sum(w_float * x_chunk[None, :, None, None].to(accum_dtype), 1))

        # we wanna sum across M in every 2D block so match up dims (fp32) (mask (1, BM, 1, 1))
        acc += scale_vals * partial_res

    # Technically this is a strided write and wouldn't read correctly due to the byte 'neighbourhoods',
    # but this way it's local to each zone 'lane' (+ we pre-transform our input for this to be correct)
    mask_out_base = offs_n < (N // 4)
    mask_out_final = mask_out_base[None, None, :]

    # Atomicity is required across M tiles, but the accumulation does not
    # synchronize any other memory.
    tl.atomic_add(Y_ptr + Y, acc, mask=mask_out_final, sem="relaxed")


@triton.jit
def grinqh_gemv_uint32_packed_batch_union(
        Y, A, X, scales,
        t0: tl.constexpr, t1: tl.constexpr, t2: tl.constexpr, t3: tl.constexpr,
        N: tl.constexpr, M: tl.constexpr,
        BATCH_SIZE: tl.constexpr, BLOCK_BATCH: tl.constexpr,
        SCALES_BLOCK_SIZE_M: tl.constexpr, SCALES_BLOCK_SIZE_N: tl.constexpr,
        MAX_BITS: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, NUM_BLOCK_M_LOOPS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    accum_dtype = tl.float32

    num_zones: tl.constexpr = MAX_BITS // 2
    padded_zones: tl.constexpr = 4 if num_zones > 2 else num_zones
    a_row_stride: tl.constexpr = num_zones * (N // 4)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    zone_indices = tl.arange(0, padded_zones)
    safe_zone_indices = tl.where(zone_indices < num_zones, zone_indices, 0)
    offs_n_zones = (
            safe_zone_indices[:, None] * (N // 4) + offs_n[None, :]
    )

    batch_offsets = tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_offsets < BATCH_SIZE
    acc = tl.zeros(
        (BLOCK_BATCH, 4, 4, BLOCK_N),
        dtype=accum_dtype,
    )

    neighbourhood_indices = tl.arange(0, 4)
    y_loc_base = neighbourhood_indices[:, None] * N + offs_n[None, :]
    zone_offsets = neighbourhood_indices[:, None, None] * (N // 4)
    y_ptr = y_loc_base[None, :, :] + zone_offsets

    bias_base: tl.constexpr = 1 << (MAX_BITS - 1)
    bias_t0: tl.constexpr = bias_base >> 2
    bias_t1: tl.constexpr = bias_base >> 4
    bias_t2: tl.constexpr = bias_base >> 6
    bias_t3: tl.constexpr = bias_base >> 8

    delta_t0: tl.constexpr = bias_t0
    delta_t1: tl.constexpr = bias_t1 - bias_t0
    delta_t2: tl.constexpr = bias_t2 - bias_t1
    delta_t3: tl.constexpr = bias_t3 - bias_t2

    zone_indices_bc = tl.arange(0, padded_zones)[None, :]
    precision_mask = zone_indices_bc < num_zones

    shift_dest = (
                         3
                         - tl.arange(0, padded_zones)[None, :, None].to(tl.uint32)
                 ) * 2
    byte_iter = tl.arange(0, 4)[:, None, None, None]
    shift_amt = byte_iter * 8

    for iblockm in range(NUM_BLOCK_M_LOOPS):
        offs_m = (
                pid_m * BLOCK_M * NUM_BLOCK_M_LOOPS
                + tl.arange(0, BLOCK_M)
                + iblockm * BLOCK_M
        )

        a_loc = A[None] + (
                offs_m[:, None, None] * a_row_stride
                + offs_n_zones[None, :, :]
        )
        x_loc = X + batch_offsets[:, None] * M + offs_m[None, :]
        x_mask = batch_mask[:, None] & (offs_m[None, :] < M)
        x_chunk = tl.load(
            x_loc,
            mask=x_mask,
            other=0.0,
            eviction_policy="evict_last",
        )

        union_abs_x = tl.max(tl.abs(x_chunk), axis=0)
        threshold_mask = tl.where(
            zone_indices_bc == 0,
            (union_abs_x > t0)[:, None],
            tl.where(
                zone_indices_bc == 1,
                (union_abs_x > t1)[:, None],
                tl.where(
                    zone_indices_bc == 2,
                    (union_abs_x > t2)[:, None],
                    (union_abs_x > t3)[:, None],
                ),
            ),
        )

        b0 = (
            tl.where(union_abs_x > t0, delta_t0, 0)
            if num_zones > 0
            else 0
        )
        b1 = (
            tl.where(union_abs_x > t1, delta_t1, 0)
            if num_zones > 1
            else 0
        )
        b2 = (
            tl.where(union_abs_x > t2, delta_t2, 0)
            if num_zones > 2
            else 0
        )
        b3 = (
            tl.where(union_abs_x > t3, delta_t3, 0)
            if num_zones > 3
            else 0
        )

        biases_per_input = b0 + b1 + b2 + b3
        bias_total = tl.sum(
            biases_per_input[None, :] * x_chunk.to(accum_dtype),
            axis=1,
        )

        zone_active_mask = threshold_mask & precision_mask
        mask_load = (
                zone_active_mask[:, :, None]
                & (offs_n[None, None, :] < (N // 4))
                & (offs_m[:, None, None] < M)
        )
        a_packed = tl.load(
            a_loc,
            mask=mask_load,
            other=0.0,
            eviction_policy="evict_first",
        )

        weights_unpacked = tl.zeros(
            (BLOCK_M, 4, BLOCK_N),
            dtype=tl.uint32,
        )
        for neighbourhood in tl.static_range(0, 4):
            shift_src = (3 - neighbourhood) * 2
            bits_isolated = (
                                    a_packed >> shift_src
                            ) & 0x03030303
            bits_positioned = bits_isolated << shift_dest
            weight_zone_value = tl.sum(bits_positioned, 1)
            weights_unpacked = tl.where(
                neighbourhood_indices[None, :, None] == neighbourhood,
                weight_zone_value[:, None, :],
                weights_unpacked,
            )

        weight_raw = (
                             weights_unpacked[None, :, :, :] >> shift_amt
                     ) & 0xFF
        weight_float = weight_raw.to(tl.int8) >> (8 - MAX_BITS)

        offs_n_scale = y_ptr // SCALES_BLOCK_SIZE_N
        offs_m_scale = (
                               pid_m * BLOCK_M * NUM_BLOCK_M_LOOPS
                               + iblockm * BLOCK_M
                       ) // SCALES_BLOCK_SIZE_M
        scale_values = tl.load(
            scales
            + offs_m_scale * (N * 4 // SCALES_BLOCK_SIZE_N)
            + offs_n_scale,
            mask=(offs_n[None, None, :] < (N // 4)),
            other=0.0,
        ).to(accum_dtype)

        partial_result = bias_total[:, None, None, None] + tl.sum(
            weight_float[None, :, :, :, :]
            * x_chunk[:, None, :, None, None].to(accum_dtype),
            axis=2,
        )
        acc += scale_values[None, :, :, :] * partial_result

    output_ptr = (
            Y
            + batch_offsets[:, None, None, None] * (N * 4)
            + y_ptr[None, :, :, :]
    )
    output_mask = (
            batch_mask[:, None, None, None]
            & (offs_n[None, None, None, :] < (N // 4))
    )
    tl.atomic_add(
        output_ptr,
        acc,
        mask=output_mask,
        sem="relaxed",
    )


@triton.jit
def _grinqh_to_marlin_int4_kernel(
        grinqh_ptr,
        marlin_ptr,
        grinqh_stride_k,
        num_groups: tl.constexpr,
        n_tiles: tl.constexpr,
):
    """Convert one 16x64 weight tile directly to Marlin's packed layout."""
    tile_id = tl.program_id(0)
    output_offset = tl.arange(0, 128)

    k_tile = tile_id // n_tiles
    n_tile = tile_id % n_tiles
    marlin_thread = output_offset // 4
    marlin_warp = output_offset % 4

    tile_k = k_tile * 16 + (marlin_thread % 4) * 2
    tile_n = (
            n_tile * 64
            + marlin_warp * 16
            + marlin_thread // 4
    )

    packed = tl.zeros((128,), dtype=tl.uint32)
    k_offsets: tl.constexpr = (0, 8, 0, 8, 1, 9, 1, 9)
    n_offsets: tl.constexpr = (0, 0, 8, 8, 0, 0, 8, 8)
    for packed_position in tl.static_range(0, 8):
        source_k = tile_k + k_offsets[packed_position]
        source_n = tile_n + n_offsets[packed_position]
        source_group = source_n % num_groups
        source_lane = source_n // num_groups
        source_shift = (
                (source_lane % 4) * 8
                + (3 - source_lane // 4) * 2
        )
        source_offset = (
                source_k * grinqh_stride_k + source_group
        )

        high_plane = tl.load(
            grinqh_ptr + source_offset,
        ).to(tl.uint32)
        low_plane = tl.load(
            grinqh_ptr + source_offset + num_groups,
        ).to(tl.uint32)
        raw_int4 = (
                (((high_plane >> source_shift) & 3) << 2)
                | ((low_plane >> source_shift) & 3)
        )
        packed |= (raw_int4 ^ 8) << (packed_position * 4)

    tl.store(
        marlin_ptr + tile_id * 128 + output_offset,
        packed,
    )


def grinqh_to_marlin_int4(
        weight_packed: torch.Tensor,
        original_n: int,
) -> torch.Tensor:
    """Return transient Marlin-packed uint4b8 weights for max-4 GRINQH."""
    size_k = weight_packed.shape[0]
    size_n = original_n * 4
    num_groups = size_n // 16

    if size_k % 16 != 0:
        raise ValueError(
            "GRINQH-to-Marlin conversion requires "
            f"K % 16 == 0, got {size_k}"
        )
    if size_n % 64 != 0:
        raise ValueError(
            "GRINQH-to-Marlin conversion requires "
            f"N % 64 == 0, got {size_n}"
        )
    if weight_packed.shape[1] != size_n // 8:
        raise ValueError(
            "Expected max-4 GRINQH weight shape "
            f"({size_k}, {size_n // 8}), got {tuple(weight_packed.shape)}"
        )

    output = torch.empty(
        (size_k // 16, size_n * 2),
        device=weight_packed.device,
        dtype=torch.uint32,
    )
    grid = ((size_k // 16) * (size_n // 64),)
    _grinqh_to_marlin_int4_kernel[grid](
        weight_packed,
        output,
        weight_packed.stride(0),
        num_groups=num_groups,
        n_tiles=size_n // 64,
        num_warps=4,
    )
    return output.view(torch.int32)


@triton.jit
def _grinqh_permute_marlin_scales_kernel(
        input_ptr,
        output_ptr,
        numel: tl.constexpr,
        PERM_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    block_offsets = offsets // PERM_SIZE * PERM_SIZE
    within_block = offsets % PERM_SIZE

    if PERM_SIZE == 64:
        source_within_block = (
                within_block // 8 + (within_block % 8) * 8
        )
    else:
        source_within_block = (
                2 * (within_block // 8)
                + within_block % 2
                + 8 * ((within_block % 8) // 2)
        )

    values = tl.load(
        input_ptr + block_offsets + source_within_block,
        mask=mask,
    )
    tl.store(output_ptr + offsets, values, mask=mask)


def grinqh_permute_marlin_scales(
        scales: torch.Tensor,
        size_k: int,
        group_size: int,
) -> torch.Tensor:
    """Permute Marlin scales without host indexing during CUDA graph capture."""
    perm_size = (
        64
        if group_size < size_k and group_size != -1
        else 32
    )
    if scales.numel() % perm_size != 0:
        raise ValueError(
            f"Marlin scale count {scales.numel()} is not divisible "
            f"by permutation size {perm_size}"
        )

    output = torch.empty_like(
        scales,
        memory_format=torch.contiguous_format,
    )
    block_size = 256
    grid = (triton.cdiv(scales.numel(), block_size),)
    _grinqh_permute_marlin_scales_kernel[grid](
        scales,
        output,
        numel=scales.numel(),
        PERM_SIZE=perm_size,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


def grinqh_prefill_marlin(
        x_flat: torch.Tensor,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        group_size: int,
        original_n: int
) -> torch.Tensor:
    """Run max-4 prefill through transient Marlin W4A16 weights."""
    size_k = weight_packed.shape[0]
    size_n = original_n * 4

    marlin_weight = grinqh_to_marlin_int4(
        weight_packed,
        original_n,
    )

    marlin_scales = grinqh_permute_marlin_scales(
        weight_scale,
        size_k=size_k,
        group_size=group_size,
    )
    return apply_gptq_marlin_linear(
        input=x_flat,
        weight=marlin_weight,
        weight_scale=marlin_scales,
        weight_zp=None,
        workspace=None,
        wtype=scalar_types.uint4b8,
        output_size_per_partition=size_n,
        input_size_per_partition=size_k
    )


@torch.compile(fullgraph=True)
def grinqh_prefill_gemm_compiled(x_flat, weight_packed, weight_scale, group_size, original_n, max_bits):
    output_dtype = x_flat.dtype
    SCALES_BLOCK_M = group_size
    SCALES_BLOCK_N = 1
    M, _ = weight_packed.shape

    w_grinqh = weight_packed.view(torch.uint32)
    row_indices = torch.arange(w_grinqh.shape[0], device=w_grinqh.device)

    w_unpacked = unpack_pyt(w_grinqh, row_indices, BitLevel(max_bits // 2), original_n)

    shift = 8 - max_bits
    w_sign_extended = (w_unpacked.view(torch.int8) << shift) >> shift

    w_recovered_i8 = grinqh_input_postshuffle(w_sign_extended)

    w_reshaped = w_recovered_i8.view(
        M // SCALES_BLOCK_M, SCALES_BLOCK_M,
        (original_n * 4) // SCALES_BLOCK_N, SCALES_BLOCK_N)

    scale_reshaped = weight_scale.view(
        weight_scale.shape[0], 1,
        weight_scale.shape[1], 1)

    w_scaled = (w_reshaped.to(output_dtype) * scale_reshaped).view(M, original_n * 4)
    return x_flat @ w_scaled


def get_env_config():
    w = int(os.environ.get("GRINQH_NUM_WARPS", 8))
    s = int(os.environ.get("GRINQH_NUM_STAGES", 1))
    bm = int(os.environ.get("GRINQH_BLOCK_M", 32))
    bn = int(os.environ.get("GRINQH_BLOCK_N", 32))
    bml = int(os.environ.get("GRINQH_NUM_BLOCK_M_LOOPS", 2))

    return triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "NUM_BLOCK_M_LOOPS": bml},
        num_warps=w,
        num_stages=s,
    )


GEMV_CONFIG = get_env_config()


def grinqh_gemv_batch_union(
        x: torch.Tensor,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        max_bits: int,
        group_size: int,
        original_n: int,
        thresholds: List[float],
) -> torch.Tensor:
    x_flat = x.view(-1, x.shape[-1])
    batch_size = x_flat.shape[0]
    M, _ = weight_packed.shape
    N = original_n

    grid = lambda META: (
        triton.cdiv(N // 4, META["BLOCK_N"]),
        triton.cdiv(M // META["NUM_BLOCK_M_LOOPS"], META["BLOCK_M"]),
    )
    y = torch.zeros((batch_size, N * 4), device=x.device, dtype=torch.float32)

    grinqh_gemv_uint32_packed_batch_union[grid](
        Y=y, A=weight_packed, X=x_flat, scales=weight_scale,
        t0=thresholds[0], t1=thresholds[1],
        t2=thresholds[2], t3=thresholds[3],
        N=N, M=M, BATCH_SIZE=batch_size, BLOCK_BATCH=triton.next_power_of_2(batch_size),
        SCALES_BLOCK_SIZE_M=group_size, SCALES_BLOCK_SIZE_N=1, MAX_BITS=max_bits,
        BLOCK_N=GEMV_CONFIG.kwargs["BLOCK_N"], BLOCK_M=GEMV_CONFIG.kwargs["BLOCK_M"],
        NUM_BLOCK_M_LOOPS=GEMV_CONFIG.kwargs["NUM_BLOCK_M_LOOPS"],
        num_warps=GEMV_CONFIG.num_warps, num_stages=GEMV_CONFIG.num_stages
    )

    y = y.to(x.dtype)
    if bias is not None:
        y.add_(bias)

    return y.view(*x.shape[:-1], N * 4)


@torch.library.custom_op("grinqh::gemm_or_gemv", mutates_args=[])
def grinqh_gemm_or_gemv(
        x: torch.Tensor,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        max_bits: int,
        group_size: int,
        original_n: int,
        thresholds: List[float],
        use_marlin: bool = False
) -> torch.Tensor:
    x_flat = x.view(-1, x.shape[-1])
    output_dtype = x.dtype
    M, _ = weight_packed.shape
    N = original_n  # weight_packed is compressed if max_bits < 8 so we can't rely on its shape for N

    if x_flat.shape[0] > 1:
        if max_bits == 4 and use_marlin:
            y = grinqh_prefill_marlin(
                x_flat,
                weight_packed,
                weight_scale,
                group_size,
                original_n
            )
        else:
            y = grinqh_prefill_gemm_compiled(
                x_flat,
                weight_packed,
                weight_scale,
                group_size,
                original_n,
                max_bits
            )
    else:
        grid = lambda META: (
            triton.cdiv(N // 4, META["BLOCK_N"]),
            triton.cdiv(M // META["NUM_BLOCK_M_LOOPS"], META["BLOCK_M"]),
        )
        y = torch.zeros((1, N * 4), device=x.device, dtype=torch.float32)

        BLOCK_N = GEMV_CONFIG.kwargs["BLOCK_N"]
        BLOCK_M = GEMV_CONFIG.kwargs["BLOCK_M"]
        NUM_WARPS = GEMV_CONFIG.num_warps
        NUM_BLOCK_M_LOOPS = GEMV_CONFIG.kwargs["NUM_BLOCK_M_LOOPS"]
        NUM_STAGES = GEMV_CONFIG.num_stages

        SCALES_BLOCK_M = group_size
        SCALES_BLOCK_N = 1

        grinqh_gemv_uint32_packed[grid](
            Y=y, A=weight_packed, X=x_flat, scales=weight_scale,
            t0=thresholds[0], t1=thresholds[1],
            t2=thresholds[2], t3=thresholds[3],
            N=N, M=M, SCALES_BLOCK_SIZE_M=SCALES_BLOCK_M, SCALES_BLOCK_SIZE_N=SCALES_BLOCK_N, MAX_BITS=max_bits,
            BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,
            NUM_BLOCK_M_LOOPS=NUM_BLOCK_M_LOOPS, num_warps=NUM_WARPS, num_stages=NUM_STAGES
        )

        y = y.to(output_dtype)
    if bias is not None:
        y.add_(bias)

    return y.view(*x.shape[:-1], N * 4)


@grinqh_gemm_or_gemv.register_fake
def _fake(
        x,
        weight_packed,
        weight_scale,
        bias,
        max_bits,
        group_size,
        original_n,
        thresholds,
        use_marlin=False
):
    return torch.empty(x.shape[:-1] + (original_n * 4,), device=x.device, dtype=x.dtype)


# You can use this to inject bash env variables and test in non-eager because otherwise triton autotune blows it up
grinqh_gemv_uint32_packed_fakeautotune = triton.autotune(
    configs=[GEMV_CONFIG],
    key=["N", "M"],
)(grinqh_gemv_uint32_packed)
