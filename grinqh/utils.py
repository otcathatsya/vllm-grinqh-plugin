from enum import Enum
from typing import Union

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton

_UNPACK_SHIFTS = (6, 4, 2, 0)


def _unpack_shifts(device: torch.device) -> torch.Tensor:
    return torch.tensor(_UNPACK_SHIFTS, device=device, dtype=torch.uint8)

# GRINQH writes output in a 16 block layout (to maintain locality) because of 4x4 pack arising from
# the 4 neighbours necessary to fill one uint32 and the 4 zones all their bits are located in.
# Easy way out: reshape beforehand so we don't need to benchmark a strided write
def grinqh_input_preshuffle(weight: torch.Tensor) -> torch.Tensor:
    M, N = weight.shape
    assert N % 16 == 0, "N must be divisible by 16 for this layout."
    w_view = weight.view(M, 16, N // 16)
    w_transposed = w_view.transpose(1, 2)
    return w_transposed.contiguous().view(M, N)


def grinqh_input_postshuffle(weight: torch.Tensor) -> torch.Tensor:
    M, N = weight.shape
    assert N % 16 == 0, "N must be divisible by 16 for this layout."
    w_view = weight.view(M, N // 16, 16)
    w_transposed = w_view.transpose(1, 2)
    return w_transposed.contiguous().view(M, N)


class BitLevel(Enum):
    TWO_BIT = 1
    FOUR_BIT = 2
    SIX_BIT = 3
    EIGHT_BIT = 4


def pack_pyt_uint8(A: torch.Tensor) -> torch.Tensor:
    # apparently transpose messes w the contiguity
    A = A.contiguous()
    M, N = A.shape

    # pad if necessary
    if N % 4 != 0:
        A = torch.nn.functional.pad(A, (0, 4 - (N % 4)))
        N = A.shape[1]

    N_zone = N // 4
    # (M, N, zones, neighbours) where zone is the 2-bit segments for each level
    # and block is the 4 neighbouring values needed to fully pack a byte
    A_view = A.view(M, N_zone, 1, 4)

    # get starting bits for in a packed byte for the unquantized values on a zone level
    shifts = _unpack_shifts(A.device)
    src_shifts = shifts.view(1, 1, 4, 1)
    # if LSB: [0, 2, 4, 6] ^
    # same but the other way round for our goal packed byte; broadcasting here is on neighbour level
    dest_shifts = shifts.view(1, 1, 1, 4)

    # do a (4, 4) grid of all bit section movements
    packed_4d = ((A_view >> src_shifts) & 3) << dest_shifts

    # sum kinda works as a |= here cause non-overlapping
    zones = packed_4d.sum(dim=-1, dtype=torch.uint8)

    # rn our block has all bytes correctly sorted but it needs to be permuted to have all zones together
    zones = zones.permute(0, 2, 1).contiguous()
    # yeet
    return zones.view(M, N)


# TODO: probably remove uint8 versions, no longer used anywhere
@triton.jit
def _pack_kernel_uint8(A_ptr, Out_ptr, stride0, stride1, N_zone, N_out, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_block = tl.program_id(1)

    idx = pid_block * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N_out

    zone_idx = idx // N_zone  # 0..3
    byte_idx = idx - zone_idx * N_zone
    base_col = byte_idx * 4

    offs = base_col[:, None] + tl.arange(0, 4)[None, :]
    a = tl.load(A_ptr + pid_m * stride0 + offs * stride1, mask=mask[:, None], other=0).to(tl.int32)

    # src shifts per zone: [6,4,2,0]; dest shifts per neighbor: [6,4,2,0]
    src_shift = ((3 - zone_idx) * 2).to(tl.int32)
    dest_shift = ((3 - tl.arange(0, 4)) * 2).to(tl.int32)  # replaces [::-1]*2

    packed = ((a >> src_shift[:, None]) & 3) << dest_shift[None, :]
    out_val = tl.sum(packed, axis=1).to(tl.uint8)

    tl.store(Out_ptr + pid_m * N_out + idx, out_val, mask=mask)


@triton.jit
def _pack_kernel_uint32(A_ptr, Out_ptr, stride0, stride1, N_zone, N_out, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_block = tl.program_id(1)

    idx = pid_block * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N_out

    zone_idx = idx // N_zone  # 0..3
    byte_idx = idx - zone_idx * N_zone
    base_col = byte_idx * 4

    offs = base_col[:, None] + tl.arange(0, 4)[None, :]
    a = tl.load(A_ptr + pid_m * stride0 + offs * stride1, mask=mask[:, None], other=0).to(tl.int32)

    # src shifts per zone: [6,4,2,0]; dest shifts per neighbor: [6,4,2,0]
    src_shift = ((3 - zone_idx) * 2).to(tl.int32)
    dest_shift = ((3 - tl.arange(0, 4)) * 2).to(tl.int32)  # replaces [::-1]*2

    packed = ((a >> src_shift[:, None]) & 0b00000011000000110000001100000011) << dest_shift[None, :]
    out_val = tl.sum(packed, axis=1)

    tl.store(Out_ptr + pid_m * N_out + idx, out_val, mask=mask)


def pack_triton(A: torch.Tensor, max_bits: int = 8) -> torch.Tensor:
    """
    A: torch.uint8 CUDA tensor (M, N), contiguous.
    Pads N to multiple of 4 and packs zones MSB-first like the original `pack`.
    Returns shape (M, N_padded).
    """
    assert A.is_cuda
    assert A.dtype == torch.uint8 or A.dtype == torch.uint32, "Triton kernel expects CUDA uint8 or uint32 tensor"
    A = A.contiguous()
    M, N = A.shape

    N_zone = N // 4
    num_zones = max_bits // 2
    N_out = N_zone * num_zones

    out = torch.empty((M, N_out), device=A.device, dtype=A.dtype)

    BLOCK = 256
    grid = (M, triton.cdiv(N_out, BLOCK))
    kernel_fn = _pack_kernel_uint8 if A.dtype == torch.uint8 else _pack_kernel_uint32
    kernel_fn[grid](A, out, A.stride(0), A.stride(1), N_zone, N_out, BLOCK=BLOCK)
    return out


def unpack_pyt(B: torch.Tensor, row: Union[int, torch.Tensor], bit_level: BitLevel, original_n: int) -> torch.Tensor:
    k = bit_level.value
    N_zone = original_n // 4

    if isinstance(row, int):
        row = torch.tensor([row], device=B.device)

    orig_dtype = B.dtype

    if orig_dtype == torch.uint32:
        B = B.view(torch.int32)
        bit_mask = 0x03030303
        sum_dtype = torch.int32
    else:
        B = B.view(torch.uint8)
        bit_mask = 0x03
        sum_dtype = torch.uint8

    packed_zone = B[row, : N_zone * k]
    batch_size = packed_zone.shape[0]
    packed_zone = packed_zone.view(batch_size, k, N_zone)

    shifts = _unpack_shifts(B.device)
    unpacked = (packed_zone.unsqueeze(-1) >> shifts) & bit_mask

    merge_shifts = (torch.arange(k, device=B.device) * 2).view(1, k, 1, 1).flip(1)
    res = (unpacked << merge_shifts).sum(dim=1, dtype=sum_dtype)

    return res.view(batch_size, -1).view(orig_dtype)


@triton.jit
def _unpack_kernel(B_ptr, out_ptr, stride0, N_zone, k, row, BLOCK: tl.constexpr):
    # program covers BLOCK output elements (2-bit values) at a time
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n_total = N_zone * 4  # 4 two-bit values per packed byte
    mask = idx < n_total

    byte_idx = idx // 4  # which packed byte inside each zone
    shift_idx = idx % 4  # which 2-bit slot inside the byte
    # shifts = [6, 4, 2, 0]
    shift = tl.where(shift_idx == 0, 6,
                     tl.where(shift_idx == 1, 4,
                              tl.where(shift_idx == 2, 2, 0))).to(tl.int32)

    acc = tl.full(idx.shape, 0, tl.int32)
    bit_mask = 0x03 if B_ptr.dtype.element_ty == tl.uint8 else 0x03030303

    for level in range(4):
        active = level < k

        ptr = B_ptr + row * stride0 + N_zone * level + byte_idx
        b = tl.load(ptr, mask=mask & active, other=0).to(tl.int32)
        val = (b >> shift) & bit_mask
        merge_shift = 2 * (k - 1 - level)  # same as flip()
        acc += tl.where(active, val << merge_shift, 0)

    tl.store(out_ptr + idx, acc.to(B_ptr.dtype.element_ty), mask=mask)


def unpack_triton(B: torch.Tensor, row: int, bit_level, original_n: int) -> torch.Tensor:
    assert B.is_cuda, "Triton kernel expects CUDA tensor"
    assert B.dtype == torch.uint8 or B.dtype == torch.uint32, "Triton kernel expects uint8 or uint32 tensor"
    B = B.contiguous()
    M, _ = B.shape
    k = int(bit_level.value)
    N_zone = original_n // 4
    out = torch.empty((original_n,), device=B.device, dtype=B.dtype)

    BLOCK = 256
    grid = (triton.cdiv(N_zone * 4, BLOCK),)
    _unpack_kernel[grid](
        B, out, B.stride(0), N_zone, k, row, BLOCK=BLOCK,
    )
    return out
