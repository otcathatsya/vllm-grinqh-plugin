import pytest
import torch

from grinqh.kernel import (
    grinqh_gemm_or_gemv,
    grinqh_prefill_gemm_compiled,
    grinqh_prefill_marlin,
)
from grinqh.utils import (
    BitLevel,
    grinqh_input_preshuffle,
    pack_triton,
    unpack_pyt,
    unpack_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def _fresh_state():
    torch.manual_seed(0)
    torch._dynamo.reset()


def get_masked_weight_ref(
        weight: torch.Tensor,
        x: torch.Tensor,
        thresholds: list[float],
        max_bits: int
) -> torch.Tensor:
    """given (in, out) int8 weights + (1, in) activations,
    this mimics the sparse GRINQH kernel results via dense PyTorch ops"""
    shift_amt = 8 - max_bits
    x_abs = x.abs().float().flatten()  # kernel compares thresholds in fp32
    w_u8 = (weight << shift_amt).view(torch.uint8)

    row_mask = torch.zeros_like(x_abs, dtype=torch.uint8)
    bias = torch.zeros_like(x_abs, dtype=torch.float32)
    bias_base = 1 << (max_bits - 1)
    prev_bias = 0

    for zone in range(max_bits // 2):
        active = x_abs > thresholds[zone]
        row_mask |= active.to(torch.uint8) * (0xC0 >> 2 * zone)

        cur_bias = bias_base >> (2 * zone + 2)
        bias += active * (cur_bias - prev_bias)
        prev_bias = cur_bias

    weight_masked = (w_u8 & row_mask[:, None]).view(torch.int8).float()

    return weight_masked / 2 ** shift_amt + bias[:, None]


@pytest.mark.parametrize("bit_level", [BitLevel.TWO_BIT, BitLevel.FOUR_BIT, BitLevel.SIX_BIT, BitLevel.EIGHT_BIT])
@pytest.mark.parametrize("backend", ["pytorch", "triton"])
def test_pack_unpack(bit_level, backend):
    in_features = 384
    out_features = 320

    # usually (out, in) but we skip the transpose and just assume GRINQH input-major format
    A = torch.randint(0, 256, (in_features, out_features), device='cuda', dtype=torch.uint8)

    if backend == "pytorch":
        pack_fn, unpack_fn = pack_triton, unpack_pyt
    else:
        pack_fn, unpack_fn = pack_triton, unpack_triton

    A_in = A.view(torch.uint32)

    B = pack_fn(A_in, max_bits=bit_level.value * 2)
    # verify compressed storage
    assert B.numel() * B.element_size() == A.numel() * bit_level.value * 2 // 8

    A_rec = torch.zeros_like(A)
    for i in range(in_features):
        A_rec[i] = unpack_fn(B, i, bit_level, original_n=A_in.shape[1]).view(torch.uint8)

    if backend == "pytorch":
        A_rec_vec = unpack_fn(B, torch.arange(in_features), bit_level=bit_level, original_n=A_in.shape[1]).view(
            torch.uint8)
        torch.testing.assert_close(A_rec, A_rec_vec, rtol=0, atol=0)

    expected = A >> (8 - bit_level.value * 2)
    torch.testing.assert_close(A_rec, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("max_bits", [2, 4, 6, 8])
def test_dense_prefill_matches_reference(max_bits, dtype):
    num_tokens = 16
    in_features = 384
    out_features = 320
    group_size = 128
    original_n = out_features // 4

    X = torch.randn((num_tokens, in_features), dtype=dtype, device='cuda')
    limit = (1 << (max_bits - 1)) - 1
    W_8bit = torch.randint(-limit - 1, limit + 1, (in_features, out_features), dtype=torch.int8, device='cuda')
    scales = torch.randn((in_features // group_size, out_features), dtype=dtype, device='cuda')

    # scale for W[k, n] is scales[k // group_size, n]
    W_ref_scaled = W_8bit.to(dtype) * scales.repeat_interleave(group_size, dim=0)
    Y_ref = X @ W_ref_scaled

    shift_amt = 8 - max_bits
    shuffled = grinqh_input_preshuffle(W_8bit)
    shifted = shuffled << shift_amt

    weight_packed = pack_triton(shifted.view(torch.uint32), max_bits=max_bits)

    Y_grinqh = grinqh_prefill_gemm_compiled(
        x_flat=X,
        weight_packed=weight_packed,
        weight_scale=scales,
        group_size=group_size,
        original_n=original_n,
        max_bits=max_bits
    )

    torch.testing.assert_close(Y_grinqh, Y_ref)


@pytest.mark.skipif(not hasattr(torch.ops._C, "marlin_gemm"), reason="MARLIN OP not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_4bit_marlin_prefill_matches_reference(dtype):
    batch_size = 65
    in_features = 384
    out_features = 320
    group_size = 128
    original_n = out_features // 4

    x = (torch.randn((batch_size, in_features), dtype=torch.float32, device="cuda") * 0.1).to(dtype)
    weight = torch.randint(-8, 8, (in_features, out_features), dtype=torch.int8, device="cuda")
    scales = (torch.rand((in_features // group_size, out_features), dtype=torch.float32, device="cuda") * 0.1).to(dtype)

    weight_ref = (
        weight.view(in_features // group_size, group_size, out_features)
        .to(dtype)
        .mul(scales.unsqueeze(1))
        .view(in_features, out_features)
    )
    expected = x @ weight_ref

    shuffled = grinqh_input_preshuffle(weight)
    packed = pack_triton((shuffled << 4).view(torch.uint32), max_bits=4)

    actual = grinqh_prefill_marlin(x, packed, scales, group_size, original_n)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("max_bits", [2, 4, 6, 8])
def test_dense_gemv_matches_reference(max_bits, dtype):
    in_features = 256
    out_features = 256
    group_size = 128
    original_n = out_features // 4

    X = torch.randn((1, in_features), dtype=dtype, device='cuda')
    limit = (1 << (max_bits - 1)) - 1
    W_8bit = torch.randint(-limit - 1, limit + 1, (in_features, out_features), dtype=torch.int8, device='cuda')
    scales = torch.randn((in_features // group_size, out_features), dtype=dtype, device='cuda')

    # accumulate in fp32 to match our Triton kernel
    Y_ref = X.float() @ (W_8bit.float() * scales.float().repeat_interleave(group_size, 0))

    # isolate top bits
    shift_amt = 8 - max_bits
    shuffled = grinqh_input_preshuffle(W_8bit)
    shifted = shuffled << shift_amt
    weight_packed = pack_triton(shifted.view(torch.uint32), max_bits=max_bits)

    # thresholds of -1 to compute everything in highest precision
    Y_grinqh = grinqh_gemm_or_gemv(
        x=X,
        weight_packed=weight_packed,
        weight_scale=scales,
        bias=None,
        max_bits=max_bits,
        group_size=group_size,
        original_n=original_n,
        thresholds=[-1.0, -1.0, -1.0, -1.0]
    )

    torch.testing.assert_close(Y_grinqh, Y_ref.to(dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("max_bits", [2, 4, 6, 8])
@pytest.mark.parametrize("quantiles", [
    [0.2, 0.4, 0.6, 0.8],
    [0.5] * 4,
    [0.1, 0.2, 0.3, 0.4],
    [0.6, 0.7, 0.8, 0.9],
    [1.0] * 4,
], ids=["spread", "all_or_nothing", "mild", "aggressive", "skip_all"])
def test_gemv_thresholds_match_reference(max_bits, dtype, quantiles):
    in_features = 256
    out_features = 256
    group_size = 128
    original_n = out_features // 4

    X = torch.randn((1, in_features), dtype=dtype, device='cuda')
    limit = (1 << (max_bits - 1)) - 1
    W_8bit = torch.randint(-limit - 1, limit + 1, (in_features, out_features), dtype=torch.int8, device='cuda')
    scales = torch.randn((in_features // group_size, out_features), dtype=dtype, device='cuda')

    # define thresholds w np.quantile to match and activate each layer:
    num_zones = max_bits // 2
    q = torch.tensor(quantiles, device=X.device)

    thresholds = torch.quantile(X.abs().float(), q)[:max_bits // 2].tolist()
    thresholds = (thresholds + [0.0] * (4 - num_zones))

    masked_weights = get_masked_weight_ref(W_8bit, X, thresholds, max_bits)
    Y_ref = X.float() @ (masked_weights.float() * scales.float().repeat_interleave(group_size, 0))

    # isolate top bits
    shift_amt = 8 - max_bits
    shuffled = grinqh_input_preshuffle(W_8bit)
    shifted = shuffled << shift_amt
    weight_packed = pack_triton(shifted.view(torch.uint32), max_bits=max_bits)

    Y_grinqh = grinqh_gemm_or_gemv(
        x=X,
        weight_packed=weight_packed,
        weight_scale=scales,
        bias=None,
        max_bits=max_bits,
        group_size=group_size,
        original_n=original_n,
        thresholds=thresholds
    )

    torch.testing.assert_close(Y_grinqh, Y_ref.to(dtype))

