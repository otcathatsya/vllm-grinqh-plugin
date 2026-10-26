import json
import os
from typing import List, Optional

import torch
from compressed_tensors import unpack_from_int32
from torch.nn.parameter import Parameter
from vllm.logger import init_logger
from vllm.model_executor import PackedvLLMParameter
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import (
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.marlin_utils import check_marlin_supports_shape, \
    check_marlin_supported
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .kernel import grinqh_gemm_or_gemv, grinqh_prefill_gemm_compiled  # noqa: F401  (registers torch.ops.grinqh)
from .utils import grinqh_input_preshuffle, pack_triton

logger = init_logger("vllm.grinqh")
_SHARD_IDS = {"q": 0, "k": 1, "v": 2, "gate": 0, "up": 1}


def _resolve_shard_id(shard_id: int | str | None) -> int:
    # None for non-merged layers (o_proj, down_proj, ...)
    if shard_id is None:
        return 0
    if isinstance(shard_id, int):
        return shard_id
    if shard_id in _SHARD_IDS:
        return _SHARD_IDS[shard_id]
    raise ValueError(f"GRINQH: unsupported shard_id {shard_id!r}")


@register_quantization_config("grinqh")
class GRINQHConfig(QuantizationConfig):
    def __init__(
            self,
            thresholds: dict[str, list[float]],
            max_bits: int = 8,
            group_size: int = 128,
    ):
        super().__init__()
        self.thresholds = thresholds
        self.num_bits = max_bits
        self.group_size = group_size

    def get_name(self) -> str:
        return "GRINQH"

    def get_supported_act_dtypes(self) -> list:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if user_quant == "grinqh":
            return "grinqh"
        return None

    @classmethod
    def from_config(cls, config: dict) -> "GRINQHConfig":
        group_0_weights = config.get("config_groups", {}).get("group_0", {}).get("weights", {})
        grinqh_cfg = config.get("grinqh", {})
        num_bits = grinqh_cfg.get("max_bits", group_0_weights.get("num_bits", 8))
        group_size = group_0_weights.get("group_size", 128)

        # Override only if you wish to use an external calibration file instead of the model's quant. config's
        json_path = os.environ.get("GRINQH_JSON_PATH")
        if json_path is not None:
            logger.warning(
                "GRINQH: GRINQH_JSON_PATH is set, overriding checkpoint "
                "thresholds with %s", json_path)
            with open(json_path) as f:
                thresholds = json.load(f)["thresholds"]
        elif "thresholds" in grinqh_cfg:
            thresholds = grinqh_cfg["thresholds"]
        else:
            raise ValueError(
                "GRINQH: checkpoint quantization_config has no "
                "'grinqh.thresholds'. Calibrate the model first or set "
                "GRINQH_JSON_PATH.")

        return cls(thresholds=thresholds, max_bits=num_bits, group_size=group_size)

    def get_quant_method(
            self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            t_vals = self.thresholds.get(prefix)
            if t_vals is None:
                raise ValueError(
                    f"GRINQH: no thresholds for layer '{prefix}'. Thresholds "
                    "must be keyed by vLLM module names (e.g. qkv_proj).")
            # pad to 4 in lower <8 bit settings
            t_vals = (t_vals + [0.0] * 4)[:4]
            return GRINQHLinearMethod(self, t_vals, layer_name=prefix)
        return None


class GRINQHLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: GRINQHConfig, layer_thresholds: List[float] = None, layer_name: str = "unknown"):
        super().__init__()
        self.quant_config = quant_config
        self.layer_thresholds = layer_thresholds
        self.layer_name = layer_name
        self.PACK_FACTOR = 4

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        if layer.tp_size > 1:
            raise NotImplementedError("GRINQH does not support tensor parallelism (tp_size > 1) as of now.")
        output_size_per_partition = sum(output_partition_sizes)
        SCALES_BLOCK_M = self.quant_config.group_size
        SCALES_BLOCK_N = 1
        is_6bit = self.quant_config.num_bits == 6
        param_name = "weight" if is_6bit else "weight_packed"

        def grinqh_weight_loader(param, loaded_weight, shard_id=None):
            current_shard = _resolve_shard_id(shard_id)
            if param is scales:
                s = loaded_weight.to(param.device).T.contiguous()
                if s.shape == param.shape:
                    param.data.copy_(s)
                else:
                    start_unpacked = sum(output_partition_sizes[:current_shard])
                    current_unpacked_len = output_partition_sizes[current_shard]

                    start_col = start_unpacked // SCALES_BLOCK_N
                    end_col = start_col + current_unpacked_len // SCALES_BLOCK_N
                    param.data[:, start_col:end_col].copy_(s)
                return

            if is_6bit:
                # already in 8-bit format
                w = loaded_weight.to(param.device)
            else:
                # assume pre-packed format
                w = unpack_from_int32(
                    loaded_weight.to(param.device),
                    self.quant_config.num_bits,
                    param.T.shape,
                )

            w = w.T.contiguous().to(torch.int8)

            if len(output_partition_sizes) > 1:
                start_unpacked = sum(output_partition_sizes[:current_shard])
                start_packed = start_unpacked // self.PACK_FACTOR
                size_packed = output_partition_sizes[current_shard] // self.PACK_FACTOR
                param.data[:, start_packed:start_packed + size_packed].copy_(w.view(torch.int32))
            else:
                param.data[:, :].copy_(w.view(torch.int32))

        weight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.PACK_FACTOR,
                dtype=torch.int32,
            ),
            input_dim=0, output_dim=1,
            packed_dim=1, packed_factor=self.PACK_FACTOR,
            weight_loader=grinqh_weight_loader,
        )
        layer.register_parameter(param_name, weight)
        # alias
        layer.grinqh_weight_param_name = param_name

        scale_rows = (input_size_per_partition + SCALES_BLOCK_M - 1) // SCALES_BLOCK_M
        scale_cols = (output_size_per_partition + SCALES_BLOCK_N - 1) // SCALES_BLOCK_N
        scales = Parameter(
            torch.empty(scale_rows, scale_cols, device="cuda", dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("weight_scale", scales)
        set_weight_attrs(scales, {"weight_loader": grinqh_weight_loader})

        weight_shape = Parameter(
            torch.empty(2, device="cuda", dtype=torch.int64),
            requires_grad=False,
        )
        set_weight_attrs(weight_shape, {"weight_loader": lambda p, w, shard_id=None: None})
        layer.register_parameter("weight_shape", weight_shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        param = getattr(layer, layer.grinqh_weight_param_name)
        shuffled = grinqh_input_preshuffle(param.data.view(torch.int8))

        shift_amt = 8 - self.quant_config.num_bits
        shifted = shuffled << shift_amt

        layer.original_n = shifted.shape[1] // self.PACK_FACTOR
        packed_weight = pack_triton(
            shifted.view(torch.uint32), max_bits=self.quant_config.num_bits)
        replace_parameter(layer, layer.grinqh_weight_param_name, packed_weight)

        layer.grinqh_use_marlin = False
        if self.quant_config.num_bits == 4 and current_platform.is_cuda():
            size_k = shifted.shape[0]
            size_n = shifted.shape[1]
            shape_supported, _ = check_marlin_supports_shape(
                output_size_per_partition=size_n,
                input_size_per_partition=size_k,
                input_size=size_k,
                group_size=self.quant_config.group_size,
            )
            ops_available = hasattr(torch.ops._C, "marlin_gemm")
            if (ops_available and shape_supported and check_marlin_supported(
                    scalar_types.uint4b8, self.quant_config.group_size)):
                layer.grinqh_use_marlin = True
                logger.info_once(
                    "GRINQH max-4 prefill is using transient Marlin W4A16 "
                    "weights."
                )
            else:
                logger.warning_once(
                    "GRINQH max-4 prefill is falling back to dense GEMM for "
                    "layers unsupported by Marlin."
                )

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.ops.grinqh.gemm_or_gemv(
            x,
            getattr(layer, layer.grinqh_weight_param_name),
            layer.weight_scale,
            bias,
            self.quant_config.num_bits,
            self.quant_config.group_size,
            layer.original_n,
            self.layer_thresholds,
            layer.grinqh_use_marlin,
        )
