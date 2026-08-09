"""ConvRot runtime package: group RHT + W8A16/W8A8 training base path."""

from library.runtime.convrot.apply import (
    ConvRotApplyResult,
    ConvRotLoRABaseForwardPatch,
    apply_convrot_to_lora_network,
)
from library.runtime.convrot.checks import (
    assert_convrot_block_swap_mutex,
    warn_convrot_blocks_to_swap,
    convrot_mode_from_base_compute,
    normalize_base_compute,
)
from library.runtime.convrot.free_base import free_linear_weight_storage, is_base_weight_freed
from library.runtime.convrot.fused import fused_w8a16_forward, fused_w8a8_forward
from library.runtime.convrot.gemm import int8_mm_scaled, w8a8_int_linear
from library.runtime.convrot.linear_w8a16 import ConvRotW8A16Linear
from library.runtime.convrot.linear_w8a8 import ConvRotW8A8Linear
from library.runtime.convrot.metadata import (
    stamp_convrot_metadata,
    raise_if_merge_with_convrot,
    metadata_indicates_convrot,
)
from library.runtime.convrot.prequant import (
    FORMAT_V1 as PREQUANT_FORMAT_V1,
    PrequantCheckpoint,
    build_prequant_layers_from_modules,
    load_prequant_checkpoint,
    save_prequant_checkpoint,
)
from library.runtime.convrot.rht import group_fwht

__all__ = [
    "ConvRotApplyResult",
    "ConvRotLoRABaseForwardPatch",
    "ConvRotW8A16Linear",
    "ConvRotW8A8Linear",
    "PREQUANT_FORMAT_V1",
    "PrequantCheckpoint",
    "apply_convrot_to_lora_network",
    "assert_convrot_block_swap_mutex",
    "build_prequant_layers_from_modules",
    "convrot_mode_from_base_compute",
    "free_linear_weight_storage",
    "fused_w8a16_forward",
    "fused_w8a8_forward",
    "group_fwht",
    "int8_mm_scaled",
    "is_base_weight_freed",
    "load_prequant_checkpoint",
    "metadata_indicates_convrot",
    "normalize_base_compute",
    "raise_if_merge_with_convrot",
    "save_prequant_checkpoint",
    "stamp_convrot_metadata",
    "w8a8_int_linear",
    "warn_convrot_blocks_to_swap",
]
