"""Safety checks for merge operations on runtime-quantized ConvRot bases."""


def raise_if_convrot_active(network, *, context: str) -> None:
    loras = list(getattr(network, "text_encoder_loras", []) or []) + list(
        getattr(network, "unet_loras", []) or []
    )
    if any(
        getattr(lora, "_convrot_mode", None) is not None
        or getattr(lora, "_convrot_quantized_weight", None) is not None
        for lora in loras
    ):
        raise RuntimeError(
            f"{context}: refused for ConvRot base path. Merge/fuse assumes "
            "high-precision writable Linear.weight. Disable ConvRot or use a "
            "dedicated dequantize-and-fold path."
        )
