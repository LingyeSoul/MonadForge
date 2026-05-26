"""Text encoding and preparation for Anima inference."""

import gc
import logging
from typing import Optional, Tuple, Any, Dict

import torch

from library.anima import models as anima_models, text_strategies
from library.runtime.device import clean_memory_on_device
from library.inference.models import load_text_encoder

logger = logging.getLogger(__name__)

# Anima's DiT expects a fixed-length cross-attention context. The pretrained
# model treats zero-padded positions as attention sinks in the cross-attention
# softmax — trimming to actual text length produces black images. All sites
# that prepare crossattn embeds (training, inference, CFG-uncond, DCW
# trajectory replay, DirectEdit, distillation) must pad to this exact length.
MAX_CROSSATTN_TOKENS = 512


def process_escape(text: str) -> str:
    """Process escape sequences in text."""
    return text.encode("utf-8").decode("unicode_escape")


def ensure_text_strategies(
    text_encoder_path: Optional[str],
    max_length: int = MAX_CROSSATTN_TOKENS,
) -> None:
    """Idempotently install the global tokenize/encode strategy singletons.

    Anima encodes prompts through two *process-global* singletons —
    ``TokenizeStrategy`` and ``TextEncodingStrategy`` (the strategy pattern in
    ``library/anima/strategy.py``). The CLI sets them in ``inference.main``; an
    embedder calling ``generate()`` / ``prepare_text_inputs()`` directly must too,
    or ``get_strategy()`` returns ``None`` and the first ``tokenize()`` call dies
    with a cryptic ``'NoneType' object has no attribute 'tokenize'``.

    This installs whichever singleton is still unset, building the tokenizer from
    ``text_encoder_path``. It is a **no-op when both are already installed**, so it
    composes with the CLI path (and is safe to call on every generation). If a
    strategy is missing *and* no path is available to build it, it raises a clear
    ``ValueError`` instead of failing later deep in the encode call.
    """
    from library.anima import strategy as strategy_anima

    need_tok = text_strategies.TokenizeStrategy.get_strategy() is None
    need_enc = text_strategies.TextEncodingStrategy.get_strategy() is None

    if need_tok:
        if not text_encoder_path:
            raise ValueError(
                "Text strategies are not initialized and no text-encoder path was "
                "provided to initialize them. Either set them yourself "
                "(text_strategies.TokenizeStrategy.set_strategy(...) + "
                "TextEncodingStrategy.set_strategy(...)) or pass a text-encoder path."
            )
        text_strategies.TokenizeStrategy.set_strategy(
            strategy_anima.AnimaTokenizeStrategy(
                qwen3_path=text_encoder_path,
                t5_tokenizer_path=None,
                qwen3_max_length=max_length,
                t5_max_length=max_length,
            )
        )
    if need_enc:
        text_strategies.TextEncodingStrategy.set_strategy(
            strategy_anima.AnimaTextEncodingStrategy()
        )


def prepare_text_inputs(
    args,
    device: torch.device,
    anima: anima_models.Anima,
    shared_models: Optional[Dict] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare text-related inputs for T2I: LLM encoding. Anima model is also needed for preprocessing.

    The tokenize/encode strategy singletons are lazily installed from
    ``args.text_encoder`` via ``ensure_text_strategies`` if a caller (an embedder
    driving ``generate()`` directly) hasn't set them — a no-op on the CLI path,
    which sets them in ``inference.main``.
    """

    # Install the global tokenize/encode strategies if the caller didn't (the
    # CLI does; a bare generate() embedder may not). No-op when already set.
    ensure_text_strategies(getattr(args, "text_encoder", None))

    # load text encoder: conds_cache holds cached encodings for prompts without padding
    conds_cache = {}
    text_encoder_device = torch.device("cpu") if args.text_encoder_cpu else device
    if shared_models is not None:
        text_encoder = shared_models.get("text_encoder")

        if "conds_cache" in shared_models:  # Use shared cache if available
            conds_cache = shared_models["conds_cache"]

        # text_encoder is on device (batched inference) or CPU (interactive inference)
    else:  # Load if not in shared_models
        text_encoder_dtype = torch.bfloat16  # Default dtype for Text Encoder
        text_encoder = load_text_encoder(
            args, dtype=text_encoder_dtype, device=text_encoder_device
        )
        text_encoder.eval()

    # Store original devices to move back later if they were shared.
    text_encoder_original_device = text_encoder.device if text_encoder else None

    if not text_encoder:
        raise ValueError("Text encoder is not loaded properly.")

    model_is_moved = False

    def move_models_to_device_if_needed():
        nonlocal model_is_moved
        nonlocal shared_models

        if model_is_moved:
            return
        model_is_moved = True

        logger.info(f"Moving Text Encoder to appropriate device: {text_encoder_device}")
        text_encoder.to(text_encoder_device)

    logger.info("Encoding prompt with Text Encoder")

    prompt = process_escape(args.prompt)
    cache_key = prompt
    if cache_key in conds_cache:
        embed = conds_cache[cache_key]
    else:
        move_models_to_device_if_needed()

        tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
        encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()

        with torch.no_grad():
            tokens = tokenize_strategy.tokenize(prompt)
            embed = encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens
            )
            crossattn_emb, _ = anima._preprocess_text_embeds(
                source_hidden_states=embed[0].to(anima.device),
                target_input_ids=embed[2].to(anima.device),
                target_attention_mask=embed[3].to(anima.device),
                source_attention_mask=embed[1].to(anima.device),
            )
            crossattn_emb[~embed[3].bool()] = 0
            if crossattn_emb.shape[1] < MAX_CROSSATTN_TOKENS:
                crossattn_emb = torch.nn.functional.pad(
                    crossattn_emb,
                    (0, 0, 0, MAX_CROSSATTN_TOKENS - crossattn_emb.shape[1]),
                )
            embed[0] = crossattn_emb
        embed[0] = embed[0].cpu()

        conds_cache[cache_key] = embed

    negative_prompt = process_escape(args.negative_prompt)
    cache_key = negative_prompt
    if cache_key in conds_cache:
        negative_embed = conds_cache[cache_key]
    else:
        move_models_to_device_if_needed()

        tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
        encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()

        with torch.no_grad():
            tokens = tokenize_strategy.tokenize(negative_prompt)
            negative_embed = encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens
            )
            crossattn_emb, _ = anima._preprocess_text_embeds(
                source_hidden_states=negative_embed[0].to(anima.device),
                target_input_ids=negative_embed[2].to(anima.device),
                target_attention_mask=negative_embed[3].to(anima.device),
                source_attention_mask=negative_embed[1].to(anima.device),
            )
            crossattn_emb[~negative_embed[3].bool()] = 0
            if crossattn_emb.shape[1] < MAX_CROSSATTN_TOKENS:
                crossattn_emb = torch.nn.functional.pad(
                    crossattn_emb,
                    (0, 0, 0, MAX_CROSSATTN_TOKENS - crossattn_emb.shape[1]),
                )
            negative_embed[0] = crossattn_emb
        negative_embed[0] = negative_embed[0].cpu()

        conds_cache[cache_key] = negative_embed

    if not (shared_models and "text_encoder" in shared_models):  # if loaded locally
        del text_encoder
        gc.collect()
    else:  # if shared, move back to original device (likely CPU)
        if text_encoder:
            text_encoder.to(text_encoder_original_device)

    clean_memory_on_device(device)

    arg_c = {"embed": embed, "prompt": prompt}
    arg_null = {"embed": negative_embed, "prompt": negative_prompt}

    return arg_c, arg_null
