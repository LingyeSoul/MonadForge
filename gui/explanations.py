"""Bilingual help text for config fields and LoRA variant descriptions."""

from __future__ import annotations

from gui.i18n import current_language

# ── Per-field tooltips ─────────────────────────────────────────
# Keys match config field names. Each maps to {lang: description}.

FIELD_HELP: dict[str, dict[str, str]] = {
    # Architecture
    "network_dim": {
        "en": "LoRA rank (dimension of low-rank matrices). Higher = more expressive but more VRAM. Typical: 8\u201364.",
        "ko": "LoRA 랭크 (저랭크 행렬의 차원). 높을수록 표현력이 좋지만 VRAM 사용량 증가. 일반적: 8\u201364.",
    },
    "network_alpha": {
        "en": "LoRA scaling factor. Effective scale = alpha / dim. When alpha == dim, scale is 1.0. Lower alpha = more conservative updates.",
        "ko": "LoRA 스케일링 계수. 실효 스케일 = alpha / dim. alpha == dim이면 1.0. 낮을수록 보수적 업데이트.",
    },
    "network_module": {
        "en": "Python module path for the LoRA network implementation.",
        "ko": "LoRA 네트워크 구현의 Python 모듈 경로.",
    },
    "use_timestep_mask": {
        "en": "Enable T-LoRA: effective rank varies with denoising timestep via power-law schedule. Full rank at high noise, reduced at low noise.",
        "ko": "T-LoRA 활성화: 디노이징 타임스텝에 따라 유효 랭크 변동. 높은 노이즈에서 전체 랭크, 낮은 노이즈에서 축소.",
    },
    "use_ortho": {
        "en": "Enable OrthoLoRA: SVD-based orthogonal parameterization of the update matrix (linear layers only). Regularizes toward structured updates; saved as plain LoRA via thin SVD at checkpoint time.",
        "ko": "OrthoLoRA 활성화: 업데이트 행렬의 SVD 기반 직교 파라미터화 (선형 레이어 전용). 구조화된 업데이트로 정규화되며, 저장 시 thin SVD로 일반 LoRA로 변환.",
    },
    "use_hydra": {
        "en": "Enable HydraLoRA: MoE-style multi-head routing with shared lora_down and per-expert lora_up heads. Produces a *_moe.safetensors sibling for router-live inference. Requires cache_llm_adapter_outputs=true.",
        "ko": "HydraLoRA 활성화: 공유 lora_down + 전문가별 lora_up 헤드를 가진 MoE 스타일 멀티헤드 라우팅. 라우터-라이브 추론용 *_moe.safetensors 동반 파일 생성. cache_llm_adapter_outputs=true 필요.",
    },
    "num_experts": {
        "en": "HydraLoRA expert count. More experts = more capacity but more VRAM and slower training. Typical: 2–8.",
        "ko": "HydraLoRA 전문가 수. 많을수록 표현력 증가하지만 VRAM 사용량 증가 및 학습 속도 감소. 일반적: 2–8.",
    },
    "balance_loss_weight": {
        "en": "HydraLoRA load-balancing loss weight. Discourages router collapse onto a single expert. Typical: 0.01.",
        "ko": "HydraLoRA 부하 균형 손실 가중치. 라우터가 단일 전문가로 붕괴되는 것을 방지. 일반적: 0.01.",
    },
    "balance_loss_warmup_ratio": {
        "en": "Fraction of training steps to hold the balance loss at 0 before activating it. Lets the router specialize first, then switches the penalty on to stop further collapse of a diverged router. 0.0 disables the warmup. Typical: 0.3–0.5.",
        "ko": "밸런스 손실을 0으로 유지하는 학습 스텝 비율. 먼저 라우터가 전문화되도록 한 뒤 페널티를 활성화해 분화된 라우터의 추가 붕괴를 방지. 0.0 = 비활성화. 일반적: 0.3–0.5.",
    },
    "add_reft": {
        "en": "Enable ReFT: block-level residual-stream intervention (Wu et al. 2024). Adds R^T·(ΔW·h + b)·scale to each selected DiT block's output. Composes with any LoRA variant.",
        "ko": "ReFT 활성화: 블록 수준 잔차 스트림 개입 (Wu et al. 2024). 선택된 DiT 블록 출력에 R^T·(ΔW·h + b)·scale 추가. 모든 LoRA 변형과 함께 사용 가능.",
    },
    "reft_dim": {
        "en": "ReFT intervention rank — dimension of R and ΔW in each ReFTModule. Typical: 32–64.",
        "ko": "ReFT 개입 랭크 — 각 ReFTModule의 R 및 ΔW 차원. 일반적: 32–64.",
    },
    "reft_alpha": {
        "en": "ReFT scaling factor (effective scale = alpha / dim). Typical: same as reft_dim.",
        "ko": "ReFT 스케일링 계수 (실효 스케일 = alpha / dim). 일반적: reft_dim과 동일.",
    },
    "reft_layers": {
        "en": "Which DiT blocks receive ReFT modules. 'all', 'last_8', 'first_4', 'stride_2', or comma-separated indices like '3,7,11,15'.",
        "ko": "ReFT 모듈이 적용될 DiT 블록. 'all', 'last_8', 'first_4', 'stride_2', 또는 '3,7,11,15'와 같은 쉼표 구분 인덱스.",
    },
    "use_sigma_router": {
        "en": "Add a tiny sinusoidal(σ)→E bias MLP to each HydraLoRA router, letting expert routing vary with denoising timestep. Zero-init at final layer → step-0 identical to base HydraLoRA.",
        "ko": "각 HydraLoRA 라우터에 sinusoidal(σ)→E 바이어스 MLP 추가하여 타임스텝에 따라 전문가 라우팅 변동. 최종 레이어 zero-init → 초기에는 기본 HydraLoRA와 동일.",
    },
    "sigma_feature_dim": {
        "en": "Sinusoidal σ feature dimension fed into the σ-router bias MLP. Typical: 128.",
        "ko": "σ 라우터 바이어스 MLP에 입력되는 sinusoidal σ 특징 차원. 일반적: 128.",
    },
    "sigma_hidden_dim": {
        "en": "σ-router bias MLP hidden dimension. Typical: 128.",
        "ko": "σ 라우터 바이어스 MLP 히든 차원. 일반적: 128.",
    },
    "sigma_router_layers": {
        "en": "Regex over layer names — only matching layers get a σ-conditional router branch. Typical: limit to cross_attn.q_proj and self_attn.qkv_proj where σ-signal lives.",
        "ko": "레이어 이름에 대한 정규식 — 일치하는 레이어만 σ-조건부 라우터 분기 추가. 일반적: σ 신호가 있는 cross_attn.q_proj 및 self_attn.qkv_proj로 제한.",
    },
    "per_bucket_balance_weight": {
        "en": "Extra per-σ-bucket load-balance penalty, scaled by balance_loss_weight. Encourages routing diversity within each timestep bucket. Typical: 0.3.",
        "ko": "σ 버킷별 추가 부하 균형 페널티, balance_loss_weight로 스케일. 각 타임스텝 버킷 내 라우팅 다양성 유도. 일반적: 0.3.",
    },
    "num_sigma_buckets": {
        "en": "Number of timestep buckets used for per-bucket balance accounting. Typical: 3 (low / mid / high noise).",
        "ko": "버킷별 균형 계산에 사용되는 타임스텝 버킷 수. 일반적: 3 (저/중/고 노이즈).",
    },
    "specialize_experts_by_sigma_buckets": {
        "en": "Hard-partition the expert pool into σ-bands: each timestep bucket only routes to its assigned experts. Forces specialization on top of the soft σ-router bias. Pairs with sigma_bucket_boundaries.",
        "ko": "전문가 풀을 σ-밴드로 하드 분할: 각 타임스텝 버킷은 할당된 전문가만 사용. 소프트 σ-라우터 바이어스 위에 강제 특화 부여. sigma_bucket_boundaries와 함께 사용.",
    },
    "sigma_bucket_boundaries": {
        "en": "Custom σ-bucket edges, length = num_sigma_buckets + 1, monotone 0.0 → 1.0. Defaults to uniform linspace(0, 1, N+1) when omitted. Example: [0.0, 0.5, 0.8, 1.0].",
        "ko": "사용자 지정 σ-버킷 경계, 길이 = num_sigma_buckets + 1, 0.0 → 1.0 단조 증가. 생략 시 uniform linspace(0, 1, N+1) 사용. 예: [0.0, 0.5, 0.8, 1.0].",
    },
    "network_args": {
        "en": "Extra kwargs passed to the network module. For postfix: list of 'key=value' strings (e.g., 'mode=cond-timestep', 'splice_position=end_of_sequence', 'cond_hidden_dim=256'). Pick a Variant to auto-fill.",
        "ko": "네트워크 모듈에 전달되는 추가 kwargs. postfix의 경우 'key=value' 문자열 리스트 (예: 'mode=cond-timestep', 'splice_position=end_of_sequence', 'cond_hidden_dim=256'). Variant 선택으로 자동 채우기 가능.",
    },
    "min_rank": {
        "en": "Minimum active rank when T-LoRA timestep masking is enabled. At the lowest-noise timesteps, rank drops to this value.",
        "ko": "T-LoRA 타임스텝 마스킹 사용 시 최소 활성 랭크. 가장 낮은 노이즈에서 이 값까지 감소.",
    },
    "alpha_rank_scale": {
        "en": "Scale alpha proportionally when T-LoRA reduces rank, keeping effective learning rate stable across timesteps.",
        "ko": "T-LoRA가 랭크를 줄일 때 alpha를 비례적으로 조정하여 타임스텝별 실효 학습률 유지.",
    },
    "network_train_unet_only": {
        "en": "Train only the DiT (U-Net). Text encoder weights are frozen. Recommended for most LoRA training.",
        "ko": "DiT(U-Net)만 학습. 텍스트 인코더 가중치는 동결. 대부분의 LoRA 학습에 권장.",
    },
    "network_weights": {
        "en": "Path to a pre-trained adapter checkpoint to warm-start from. Leave empty for plain LoRA training.",
        "ko": "워밍업으로 사용할 사전 학습 어댑터 체크포인트 경로. 일반 LoRA 학습 시에는 비워두세요.",
    },
    "dim_from_weights": {
        "en": "Read network_dim from the warm-start checkpoint instead of the form value. Set together with network_weights so rank matches the warm-start LoRA.",
        "ko": "network_dim을 폼 값 대신 워밍업 체크포인트에서 읽기. network_weights와 함께 설정하여 랭크를 워밍업 LoRA와 일치시킵니다.",
    },
    # Training
    "learning_rate": {
        "en": "Base learning rate for the optimizer. Typical: 1e-5 to 1e-4.",
        "ko": "옵티마이저 기본 학습률. 일반적: 1e-5 ~ 1e-4.",
    },
    "max_train_epochs": {
        "en": "Total training epochs. One epoch = one full pass through the dataset.",
        "ko": "총 학습 에폭 수. 1 에폭 = 데이터셋 전체를 1회 순회.",
    },
    "save_every_n_epochs": {
        "en": "Save a checkpoint every N epochs. Set equal to max_train_epochs to save only the final model.",
        "ko": "N 에폭마다 체크포인트 저장. max_train_epochs와 같게 설정하면 최종 모델만 저장.",
    },
    "checkpointing_epochs": {
        "en": "Save resumable training state every N epochs. State files are large; use a larger interval than save_every_n_epochs.",
        "ko": "N 에폭마다 학습 재개 상태 저장. 상태 파일이 크므로 save_every_n_epochs보다 큰 간격 권장.",
    },
    "gradient_accumulation_steps": {
        "en": "Accumulate gradients over N steps before updating. Effective batch size = batch_size \u00d7 accumulation_steps.",
        "ko": "N 스텝 동안 그레이디언트 누적 후 업데이트. 실효 배치 크기 = batch_size \u00d7 accumulation_steps.",
    },
    "use_shuffled_caption_variants": {
        "en": "Consume preprocessed caption-shuffle variants from the text-encoder cache. When the cache holds multiple variants, a random one is drawn per sample. Falls back silently to single-variant if no variants were preprocessed.",
        "ko": "전처리된 캡션 셔플 변형을 텍스트 인코더 캐시에서 사용. 캐시에 여러 변형이 있으면 샘플당 무작위 선택. 변형이 전처리되지 않았다면 단일 캡션으로 자동 대체.",
    },
    "caption_dropout_rate": {
        "en": "Probability per sample of dropping the caption (replaced with empty text embedding). Pushes the LoRA toward an unconditional bias — useful for style training where you want the look to apply regardless of prompt. Typical: 0.0–0.05 for character/concept LoRAs, 0.1–0.25 for style LoRAs (그림체 학습). Too high can blur prompt-driven diversity (pose/composition).",
        "ko": "샘플별로 캡션을 비울(빈 텍스트 임베딩으로 대체) 확률. LoRA를 무조건부(unconditional) 방향으로 학습시켜, 프롬프트와 무관하게 항상 적용되는 \"스타일\"을 학습할 때 유리. 일반적: 캐릭터/컨셉 LoRA는 0.0–0.05, 그림체 학습은 0.1–0.25. 너무 높이면 캡션이 담당하던 다양성(포즈/구도)까지 함께 약해짐.",
    },
    "optimizer_type": {
        "en": "Optimizer algorithm. AdamW8bit: memory-efficient 8-bit Adam. Others: AdamW, Lion, Prodigy, etc.",
        "ko": "옵티마이저 알고리즘. AdamW8bit: 메모리 효율적 8비트 Adam. 기타: AdamW, Lion, Prodigy 등.",
    },
    "lr_scheduler": {
        "en": "Learning rate schedule. constant: fixed LR. Others: cosine, cosine_with_restarts, polynomial.",
        "ko": "학습률 스케줄. constant: 고정 LR. 기타: cosine, cosine_with_restarts, polynomial.",
    },
    "timestep_sampling": {
        "en": "How denoising timesteps are sampled during training. sigmoid: biased toward middle timesteps (recommended for flow matching).",
        "ko": "학습 중 디노이징 타임스텝 샘플링 방법. sigmoid: 중간 타임스텝 편향 (flow matching 권장).",
    },
    "discrete_flow_shift": {
        "en": "Flow-matching shift parameter controlling the noise schedule distribution. Default: 1.0.",
        "ko": "노이즈 스케줄 분포를 제어하는 flow-matching 시프트 매개변수. 기본값: 1.0.",
    },
    # Performance
    "attn_mode": {
        "en": "Attention backend. flash4: FlashAttention-4 (Linux, fastest). flash: FlashAttention-2. flex: PyTorch flex attention (cross-platform).",
        "ko": "어텐션 백엔드. flash4: FlashAttention-4 (Linux, 최속). flash: FlashAttention-2. flex: PyTorch flex attention (크로스 플랫폼).",
    },
    "gradient_checkpointing": {
        "en": "Recompute activations during backward pass instead of storing them. Trades compute for VRAM. Essential for low-VRAM setups.",
        "ko": "역전파 시 활성값을 저장 대신 재계산. 연산으로 VRAM 절약. 저사양 필수.",
    },
    "unsloth_offload_checkpointing": {
        "en": "Offload gradient checkpoints to CPU RAM. Further VRAM reduction at cost of speed. Requires gradient_checkpointing=true.",
        "ko": "그레이디언트 체크포인트를 CPU RAM으로 오프로드. 속도 감소 대신 VRAM 추가 절약. gradient_checkpointing=true 필요.",
    },
    "blocks_to_swap": {
        "en": "Number of DiT blocks to swap between GPU and CPU. 0: all on GPU. Higher values = more CPU offloading for low VRAM.",
        "ko": "GPU와 CPU 간 스왑할 DiT 블록 수. 0: 전부 GPU. 높을수록 더 많이 CPU로 오프로드.",
    },
    "torch_compile": {
        "en": "Enable torch.compile for the forward pass. Faster training after initial compilation. Best with static_token_count=true.",
        "ko": "torch.compile 활성화. 초기 컴파일 후 학습 속도 향상. static_token_count=true와 함께 사용 권장.",
    },
    "compile_mode": {
        "en": "'blocks': compile each DiT block individually (default). 'full': compile entire model as one graph for cross-block memory optimization. Full mode is incompatible with gradient checkpointing and block swap.",
        "ko": "'blocks': 각 DiT 블록을 개별 컴파일 (기본값). 'full': 전체 모델을 하나의 그래프로 컴파일하여 블록 간 메모리 최적화. full 모드는 gradient checkpointing 및 block swap과 호환 불가.",
    },
    "trim_crossattn_kv": {
        "en": "Remove zero-padding from cross-attention KV for efficiency. Flash4 applies LSE correction to maintain correct softmax.",
        "ko": "효율을 위해 크로스 어텐션 KV에서 제로 패딩 제거. Flash4는 정확한 softmax를 위해 LSE 보정 적용.",
    },
    "cache_llm_adapter_outputs": {
        "en": "Cache the LLM adapter layer outputs to disk. Avoids recomputing text encoder projections each epoch.",
        "ko": "LLM 어댑터 레이어 출력을 디스크에 캐싱. 매 에폭 텍스트 인코더 투영 재계산 회피.",
    },
    "masked_loss": {
        "en": "Apply loss only to non-masked regions (e.g., exclude text bubbles). Requires mask files in masks/ directory.",
        "ko": "마스크되지 않은 영역에만 손실 적용 (예: 말풍선 제외). masks/ 디렉토리에 마스크 파일 필요.",
    },
    "mixed_precision": {
        "en": "Mixed precision mode. bf16: recommended for modern GPUs. fp16: for older GPUs without bf16 support.",
        "ko": "혼합 정밀도 모드. bf16: 최신 GPU 권장. fp16: bf16 미지원 구형 GPU용.",
    },
    "static_token_count": {
        "en": "Fixed 4096 token count for all batches. Gives torch.compile a single static shape \u2014 no recompilation across aspect ratios.",
        "ko": "모든 배치에 4096 토큰 고정. torch.compile에 단일 정적 셰이프 제공 \u2014 화면비별 재컴파일 없음.",
    },
    "vae_chunk_size": {
        "en": "VAE decoding chunk size. Larger = faster but more VRAM. 64 is a good balance.",
        "ko": "VAE 디코딩 청크 크기. 클수록 빠르지만 VRAM 더 사용. 64가 적절.",
    },
    "vae_disable_cache": {
        "en": "Disable VAE's internal KV cache. Reduces VRAM during VAE encoding/decoding.",
        "ko": "VAE 내부 KV 캐시 비활성화. VAE 인코딩/디코딩 시 VRAM 감소.",
    },
    "cache_latents": {
        "en": "Cache VAE-encoded latents in memory. Avoids re-encoding images every epoch.",
        "ko": "VAE 인코딩된 레이턴트를 메모리에 캐싱. 매 에폭 이미지 재인코딩 회피.",
    },
    "cache_latents_to_disk": {
        "en": "Save cached latents to disk instead of RAM. Frees system memory at cost of disk I/O.",
        "ko": "캐시된 레이턴트를 RAM 대신 디스크에 저장. 디스크 I/O 대신 시스템 메모리 절약.",
    },
    "cache_text_encoder_outputs": {
        "en": "Cache text encoder outputs. Essential for lazy loading: encode \u2192 cache \u2192 free encoder \u2192 load DiT.",
        "ko": "텍스트 인코더 출력 캐싱. 지연 로딩 필수: 인코딩 \u2192 캐시 \u2192 인코더 해제 \u2192 DiT 로드.",
    },
    "cache_text_encoder_outputs_to_disk": {
        "en": "Save cached text encoder outputs to disk. Required for the lazy loading sequence to free VRAM before loading DiT.",
        "ko": "캐시된 텍스트 인코더 출력을 디스크에 저장. DiT 로드 전 VRAM 해제를 위한 지연 로딩 필수.",
    },
    "skip_cache_check": {
        "en": "Skip validation of cached files on startup. Faster startup when caches are known to be valid.",
        "ko": "시작 시 캐시 파일 검증 건너뛰기. 캐시가 유효함을 알 때 빠른 시작.",
    },
    # Paths
    "pretrained_model_name_or_path": {
        "en": "Path to the base DiT model weights (.safetensors).",
        "ko": "기본 DiT 모델 가중치 경로 (.safetensors).",
    },
    "qwen3": {
        "en": "Path to the Qwen3 text encoder weights for text-to-image conditioning.",
        "ko": "텍스트-투-이미지 컨디셔닝용 Qwen3 텍스트 인코더 가중치 경로.",
    },
    "vae": {
        "en": "Path to the VAE model for image encoding/decoding.",
        "ko": "이미지 인코딩/디코딩용 VAE 모델 경로.",
    },
    "output_dir": {
        "en": "Directory for saving trained LoRA checkpoints.",
        "ko": "학습된 LoRA 체크포인트 저장 디렉토리.",
    },
    "output_name": {
        "en": "Base filename for saved checkpoints (epoch number is appended automatically).",
        "ko": "저장되는 체크포인트의 기본 파일명 (에폭 번호 자동 추가).",
    },
    "save_model_as": {
        "en": "Checkpoint format. safetensors: recommended (fast, safe).",
        "ko": "체크포인트 형식. safetensors: 권장 (빠르고 안전).",
    },
    "source_image_dir": {
        "en": (
            "Where raw images and .txt captions live. The Preprocess button feeds "
            "this to resize_images.py (writes resized PNGs) and "
            "cache_text_embeddings.py (caches captions). Override per preset/method "
            "if you keep multiple datasets side by side."
        ),
        "ko": (
            "원본 이미지와 .txt 캡션이 있는 디렉토리. 전처리 버튼이 이 경로를 "
            "resize_images.py(리사이즈된 PNG 저장)와 cache_text_embeddings.py"
            "(캡션 캐시)에 전달합니다. 여러 데이터셋을 병행할 때 프리셋/메소드별로 "
            "오버라이드하세요."
        ),
    },
    "resized_image_dir": {
        "en": (
            "Where preprocess writes VAE-aligned PNGs. Also resolved into the dataset "
            "subset's image_dir at training time (via {resized_image_dir} template "
            "in base.toml), so editing this propagates to both preprocess and training."
        ),
        "ko": (
            "전처리가 VAE에 맞춰 리사이즈한 PNG를 저장하는 디렉토리. 학습 시 "
            "데이터셋 서브셋의 image_dir로도 사용됩니다(base.toml의 "
            "{resized_image_dir} 템플릿 치환). 이 값을 바꾸면 전처리와 학습 양쪽에 "
            "반영됩니다."
        ),
    },
    "lora_cache_dir": {
        "en": (
            "Where preprocess writes VAE latent (.npz) and text-encoder "
            "(_anima_te.safetensors) caches. Also resolved into the dataset subset's "
            "cache_dir at training time."
        ),
        "ko": (
            "전처리가 VAE 잠재 변수(.npz)와 텍스트 인코더 출력"
            "(_anima_te.safetensors) 캐시를 저장하는 디렉토리. 학습 시 데이터셋 "
            "서브셋의 cache_dir로도 사용됩니다."
        ),
    },
}


def field_help(key: str) -> str | None:
    """Return the help string for *key* in the current language, or None."""
    entry = FIELD_HELP.get(key)
    if entry is None:
        return None
    lang = current_language()
    return entry.get(lang) or entry.get("en")


# ── Preprocessing tab field help ───────────────────────────────
# Distinct from the training-time `caption_dropout_rate` entry above:
# these knobs are consumed by preprocess/cache_text_embeddings.py at
# cache-build time, not by the dataloader.
PREPROCESS_FIELD_HELP: dict[str, dict[str, str]] = {
    "caption_shuffle_variants": {
        "en": (
            "Number of caption variants generated per image during text-encoder "
            "caching. v0 is the pristine original caption; v1..v(N-1) are smart-"
            "shuffled (the @artist prefix and 'On the …' / 'In the …' section "
            "anchors are preserved). The dataloader picks v0 with 20% probability "
            "and v1..v(N-1) uniformly otherwise — but only when "
            "use_shuffled_caption_variants=true in your method config. Set to 0 "
            "to cache a single pristine caption only."
        ),
        "ko": (
            "텍스트 인코더 캐싱 시 이미지당 생성되는 캡션 변형 수. v0은 원본 "
            "캡션 그대로이고, v1..v(N-1)은 스마트 셔플됩니다 (@artist 접두사와 "
            "'On the …' / 'In the …' 섹션 앵커는 보존). "
            "데이터로더는 v0을 20% 확률로 선택하고 나머지는 v1..v(N-1) 균등 "
            "분포로 선택합니다 — 단, 메소드 설정에 "
            "use_shuffled_caption_variants=true 일 때만 적용됩니다. "
            "0으로 설정하면 원본 캡션 하나만 캐싱합니다."
        ),
    },
    "caption_tag_dropout_rate": {
        "en": (
            "Per-tag dropout probability applied only to v1..v(N-1) shuffle "
            "variants. Tags up to and including the first @artist marker are "
            "never dropped — the artist tag is structurally important and the "
            "rating/character/series prefix is order-sensitive. Ignored when "
            "shuffle variants ≤ 0. Typical: 0.05–0.15. Higher values teach the "
            "LoRA to generalize across missing tags but can dilute the signal."
        ),
        "ko": (
            "v1..v(N-1) 셔플 변형에만 적용되는 태그별 드롭아웃 확률입니다. "
            "첫 번째 @artist 마커까지의 태그는 절대 드롭되지 않습니다 — "
            "작가 태그는 구조적으로 중요하고, 등급/캐릭터/작품 접두사는 "
            "순서가 중요합니다. 셔플 변형 수가 0 이하이면 무시됩니다. "
            "일반적: 0.05–0.15. 값이 높을수록 누락된 태그에 강건해지지만 "
            "학습 신호가 희석될 수 있습니다."
        ),
    },
    "sam_prompts": {
        "en": (
            "Text prompts SAM3 will look for in each image. One prompt per "
            "line. Defaults are tuned for manga-style speech / text bubbles. "
            "Add custom prompts if your dataset has additional regions you "
            "want masked out (e.g. 'sound effect', 'caption box'). Saved to "
            "configs/sam_mask.yaml — read directly by "
            "preprocess/generate_masks.py."
        ),
        "ko": (
            "SAM3이 각 이미지에서 찾을 텍스트 프롬프트. 한 줄에 하나씩. "
            "기본값은 만화 스타일 말풍선 / 텍스트 영역에 맞춰져 있습니다. "
            "추가로 마스킹하고 싶은 영역이 있으면 커스텀 프롬프트를 "
            "추가하세요 (예: 'sound effect', 'caption box'). "
            "configs/sam_mask.yaml에 저장되며 "
            "preprocess/generate_masks.py가 직접 읽습니다."
        ),
    },
    "sam_threshold": {
        "en": (
            "Minimum confidence required for a SAM3 detection to be kept. "
            "Range 0.0–1.0. Lower values produce more masks (with more false "
            "positives); higher values are stricter. Default 0.5. If SAM is "
            "missing real bubbles, lower this; if it's masking unrelated "
            "regions, raise it."
        ),
        "ko": (
            "SAM3 탐지를 유지하기 위한 최소 신뢰도. 범위 0.0–1.0. 값이 "
            "낮을수록 더 많은 마스크가 생성되지만 오탐도 늘어나며, 값이 "
            "높을수록 엄격해집니다. 기본값 0.5. SAM이 실제 말풍선을 "
            "놓치면 값을 낮추고, 관련 없는 영역을 마스킹하면 높이세요."
        ),
    },
    "sam_dilate": {
        "en": (
            "Pixels of binary dilation applied to each SAM mask after "
            "thresholding. Larger values blur mask edges outward — useful "
            "when the underlying segmentation undershoots the actual text "
            "bubble border. Default 5. Set to 0 to disable dilation."
        ),
        "ko": (
            "임계값 처리 후 각 SAM 마스크에 적용되는 이진 팽창 픽셀 수. "
            "값이 클수록 마스크 가장자리가 바깥으로 번집니다 — 실제 "
            "말풍선 경계보다 분할 결과가 작을 때 유용합니다. 기본값 5. "
            "0으로 비활성화."
        ),
    },
    "mit_text_threshold": {
        "en": (
            "Confidence threshold for the MIT/ComicTextDetector text "
            "segmenter. Range 0.0–1.0. Independent of SAM threshold — MIT "
            "uses a different model trained specifically on manga text "
            "regions. Default 0.8. Lower if MIT is missing text inside "
            "panels; raise if it's catching non-text artifacts."
        ),
        "ko": (
            "MIT/ComicTextDetector 텍스트 분할기의 신뢰도 임계값. 범위 "
            "0.0–1.0. SAM 임계값과는 별개입니다 — MIT는 만화 텍스트 "
            "영역에 특화된 다른 모델을 사용합니다. 기본값 0.8. 패널 "
            "내부 텍스트를 놓치면 낮추고, 텍스트가 아닌 부분을 잡으면 "
            "높이세요."
        ),
    },
    "mit_dilate": {
        "en": (
            "Pixels of binary dilation applied to each MIT mask. Same role "
            "as SAM dilate but tuned independently — MIT typically segments "
            "tight bounding regions around individual glyphs, so a moderate "
            "dilate value joins them into per-bubble blobs. Default 5."
        ),
        "ko": (
            "각 MIT 마스크에 적용되는 이진 팽창 픽셀 수. SAM 팽창과 같은 "
            "역할이지만 독립적으로 조정 — MIT는 일반적으로 개별 글리프 "
            "주변에 타이트한 경계를 분할하므로, 적당한 팽창 값으로 "
            "말풍선 단위 블롭으로 합쳐줍니다. 기본값 5."
        ),
    },
}


def preprocess_field_help(key: str) -> str | None:
    """Per-field help for the Preprocessing tab. Falls back to FIELD_HELP."""
    entry = PREPROCESS_FIELD_HELP.get(key)
    if entry is None:
        return field_help(key)
    lang = current_language()
    return entry.get(lang) or entry.get("en")


PREPROCESS_GUIDE: dict[str, str] = {
    "en": (
        "<h2 style='margin:0 0 10px 0; font-size:18px;'>Preprocessing</h2>"
        "<p>Three artifact lanes feed training:</p>"
        "<p><b>1. Text caching</b> &mdash; tokenizes captions through Qwen3 "
        "+ T5 and writes <code>{stem}_anima_te.safetensors</code> sidecars. "
        "When <i>shuffle variants</i> &gt; 0, each cache holds N captions and "
        "the dataloader samples one per training step (turn on "
        "<code>use_shuffled_caption_variants</code> in your method config to "
        "use them).</p>"
        "<p><b>2. SAM3 masking</b> &mdash; finds text-bubble regions using "
        "natural-language prompts. Saved to <code>masks/sam/</code>.</p>"
        "<p><b>3. MIT masking</b> &mdash; runs a manga-specific text "
        "segmenter trained on glyph-level data. Saved to "
        "<code>masks/mit/</code>.</p>"
        "<p>The <b>Merge</b> step unions SAM + MIT into "
        "<code>masks/merged/</code>. When the merged dir exists, training "
        "subsets pick it up automatically (see <code>masked_loss</code> in "
        "the method config).</p>"
        "<p style='color:#888; font-style:italic; margin-top:12px;'>"
        "Click any field label on the left to see its explanation here.</p>"
    ),
    "ko": (
        "<h2 style='margin:0 0 10px 0; font-size:18px;'>전처리</h2>"
        "<p>학습은 세 가지 산출물 레인을 사용합니다:</p>"
        "<p><b>1. 텍스트 캐싱</b> &mdash; Qwen3 + T5로 캡션을 토큰화하고 "
        "<code>{stem}_anima_te.safetensors</code> 파일로 저장합니다. "
        "<i>셔플 변형 수</i>가 0보다 크면 각 캐시에 N개의 캡션이 들어가며 "
        "데이터로더는 학습 스텝마다 하나를 샘플링합니다 (메소드 설정에서 "
        "<code>use_shuffled_caption_variants</code>를 켜야 사용됩니다).</p>"
        "<p><b>2. SAM3 마스킹</b> &mdash; 자연어 프롬프트로 말풍선 영역을 "
        "찾습니다. <code>masks/sam/</code>에 저장.</p>"
        "<p><b>3. MIT 마스킹</b> &mdash; 글리프 단위 데이터로 학습된 "
        "만화 전용 텍스트 분할기를 실행합니다. <code>masks/mit/</code>에 "
        "저장.</p>"
        "<p><b>병합</b> 단계는 SAM + MIT의 합집합을 "
        "<code>masks/merged/</code>에 만듭니다. 병합 디렉토리가 있으면 "
        "학습 서브셋이 자동으로 사용합니다 (메소드 설정의 "
        "<code>masked_loss</code> 참조).</p>"
        "<p style='color:#888; font-style:italic; margin-top:12px;'>"
        "왼쪽 필드 라벨을 클릭하면 여기에 설명이 표시됩니다.</p>"
    ),
}


def preprocess_guide() -> str:
    lang = current_language()
    return PREPROCESS_GUIDE.get(lang) or PREPROCESS_GUIDE["en"]


# ── LoRA variant guide (rich HTML) ────────────────────────────

APPLY_NOTE_HTML: dict[str, str] = {
    "en": (
        "<p style='margin:0 0 12px 0; color:#f0c14b; font-size:12px;'>"
        "<b>Apply</b> fills the form with the picked variant's defaults — "
        "<b>nothing is saved until you click Save</b>.</p>"
    ),
    "ko": (
        "<p style='margin:0 0 12px 0; color:#f0c14b; font-size:12px;'>"
        "<b>Apply</b>는 선택한 variant의 기본값으로 폼을 채웁니다 — "
        "<b>Save를 누르기 전까지는 저장되지 않습니다</b>.</p>"
    ),
}


LORA_GUIDE: dict[str, str] = {
    "en": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>LoRA</h2>"
        "<p>Classic low-rank adaptation. Adds small trainable matrices "
        "(down &times; up) to existing weight layers.<br>"
        "<code>y = x + (x @ down @ up) &times; scale &times; multiplier</code><br>"
        "Simple, effective, and the default choice for most fine-tuning tasks. "
        "Fully bakeable into the base DiT.</p>"
        "<p><b>Variants</b><br>"
        "&bull; <b>lora</b> &mdash; default 16GB+ profile.<br>"
        "&bull; <b>lora_longer</b> &mdash; same architecture, more epochs for "
        "datasets that haven't converged.<br>"
        "&bull; <b>lora-8gb</b> &mdash; low-VRAM profile (block swap + offload "
        "checkpointing); pick this if you OOM on the default.</p>"
    ),
    "ko": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>LoRA</h2>"
        "<p>클래식 저랭크 적응. 기존 가중치 레이어에 작은 학습 가능한 "
        "행렬(down &times; up)을 추가.<br>"
        "<code>y = x + (x @ down @ up) &times; scale &times; multiplier</code><br>"
        "간단하고 효과적이며, 대부분의 파인튜닝에 기본 선택. 베이스 DiT에 "
        "완전히 베이킹 가능.</p>"
        "<p><b>변형</b><br>"
        "&bull; <b>lora</b> &mdash; 기본 16GB+ 프로필.<br>"
        "&bull; <b>lora_longer</b> &mdash; 동일 아키텍처, 더 긴 에폭 (수렴이 "
        "느린 데이터셋용).<br>"
        "&bull; <b>lora-8gb</b> &mdash; 저VRAM 프로필 (블록 스왑 + 오프로드 "
        "체크포인팅); 기본 프로필에서 OOM이 나면 이 변형을 사용.</p>"
    ),
}


def lora_guide() -> str:
    """Return the LoRA method guide HTML for the current language."""
    lang = current_language()
    return LORA_GUIDE.get(lang) or LORA_GUIDE["en"]


# ── OrthoLoRA guide ─────────────────────────────────────────

ORTHOLORA_GUIDE: dict[str, str] = {
    "en": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>OrthoLoRA</h2>"
        "<p>SVD-based orthogonal parameterization of the update matrix "
        "(linear layers only). Uses QR-decomposed orthonormal bases with "
        "learned singular values: <code>P @ diag(&lambda;) @ Q</code>. "
        "Orthogonality regularization keeps the update structured, which "
        "tends to reduce interference with unrelated concepts.</p>"
        "<p>Saved as a plain LoRA at checkpoint time via thin SVD on "
        "&Delta;W &mdash; the result is a regular LoRA "
        "<code>.safetensors</code> bakeable into the base DiT. The "
        "orthogonality machinery is training-only.</p>"
        "<p>Toggle: <code>use_ortho = true</code>.</p>"
    ),
    "ko": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>OrthoLoRA</h2>"
        "<p>업데이트 행렬의 SVD 기반 직교 파라미터화 (선형 레이어 전용). "
        "QR 분해된 정규 직교 기저와 학습된 특이값을 사용: "
        "<code>P @ diag(&lambda;) @ Q</code>. 직교성 정규화로 업데이트가 "
        "구조화되며, 무관한 컨셉과의 간섭을 줄이는 경향이 있습니다.</p>"
        "<p>저장 시 &Delta;W에 대한 thin SVD로 일반 LoRA로 변환됨 &mdash; "
        "결과물은 베이스 DiT에 베이킹 가능한 일반 LoRA "
        "<code>.safetensors</code>입니다. 직교성 메커니즘은 학습 시에만 "
        "작동합니다.</p>"
        "<p>토글: <code>use_ortho = true</code>.</p>"
    ),
}


def ortholora_guide() -> str:
    lang = current_language()
    return ORTHOLORA_GUIDE.get(lang) or ORTHOLORA_GUIDE["en"]


# ── T-LoRA guide ────────────────────────────────────────────

TLORA_GUIDE: dict[str, str] = {
    "en": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>T-LoRA</h2>"
        "<p>Timestep-dependent rank masking. The effective LoRA rank varies "
        "with the denoising timestep via a power-law schedule:<br>"
        "&bull; High noise (early steps) &rarr; full rank (maximum expressiveness)<br>"
        "&bull; Low noise (late steps) &rarr; reduced rank (down to "
        "<code>min_rank</code>)<br>"
        "Concentrates capacity where structure is being decided. The mask is "
        "training-only — inference uses full rank at every t by design — so "
        "the saved <code>.safetensors</code> is a regular bakeable LoRA. "
        "See <code>docs/methods/timestep_mask.md</code>.</p>"
        "<p>Toggle: <code>use_timestep_mask = true</code>.</p>"
        "<p><b>Variants</b><br>"
        "&bull; <b>tlora</b> &mdash; pure T-LoRA, no orthogonality constraint.<br>"
        "&bull; <b>tlora_ortho</b> &mdash; T-LoRA stacked on OrthoLoRA "
        "(<code>use_ortho = true</code> + <code>use_timestep_mask = true</code>). "
        "Recommended general-purpose pick when you want both rank-by-σ scheduling "
        "and orthogonal regularization. Still fully bakeable.</p>"
    ),
    "ko": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>T-LoRA</h2>"
        "<p>타임스텝 의존 랭크 마스킹. 디노이징 타임스텝에 따라 유효 LoRA "
        "랭크가 거듭제곱 스케줄로 변동:<br>"
        "&bull; 높은 노이즈 (초기 스텝) &rarr; 전체 랭크 (최대 표현력)<br>"
        "&bull; 낮은 노이즈 (후기 스텝) &rarr; 축소된 랭크 (<code>min_rank</code>까지)<br>"
        "구조가 결정되는 시점에 표현력을 집중시킵니다. 마스크는 학습 시에만 "
        "적용되며 추론은 모든 t에서 전체 랭크를 사용하므로, 저장된 "
        "<code>.safetensors</code>는 베이킹 가능한 일반 LoRA입니다. "
        "<code>docs/methods/timestep_mask.md</code> 참조.</p>"
        "<p>토글: <code>use_timestep_mask = true</code>.</p>"
        "<p><b>변형</b><br>"
        "&bull; <b>tlora</b> &mdash; 순수 T-LoRA, 직교성 제약 없음.<br>"
        "&bull; <b>tlora_ortho</b> &mdash; T-LoRA + OrthoLoRA 스택 "
        "(<code>use_ortho = true</code> + <code>use_timestep_mask = true</code>). "
        "σ별 랭크 스케줄링과 직교성 정규화를 모두 원할 때 추천하는 범용 선택. "
        "여전히 완전히 베이킹 가능합니다.</p>"
    ),
}


def tlora_guide() -> str:
    lang = current_language()
    return TLORA_GUIDE.get(lang) or TLORA_GUIDE["en"]


POSTFIX_GUIDE: dict[str, str] = {
    "en": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>Postfix (cond + ortho)</h2>"
        "<p><b>postfix_ortho_cond</b> &mdash; Caption-conditional postfix with "
        "structural orthogonality. A small <code>cond_mlp</code> reads the pooled "
        "caption embedding and emits <code>(S(c), λ(c))</code> per caption — "
        "<code>K(K-1)/2 + 1</code> scalars — instead of a direct <code>K·D</code> "
        "postfix tensor. Each caption's postfix is "
        "<code>Cayley(S(c) − S(c)ᵀ) @ basis · λ(c)</code>, so "
        "<code>postfix(c) @ postfix(c).T = λ(c)² · I_K</code> structurally: "
        "uniform per-slot magnitude within a caption, magnitude varies across "
        "captions.</p>"
        "<p>Basis is the top-K right singular vectors of cached "
        "<code>_anima_te.safetensors</code>, row-shuffled deterministically by "
        "<code>ortho_basis_seed</code>. Trainable surface ~390k params at "
        "K=32 / hidden=256 / D=1024 (vs ~8.65M for a legacy unconstrained cond "
        "postfix at K=64). See "
        "<code>docs/proposal/orthogonal_postfix.md §C</code>.</p>"
    ),
    "ko": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>Postfix (cond + ortho)</h2>"
        "<p><b>postfix_ortho_cond</b> &mdash; 구조적 직교성을 갖춘 캡션 조건부 postfix. "
        "작은 <code>cond_mlp</code>이 풀링된 캡션 임베딩을 읽어 캡션별로 "
        "<code>(S(c), λ(c))</code>를 내놓습니다 — 직접적인 <code>K·D</code> "
        "postfix 텐서 대신 캡션당 <code>K(K-1)/2 + 1</code>개의 스칼라만 출력. "
        "각 캡션의 postfix는 <code>Cayley(S(c) − S(c)ᵀ) @ basis · λ(c)</code>이며, "
        "따라서 구조적으로 <code>postfix(c) @ postfix(c).T = λ(c)² · I_K</code>: "
        "캡션 내 슬롯별 크기는 균일하고, 캡션 간에는 크기가 변동합니다.</p>"
        "<p>basis는 캐시된 <code>_anima_te.safetensors</code>의 top-K 우특이벡터로, "
        "<code>ortho_basis_seed</code>에 따라 결정론적으로 행-셔플됩니다. "
        "학습 가능한 파라미터는 K=32 / hidden=256 / D=1024 기준 약 390k개 "
        "(K=64에서 제약 없는 레거시 cond postfix의 ~8.65M 대비). "
        "<code>docs/proposal/orthogonal_postfix.md §C</code> 참조.</p>"
    ),
}


def postfix_guide() -> str:
    """Return the postfix/prefix variant guide HTML for the current language."""
    lang = current_language()
    return POSTFIX_GUIDE.get(lang) or POSTFIX_GUIDE["en"]


# ── "Not mergeable" callout ───────────────────────────────────
# Reused by HydraLoRA / ReFT / Postfix guides — these methods can't be
# baked into a plain DiT via scripts/merge_to_dit.py (router is layer-local
# / hook-only / not a weight delta), so the user has to load the adapter at
# inference time instead of distributing a merged checkpoint.

NOT_MERGEABLE_HTML: dict[str, str] = {
    "en": (
        "<div style='background:#33231e; padding:10px 14px; border-left:3px solid #e67e22; "
        "margin-bottom:14px; border-radius:3px;'>"
        "<p style='margin:0; color:#f0c14b;'><b>⚠ Not mergeable into the base DiT.</b> "
        "This method can't be baked via the Merge tab / "
        "<code>scripts/merge_to_dit.py</code> — it relies on a runtime hook, "
        "layer-local router, or non-weight delta. Distribute the adapter "
        "<code>.safetensors</code> alongside the base model and load it at "
        "inference time (e.g. ComfyUI <i>Anima Adapter Loader</i>).</p>"
        "</div>"
    ),
    "ko": (
        "<div style='background:#33231e; padding:10px 14px; border-left:3px solid #e67e22; "
        "margin-bottom:14px; border-radius:3px;'>"
        "<p style='margin:0; color:#f0c14b;'><b>⚠ 베이스 DiT에 병합 불가능.</b> "
        "이 방식은 Merge 탭 / <code>scripts/merge_to_dit.py</code>로 베이킹할 수 "
        "없습니다 — 런타임 훅, 레이어 로컬 라우터, 또는 가중치 델타가 아닌 형태로 동작하기 "
        "때문입니다. 어댑터 <code>.safetensors</code>를 베이스 모델과 함께 배포하고 추론 "
        "시점에 로드하세요 (예: ComfyUI <i>Anima Adapter Loader</i>).</p>"
        "</div>"
    ),
}


def not_mergeable_note() -> str:
    lang = current_language()
    return NOT_MERGEABLE_HTML.get(lang) or NOT_MERGEABLE_HTML["en"]


# ── HydraLoRA guide ──────────────────────────────────────────

HYDRALORA_GUIDE: dict[str, str] = {
    "en": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>HydraLoRA</h2>"
        "<p>MoE-style routing on top of LoRA: shared <code>lora_down</code> + "
        "<code>num_experts</code> per-expert <code>lora_up</code> heads, routed "
        "layer-locally from the adapted Linear's input. A load-balance loss "
        "(<code>balance_loss_weight</code>) discourages router collapse onto a "
        "single expert.</p>"
        "<p><b>hydralora_sigma</b> &mdash; Adds a tiny sinusoidal(σ)→E bias MLP per "
        "router so expert choice can vary with denoising timestep. Zero-init at "
        "the final layer means step-0 starts identical to base HydraLoRA; σ-"
        "dependence only emerges if gradients push it.</p>"
        "<p><b>hydralora_experimental</b> &mdash; <code>hydralora_sigma</code> plus "
        "hard σ-band specialization: <code>specialize_experts_by_sigma_buckets = "
        "true</code> partitions the expert pool by timestep bucket so each σ-"
        "band only routes to its assigned experts (e.g. with "
        "<code>num_experts = 6</code> and 3 buckets, 2 experts per band). "
        "<code>sigma_bucket_boundaries = [0.0, 0.5, 0.8, 1.0]</code> places the "
        "splits unevenly &mdash; more capacity in the late, low-noise refinement "
        "regime. More opinionated than the soft σ-bias router; useful when you "
        "want experts to diverge along the timestep axis rather than "
        "co-specialize.</p>"
        "<p>Training produces a <code>*_moe.safetensors</code> sibling next to "
        "the adapter; both files are needed for router-live inference "
        "(<code>make test-hydra</code> / ComfyUI <i>Anima Adapter Loader</i>). "
        "Requires <code>cache_llm_adapter_outputs = true</code>.</p>"
    ),
    "ko": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>HydraLoRA</h2>"
        "<p>LoRA 위에 MoE 스타일 라우팅을 얹은 방식: 공유 <code>lora_down</code> + "
        "<code>num_experts</code>개의 전문가별 <code>lora_up</code> 헤드를, 적응된 "
        "Linear의 입력으로부터 레이어 로컬하게 라우팅합니다. 부하 균형 손실"
        "(<code>balance_loss_weight</code>)이 단일 전문가 붕괴를 방지합니다.</p>"
        "<p><b>hydralora_sigma</b> &mdash; 각 라우터에 sinusoidal(σ)→E 바이어스 MLP를 "
        "추가하여 전문가 선택이 디노이징 타임스텝에 따라 변동. 최종 레이어 zero-init → "
        "초기에는 기본 HydraLoRA와 동일하며, σ-의존성은 그래디언트가 발생시킬 때만 "
        "발현합니다.</p>"
        "<p><b>hydralora_experimental</b> &mdash; <code>hydralora_sigma</code>에 "
        "σ-밴드 하드 특화를 추가: <code>specialize_experts_by_sigma_buckets = "
        "true</code>로 전문가 풀을 타임스텝 버킷별로 분할하여 각 σ-밴드가 할당된 "
        "전문가만 사용합니다 (예: <code>num_experts = 6</code>, 버킷 3개일 때 밴드당 "
        "전문가 2개). <code>sigma_bucket_boundaries = [0.0, 0.5, 0.8, 1.0]</code>로 "
        "분할 지점을 비균등하게 배치 &mdash; 후반 저노이즈 디테일 단계에 더 많은 용량을 "
        "할당합니다. 소프트 σ-바이어스 라우터보다 강한 제약이며, 전문가가 타임스텝 축을 "
        "따라 분화되도록 명시적으로 강제하고 싶을 때 유용합니다.</p>"
        "<p>학습 시 어댑터 옆에 <code>*_moe.safetensors</code> 동반 파일이 생성되며, "
        "라우터-라이브 추론에는 두 파일 모두 필요합니다 "
        "(<code>make test-hydra</code> / ComfyUI <i>Anima Adapter Loader</i>). "
        "<code>cache_llm_adapter_outputs = true</code> 필요.</p>"
    ),
}


def hydralora_guide() -> str:
    lang = current_language()
    return HYDRALORA_GUIDE.get(lang) or HYDRALORA_GUIDE["en"]


# ── ReFT guide ──────────────────────────────────────────────

REFT_GUIDE: dict[str, str] = {
    "en": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>ReFT</h2>"
        "<p>Block-level residual-stream intervention (Wu et al., NeurIPS 2024). "
        "One <code>ReFTModule</code> per selected DiT block adds "
        "<code>R^T &middot; (&Delta;W&middot;h + b) &middot; scale</code> to the "
        "block's output — an additive side-channel, not a weight delta on any "
        "Linear. Composes with any LoRA variant and lives in the same "
        "<code>.safetensors</code>.</p>"
        "<p>Pick blocks with <code>reft_layers</code> (e.g. <code>last_8</code>, "
        "<code>stride_2</code>, or comma-separated indices). <code>reft_dim</code> "
        "controls intervention rank; <code>reft_alpha</code> sets effective scale "
        "via <code>alpha / dim</code>.</p>"
        "<p><b>tlora_ortho_reft</b> bundles ReFT with T-LoRA + OrthoLoRA in one "
        "training run — the LoRA half is bakeable, but the ReFT block hooks "
        "aren't, so the merged DiT loses ReFT's contribution (the merge tool "
        "warns and asks for <code>--allow-partial</code>).</p>"
    ),
    "ko": (
        "<h2 style='margin:0 0 10px 0; font-size:17px;'>ReFT</h2>"
        "<p>블록 수준 잔차 스트림 개입 (Wu et al., NeurIPS 2024). 선택된 각 DiT "
        "블록에 하나의 <code>ReFTModule</code>이 "
        "<code>R^T &middot; (&Delta;W&middot;h + b) &middot; scale</code>을 블록 "
        "출력에 추가합니다 — Linear의 가중치 델타가 아닌 추가 사이드 채널입니다. 모든 "
        "LoRA 변형과 함께 사용 가능하며 동일한 <code>.safetensors</code>에 저장됩니다.</p>"
        "<p><code>reft_layers</code>로 블록 선택 (예: <code>last_8</code>, "
        "<code>stride_2</code>, 또는 쉼표 구분 인덱스). <code>reft_dim</code>은 개입 "
        "랭크를, <code>reft_alpha</code>는 <code>alpha / dim</code>으로 실효 스케일을 "
        "설정합니다.</p>"
        "<p><b>tlora_ortho_reft</b>는 ReFT를 T-LoRA + OrthoLoRA와 한 번의 학습으로 "
        "묶은 변형입니다. LoRA 부분은 베이킹 가능하지만 ReFT 블록 훅은 베이킹할 수 "
        "없어 병합된 DiT에서는 ReFT 기여가 사라집니다 (Merge 도구가 경고하고 "
        "<code>--allow-partial</code>을 요구합니다).</p>"
    ),
}


def reft_guide() -> str:
    lang = current_language()
    return REFT_GUIDE.get(lang) or REFT_GUIDE["en"]


def apply_note() -> str:
    """HTML block explaining Apply semantics — shown above variant guides."""
    lang = current_language()
    return APPLY_NOTE_HTML.get(lang) or APPLY_NOTE_HTML["en"]


def method_guide(method: str) -> str | None:
    """Right-panel default HTML for *method*, or None if no guide is registered."""
    if method == "lora":
        return apply_note() + lora_guide()
    if method == "ortholora":
        return apply_note() + ortholora_guide()
    if method == "tlora":
        return apply_note() + tlora_guide()
    if method == "postfix":
        return apply_note() + not_mergeable_note() + postfix_guide()
    if method == "hydralora":
        return apply_note() + not_mergeable_note() + hydralora_guide()
    if method == "reft":
        return apply_note() + not_mergeable_note() + reft_guide()
    return None
