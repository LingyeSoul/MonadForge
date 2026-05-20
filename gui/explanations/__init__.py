"""Trilingual help text for config fields and LoRA variant descriptions.

Per-field tooltips (short, dense, frequently looked up) live inline as
``FIELD_HELP`` / ``PREPROCESS_FIELD_HELP``. The bulky method/variant guide
HTML blocks live under ``guides/<name>.<lang>.html`` — one file per method
per language — and are loaded lazily on first access. Shared snippets
(``_apply_note``, ``_not_mergeable``) follow the same convention with an
underscore prefix.
"""

from __future__ import annotations

import functools
from pathlib import Path

from gui.i18n import current_language

_GUIDES_DIR = Path(__file__).parent / "guides"


@functools.lru_cache(maxsize=None)
def _read_guide(name: str, lang: str) -> str:
    path = _GUIDES_DIR / f"{name}.{lang}.html"
    if not path.exists():
        path = _GUIDES_DIR / f"{name}.en.html"
    return path.read_text(encoding="utf-8")


def _guide(name: str) -> str:
    return _read_guide(name, current_language())


# ── Per-field tooltips ─────────────────────────────────────────
# Keys match config field names. Each maps to {lang: description}.

FIELD_HELP: dict[str, dict[str, str]] = {
    # Architecture
    "network_dim": {
        "en": "LoRA rank (dimension of low-rank matrices). Higher = more expressive but more VRAM. Typical: 8–64.",
        "ko": "LoRA 랭크 (저랭크 행렬의 차원). 높을수록 표현력이 좋지만 VRAM 사용량 증가. 일반적: 8–64.",
        "cn": "LoRA 秩（低秩矩阵的维度）。越高表达力越强，但显存占用越大。典型值：8–64。",
    },
    "network_alpha": {
        "en": "LoRA scaling factor. Effective scale = alpha / dim. When alpha == dim, scale is 1.0. Lower alpha = more conservative updates.",
        "ko": "LoRA 스케일링 계수. 실효 스케일 = alpha / dim. alpha == dim이면 1.0. 낮을수록 보수적 업데이트.",
        "cn": "LoRA 缩放因子。实际缩放 = alpha / dim。alpha == dim 时缩放为 1.0。alpha 越低，更新越保守。",
    },
    "network_module": {
        "en": "Python module path for the LoRA network implementation.",
        "ko": "LoRA 네트워크 구현의 Python 모듈 경로.",
        "cn": "LoRA 网络实现的 Python 模块路径。",
    },
    "use_timestep_mask": {
        "en": "Enable T-LoRA: effective rank varies with denoising timestep via power-law schedule. Full rank at high noise, reduced at low noise.",
        "ko": "T-LoRA 활성화: 디노이징 타임스텝에 따라 유효 랭크 변동. 높은 노이즈에서 전체 랭크, 낮은 노이즈에서 축소.",
        "cn": "启用 T-LoRA：有效秩随去噪时间步按幂律调度变化。高噪声时为满秩，低噪声时降秩。",
    },
    "use_ortho": {
        "en": "Enable OrthoLoRA: SVD-based orthogonal parameterization of the update matrix (linear layers only). Regularizes toward structured updates; saved as plain LoRA via thin SVD at checkpoint time.",
        "ko": "OrthoLoRA 활성화: 업데이트 행렬의 SVD 기반 직교 파라미터화 (선형 레이어 전용). 구조화된 업데이트로 정규화되며, 저장 시 thin SVD로 일반 LoRA로 변환.",
        "cn": "启用 OrthoLoRA：基于 SVD 的更新矩阵正交参数化（仅线性层）。正则化为结构化更新；保存时通过 thin SVD 转为普通 LoRA 格式。",
    },
    "use_moe_style": {
        "en": "MoE expert layout: 'shared_A' (HydraLoRA — one shared lora_down + N per-expert lora_up heads), 'independent_A' (FeRA — N fully-independent down/up pairs), or false (no MoE). Produces a *_moe.safetensors sibling for router-live inference; requires cache_llm_adapter_outputs=true.",
        "ko": "MoE 전문가 레이아웃: 'shared_A' (HydraLoRA — 공유 lora_down + N개 전문가별 lora_up), 'independent_A' (FeRA — 독립적인 N쌍의 down/up), 또는 false (MoE 비활성화). 라우터-라이브 추론용 *_moe.safetensors 동반 파일 생성. cache_llm_adapter_outputs=true 필요.",
        "cn": "MoE 专家布局：'shared_A'（HydraLoRA — 共享 lora_down + N 个专家 lora_up 头）、'independent_A'（FeRA — N 组完全独立的 down/up 对）或 false（不使用 MoE）。生成 *_moe.safetensors 配套文件用于路由推理；需要 cache_llm_adapter_outputs=true。",
    },
    "route_per_layer": {
        "en": "If true, each layer owns its own router (per-layer routing). If false, a single network-level GlobalRouter broadcasts gate weights to every routed module.",
        "ko": "true이면 레이어별 라우터 사용 (per-layer routing). false이면 네트워크 전역 GlobalRouter 하나가 모든 라우팅 모듈에 게이트 가중치 브로드캐스트.",
        "cn": "true 时每层拥有独立路由器（逐层路由）。false 时由单个网络级 GlobalRouter 向所有路由模块广播门控权重。",
    },
    "router_source": {
        "en": "Routing signal: 'sigma' (sinusoidal embedding of the denoising timestep), 'fei' (mean-pooled rank features from preceding LoRA modules), or 'pooled_text' (pooled T5 caption embedding).",
        "ko": "라우팅 신호: 'sigma' (디노이징 타임스텝의 sinusoidal 임베딩), 'fei' (선행 LoRA 모듈의 평균 풀링 랭크 특징), 또는 'pooled_text' (T5 캡션 풀링 임베딩).",
        "cn": "路由信号：'sigma'（去噪时间步的正弦嵌入）、'fei'（前序 LoRA 模块的均值池化秩特征）或 'pooled_text'（T5 标签的池化嵌入）。",
    },
    "num_experts": {
        "en": "HydraLoRA expert count. More experts = more capacity but more VRAM and slower training. Typical: 2–8.",
        "ko": "HydraLoRA 전문가 수. 많을수록 표현력 증가하지만 VRAM 사용량 증가 및 학습 속도 감소. 일반적: 2–8.",
        "cn": "HydraLoRA 专家数量。越多容量越大，但显存占用增加、训练变慢。典型值：2–8。",
    },
    "balance_loss_weight": {
        "en": "HydraLoRA load-balancing loss weight. Discourages router collapse onto a single expert. Typical: 0.01.",
        "ko": "HydraLoRA 부하 균형 손실 가중치. 라우터가 단일 전문가로 붕괴되는 것을 방지. 일반적: 0.01.",
        "cn": "HydraLoRA 负载均衡损失权重。防止路由器坍缩到单一专家。典型值：0.01。",
    },
    "balance_loss_warmup_ratio": {
        "en": "Fraction of training steps to hold the balance loss at 0 before activating it. Lets the router specialize first, then switches the penalty on to stop further collapse of a diverged router. 0.0 disables the warmup. Typical: 0.3–0.5.",
        "ko": "밸런스 손실을 0으로 유지하는 학습 스텝 비율. 먼저 라우터가 전문화되도록 한 뒤 페널티를 활성화해 분화된 라우터의 추가 붕괴를 방지. 0.0 = 비활성화. 일반적: 0.3–0.5.",
        "cn": "训练步数中将平衡损失保持为 0 的比例。让路由器先自主专业化，再启用惩罚阻止坍缩。0.0 禁用预热。典型值：0.3–0.5。",
    },
    "add_reft": {
        "en": "Enable ReFT: block-level residual-stream intervention (Wu et al. 2024). Adds R^T·(ΔW·h + b)·scale to each selected DiT block's output. Composes with any LoRA variant.",
        "ko": "ReFT 활성화: 블록 수준 잔차 스트림 개입 (Wu et al. 2024). 선택된 DiT 블록 출력에 R^T·(ΔW·h + b)·scale 추가. 모든 LoRA 변형과 함께 사용 가능.",
        "cn": "启用 ReFT：块级残差流干预（Wu et al. 2024）。向选定的 DiT 块输出添加 R^T·(ΔW·h + b)·scale。可与任何 LoRA 变体组合。",
    },
    "reft_dim": {
        "en": "ReFT intervention rank — dimension of R and ΔW in each ReFTModule. Typical: 32–64.",
        "ko": "ReFT 개입 랭크 — 각 ReFTModule의 R 및 ΔW 차원. 일반적: 32–64.",
        "cn": "ReFT 干预秩 — 每个 ReFTModule 中 R 和 ΔW 的维度。典型值：32–64。",
    },
    "reft_alpha": {
        "en": "ReFT scaling factor (effective scale = alpha / dim). Typical: same as reft_dim.",
        "ko": "ReFT 스케일링 계수 (실효 스케일 = alpha / dim). 일반적: reft_dim과 동일.",
        "cn": "ReFT 缩放因子（实际缩放 = alpha / dim）。典型值：与 reft_dim 相同。",
    },
    "reft_layers": {
        "en": "Which DiT blocks receive ReFT modules. 'all', 'last_8', 'first_4', 'stride_2', or comma-separated indices like '3,7,11,15'.",
        "ko": "ReFT 모듈이 적용될 DiT 블록. 'all', 'last_8', 'first_4', 'stride_2', 또는 '3,7,11,15'와 같은 쉼표 구분 인덱스.",
        "cn": "哪些 DiT 块接收 ReFT 模块。'all'、'last_8'、'first_4'、'stride_2'，或逗号分隔的索引如 '3,7,11,15'。",
    },
    "sigma_feature_dim": {
        "en": "Sinusoidal σ feature dimension fed into the σ-router bias MLP. Typical: 128.",
        "ko": "σ 라우터 바이어스 MLP에 입력되는 sinusoidal σ 특징 차원. 일반적: 128.",
        "cn": "馈入 σ 路由器偏置 MLP 的正弦 σ 特征维度。典型值：128。",
    },
    "router_targets": {
        "en": "Regex over layer names — only matching Linears participate in routed adaptation (Hydra MoE leaves + σ / FEI feature concatenation share the same scope). Typical: '.*(mlp\\.layer[12])$' to confine MoE to the FFN sublayers.",
        "ko": "레이어 이름에 대한 정규식 — 일치하는 Linear만 라우팅된 적응에 참여 (Hydra MoE leaves + σ / FEI 특징 연결이 동일한 범위를 공유). 일반적: '.*(mlp\\.layer[12])$' — FFN 서브레이어로 MoE 제한.",
        "cn": "层名正则表达式 — 仅匹配的 Linear 层参与路由适配（Hydra MoE 叶节点 + σ / FEI 特征拼接共享同一范围）。典型值：'.*(mlp\\.layer[12])$' 将 MoE 限制在 FFN 子层。",
    },
    "per_bucket_balance_weight": {
        "en": "Extra per-σ-bucket load-balance penalty, scaled by balance_loss_weight. Encourages routing diversity within each timestep bucket. Typical: 0.3.",
        "ko": "σ 버킷별 추가 부하 균형 페널티, balance_loss_weight로 스케일. 각 타임스텝 버킷 내 라우팅 다양성 유도. 일반적: 0.3.",
        "cn": "按 σ 桶的额外负载均衡惩罚，由 balance_loss_weight 缩放。鼓励每个时间步桶内的路由多样性。典型值：0.3。",
    },
    "num_sigma_buckets": {
        "en": "Number of timestep buckets used for per-bucket balance accounting. Typical: 3 (low / mid / high noise).",
        "ko": "버킷별 균형 계산에 사용되는 타임스텝 버킷 수. 일반적: 3 (저/중/고 노이즈).",
        "cn": "用于按桶平衡计算的时间步桶数量。典型值：3（低/中/高噪声）。",
    },
    "specialize_experts_by_sigma_buckets": {
        "en": "Hard-partition the expert pool into σ-bands: each timestep bucket only routes to its assigned experts. Forces specialization on top of the soft σ-router bias. Pairs with sigma_bucket_boundaries.",
        "ko": "전문가 풀을 σ-밴드로 하드 분할: 각 타임스텝 버킷은 할당된 전문가만 사용. 소프트 σ-라우터 바이어스 위에 강제 특화 부여. sigma_bucket_boundaries와 함께 사용.",
        "cn": "将专家池硬划分为 σ 频段：每个时间步桶仅路由到其分配的专家。在软 σ 路由器偏置之上强制专业化。配合 sigma_bucket_boundaries 使用。",
    },
    "sigma_bucket_boundaries": {
        "en": "Custom σ-bucket edges, length = num_sigma_buckets + 1, monotone 0.0 → 1.0. Defaults to uniform linspace(0, 1, N+1) when omitted. Example: [0.0, 0.5, 0.8, 1.0].",
        "ko": "사용자 지정 σ-버킷 경계, 길이 = num_sigma_buckets + 1, 0.0 → 1.0 단조 증가. 생략 시 uniform linspace(0, 1, N+1) 사용. 예: [0.0, 0.5, 0.8, 1.0].",
        "cn": "自定义 σ 桶边界，长度 = num_sigma_buckets + 1，单调递增 0.0 → 1.0。省略时默认均匀 linspace(0, 1, N+1)。示例：[0.0, 0.5, 0.8, 1.0]。",
    },
    "network_args": {
        "en": "Extra kwargs passed to the network module. For postfix: list of 'key=value' strings (e.g., 'mode=cond', 'splice_position=end_of_sequence', 'cond_hidden_dim=256'). Pick a Variant to auto-fill.",
        "ko": "네트워크 모듈에 전달되는 추가 kwargs. postfix의 경우 'key=value' 문자열 리스트 (예: 'mode=cond', 'splice_position=end_of_sequence', 'cond_hidden_dim=256'). Variant 선택으로 자동 채우기 가능.",
        "cn": "传递给网络模块的额外参数。postfix 模式下为 'key=value' 字符串列表（如 'mode=cond'、'splice_position=end_of_sequence'、'cond_hidden_dim=256'）。选择变体可自动填充。",
    },
    "min_rank": {
        "en": "Minimum active rank when T-LoRA timestep masking is enabled. At the lowest-noise timesteps, rank drops to this value.",
        "ko": "T-LoRA 타임스텝 마스킹 사용 시 최소 활성 랭크. 가장 낮은 노이즈에서 이 값까지 감소.",
        "cn": "启用 T-LoRA 时间步掩码时的最小活跃秩。在最低噪声时间步，秩降至该值。",
    },
    "alpha_rank_scale": {
        "en": "Scale alpha proportionally when T-LoRA reduces rank, keeping effective learning rate stable across timesteps.",
        "ko": "T-LoRA가 랭크를 줄일 때 alpha를 비례적으로 조정하여 타임스텝별 실효 학습률 유지.",
        "cn": "T-LoRA 降秩时按比例缩放 alpha，使有效学习率在各时间步保持稳定。",
    },
    "network_train_unet_only": {
        "en": "Train only the DiT (U-Net). Text encoder weights are frozen. Recommended for most LoRA training.",
        "ko": "DiT(U-Net)만 학습. 텍스트 인코더 가중치는 동결. 대부분의 LoRA 학습에 권장.",
        "cn": "仅训练 DiT（U-Net）。文本编码器权重冻结。大多数 LoRA 训练推荐使用。",
    },
    "network_weights": {
        "en": "Path to a pre-trained adapter checkpoint to warm-start from. Leave empty for plain LoRA training.",
        "ko": "워밍업으로 사용할 사전 학습 어댑터 체크포인트 경로. 일반 LoRA 학습 시에는 비워두세요.",
        "cn": "用于热启动的预训练适配器检查点路径。普通 LoRA 训练请留空。",
    },
    "dim_from_weights": {
        "en": "Read network_dim from the warm-start checkpoint instead of the form value. Set together with network_weights so rank matches the warm-start LoRA.",
        "ko": "network_dim을 폼 값 대신 워밍업 체크포인트에서 읽기. network_weights와 함께 설정하여 랭크를 워밍업 LoRA와 일치시킵니다.",
        "cn": "从热启动检查点读取 network_dim 而非表单值。需与 network_weights 一起设置，使秩与热启动 LoRA 匹配。",
    },
    # Training
    "learning_rate": {
        "en": "Base learning rate for the optimizer. Typical: 1e-5 to 1e-4.",
        "ko": "옵티마이저 기본 학습률. 일반적: 1e-5 ~ 1e-4.",
        "cn": "优化器基础学习率。典型值：1e-5 至 1e-4。",
    },
    "max_train_epochs": {
        "en": "Total training epochs. One epoch = one full pass through the dataset.",
        "ko": "총 학습 에폭 수. 1 에폭 = 데이터셋 전체를 1회 순회.",
        "cn": "总训练轮数。一个 epoch = 完整遍历数据集一次。",
    },
    "save_every_n_epochs": {
        "en": "Save a checkpoint every N epochs. Set equal to max_train_epochs to save only the final model.",
        "ko": "N 에폭마다 체크포인트 저장. max_train_epochs와 같게 설정하면 최종 모델만 저장.",
        "cn": "每 N 个 epoch 保存一次检查点。设为与 max_train_epochs 相同则仅保存最终模型。",
    },
    "checkpointing_epochs": {
        "en": "Save resumable training state every N epochs. State files are large; use a larger interval than save_every_n_epochs.",
        "ko": "N 에폭마다 학습 재개 상태 저장. 상태 파일이 크므로 save_every_n_epochs보다 큰 간격 권장.",
        "cn": "每 N 个 epoch 保存可恢复的训练状态。状态文件较大；建议间隔大于 save_every_n_epochs。",
    },
    "gradient_accumulation_steps": {
        "en": "Accumulate gradients over N steps before updating. Effective batch size = batch_size × accumulation_steps.",
        "ko": "N 스텝 동안 그레이디언트 누적 후 업데이트. 실효 배치 크기 = batch_size × accumulation_steps.",
        "cn": "累积 N 步梯度后更新。有效批次大小 = batch_size × accumulation_steps。",
    },
    "use_shuffled_caption_variants": {
        "en": "Consume preprocessed caption-shuffle variants from the text-encoder cache. When the cache holds multiple variants, a random one is drawn per sample. Falls back silently to single-variant if no variants were preprocessed.",
        "ko": "전처리된 캡션 셔플 변형을 텍스트 인코더 캐시에서 사용. 캐시에 여러 변형이 있으면 샘플당 무작위 선택. 변형이 전처리되지 않았다면 단일 캡션으로 자동 대체.",
        "cn": "从文本编码器缓存中使用预处理的标签洗牌变体。缓存含多个变体时，每个样本随机抽取一个。若未预处理变体则静默回退为单标签。",
    },
    "caption_dropout_rate": {
        "en": "Probability per sample of dropping the caption (replaced with empty text embedding). Pushes the LoRA toward an unconditional bias — useful for style training where you want the look to apply regardless of prompt. Typical: 0.0–0.05 for character/concept LoRAs, 0.1–0.25 for style LoRAs. Too high can blur prompt-driven diversity (pose/composition).",
        "ko": "샘플별로 캡션을 비울 확률. LoRA를 무조건부 방향으로 학습시켜, 프롬프트와 무관하게 항상 적용되는 스타일을 학습할 때 유리. 일반적: 캐릭터/컨셉 LoRA는 0.0–0.05, 그림체 학습은 0.1–0.25. 너무 높이면 다양성까지 약해짐.",
        "cn": "每个样本丢弃标签的概率（替换为空文本嵌入）。使 LoRA 偏向无条件生成——适用于风格训练。典型值：角色/概念 LoRA 为 0.0–0.05，画风 LoRA 为 0.1–0.25。过高会削弱提示词驱动的多样性（姿态/构图）。",
    },
    "optimizer_type": {
        "en": "Optimizer algorithm. AdamW8bit: memory-efficient 8-bit Adam. Others: AdamW, Lion, Prodigy, etc.",
        "ko": "옵티마이저 알고리즘. AdamW8bit: 메모리 효율적 8비트 Adam. 기타: AdamW, Lion, Prodigy 등.",
        "cn": "优化器算法。AdamW8bit：内存高效的 8 位 Adam。其他：AdamW、Lion、Prodigy 等。",
    },
    "lr_scheduler": {
        "en": "Learning rate schedule. constant: fixed LR. Others: cosine, cosine_with_restarts, polynomial.",
        "ko": "학습률 스케줄. constant: 고정 LR. 기타: cosine, cosine_with_restarts, polynomial.",
        "cn": "学习率调度策略。constant：固定学习率。其他：cosine、cosine_with_restarts、polynomial。",
    },
    "timestep_sampling": {
        "en": "How denoising timesteps are sampled during training. sigmoid: biased toward middle timesteps (recommended for flow matching).",
        "ko": "학습 중 디노이징 타임스텝 샘플링 방법. sigmoid: 중간 타임스텝 편향 (flow matching 권장).",
        "cn": "训练时的去噪时间步采样方式。sigmoid：偏向中间时间步（flow matching 推荐）。",
    },
    "discrete_flow_shift": {
        "en": "Flow-matching shift parameter controlling the noise schedule distribution. Default: 1.0.",
        "ko": "노이즈 스케줄 분포를 제어하는 flow-matching 시프트 매개변수. 기본값: 1.0.",
        "cn": "控制噪声调度分布的 flow-matching 偏移参数。默认值：1.0。",
    },
    # Performance
    "attn_mode": {
        "en": "Attention backend. flash4: FlashAttention-4 (Linux, fastest). flash: FlashAttention-2. flex: PyTorch flex attention (cross-platform).",
        "ko": "어텐션 백엔드. flash4: FlashAttention-4 (Linux, 최속). flash: FlashAttention-2. flex: PyTorch flex attention (크로스 플랫폼).",
        "cn": "注意力后端。flash4：FlashAttention-4（Linux，最快）。flash：FlashAttention-2。flex：PyTorch flex attention（跨平台）。",
    },
    "gradient_checkpointing": {
        "en": "Recompute activations during backward pass instead of storing them. Trades compute for VRAM. Essential for low-VRAM setups.",
        "ko": "역전파 시 활성값을 저장 대신 재계산. 연산으로 VRAM 절약. 저사양 필수.",
        "cn": "反向传播时重算激活值而非存储。以计算换显存。低显存环境必需。",
    },
    "unsloth_offload_checkpointing": {
        "en": "Offload gradient checkpoints to CPU RAM. Further VRAM reduction at cost of speed. Requires gradient_checkpointing=true.",
        "ko": "그레이디언트 체크포인트를 CPU RAM으로 오프로드. 속도 감소 대신 VRAM 추가 절약. gradient_checkpointing=true 필요.",
        "cn": "将梯度检查点卸载到 CPU 内存。进一步减少显存但降低速度。需要 gradient_checkpointing=true。",
    },
    "blocks_to_swap": {
        "en": "Number of DiT blocks to swap between GPU and CPU. 0: all on GPU. Higher values = more CPU offloading for low VRAM.",
        "ko": "GPU와 CPU 간 스왑할 DiT 블록 수. 0: 전부 GPU. 높을수록 더 많이 CPU로 오프로드.",
        "cn": "在 GPU 和 CPU 之间交换的 DiT 块数。0：全部在 GPU 上。值越大，CPU 卸载越多（适用于低显存）。",
    },
    "torch_compile": {
        "en": "Enable torch.compile for the forward pass. Faster training after initial compilation. Best with static_token_count=true.",
        "ko": "torch.compile 활성화. 초기 컴파일 후 학습 속도 향상. static_token_count=true와 함께 사용 권장.",
        "cn": "启用 torch.compile 加速前向传播。初始编译后训练更快。配合 static_token_count=true 效果最佳。",
    },
    "compile_mode": {
        "en": "'blocks': compile each DiT block individually (default). 'full': compile entire model as one graph for cross-block memory optimization. Full mode is incompatible with gradient checkpointing and block swap.",
        "ko": "'blocks': 각 DiT 블록을 개별 컴파일 (기본값). 'full': 전체 모델을 하나의 그래프로 컴파일하여 블록 간 메모리 최적화. full 모드는 gradient checkpointing 및 block swap과 호환 불가.",
        "cn": "'blocks'：逐块编译每个 DiT 块（默认）。'full'：将整个模型编译为单图以实现跨块内存优化。full 模式与梯度检查点和块交换不兼容。",
    },
    "trim_crossattn_kv": {
        "en": "Remove zero-padding from cross-attention KV for efficiency. Flash4 applies LSE correction to maintain correct softmax.",
        "ko": "효율을 위해 크로스 어텐션 KV에서 제로 패딩 제거. Flash4는 정확한 softmax를 위해 LSE 보정 적용.",
        "cn": "移除交叉注意力 KV 中的零填充以提升效率。Flash4 应用 LSE 校正以保持正确的 softmax。",
    },
    "cache_llm_adapter_outputs": {
        "en": "Cache the LLM adapter layer outputs to disk. Avoids recomputing text encoder projections each epoch.",
        "ko": "LLM 어댑터 레이어 출력을 디스크에 캐싱. 매 에폭 텍스트 인코더 투영 재계산 회피.",
        "cn": "将 LLM 适配器层输出缓存到磁盘。避免每个 epoch 重新计算文本编码器投影。",
    },
    "masked_loss": {
        "en": "Apply loss only to non-masked regions (e.g., exclude text bubbles). Requires mask files in post_image_dataset/masks/ (or legacy masks/merged/) — run `make mask` to generate them.",
        "ko": "마스크되지 않은 영역에만 손실 적용 (예: 말풍선 제외). post_image_dataset/masks/ 에 마스크 파일 필요 — `make mask` 로 생성.",
        "cn": "仅对非遮罩区域计算损失（如排除文字气泡）。需要 post_image_dataset/masks/ 下的遮罩文件——运行 `make mask` 生成。",
    },
    "mixed_precision": {
        "en": "Mixed precision mode. bf16: recommended for modern GPUs. fp16: for older GPUs without bf16 support.",
        "ko": "혼합 정밀도 모드. bf16: 최신 GPU 권장. fp16: bf16 미지원 구형 GPU용.",
        "cn": "混合精度模式。bf16：推荐用于现代 GPU。fp16：适用于不支持 bf16 的旧 GPU。",
    },
    "static_token_count": {
        "en": "Fixed 4096 token count for all batches. Gives torch.compile a single static shape — no recompilation across aspect ratios.",
        "ko": "모든 배치에 4096 토큰 고정. torch.compile에 단일 정적 셰이프 제공 — 화면비별 재컴파일 없음.",
        "cn": "所有批次固定 4096 个 token。为 torch.compile 提供单一静态形状——不同宽高比无需重新编译。",
    },
    "vae_chunk_size": {
        "en": "VAE decoding chunk size. Larger = faster but more VRAM. 64 is a good balance.",
        "ko": "VAE 디코딩 청크 크기. 클수록 빠르지만 VRAM 더 사용. 64가 적절.",
        "cn": "VAE 解码分块大小。越大越快但显存占用越高。64 是较好的平衡点。",
    },
    "vae_disable_cache": {
        "en": "Disable VAE's internal KV cache. Reduces VRAM during VAE encoding/decoding.",
        "ko": "VAE 내부 KV 캐시 비활성화. VAE 인코딩/디코딩 시 VRAM 감소.",
        "cn": "禁用 VAE 内部 KV 缓存。减少 VAE 编码/解码时的显存占用。",
    },
    "cache_latents": {
        "en": "Cache VAE-encoded latents in memory. Avoids re-encoding images every epoch.",
        "ko": "VAE 인코딩된 레이턴트를 메모리에 캐싱. 매 에폭 이미지 재인코딩 회피.",
        "cn": "将 VAE 编码的潜变量缓存到内存中。避免每个 epoch 重新编码图片。",
    },
    "cache_latents_to_disk": {
        "en": "Save cached latents to disk instead of RAM. Frees system memory at cost of disk I/O.",
        "ko": "캐시된 레이턴트를 RAM 대신 디스크에 저장. 디스크 I/O 대신 시스템 메모리 절약.",
        "cn": "将缓存的潜变量保存到磁盘而非 RAM。以磁盘 I/O 换取系统内存释放。",
    },
    "cache_text_encoder_outputs": {
        "en": "Cache text encoder outputs. Essential for lazy loading: encode → cache → free encoder → load DiT.",
        "ko": "텍스트 인코더 출력 캐싱. 지연 로딩 필수: 인코딩 → 캐시 → 인코더 해제 → DiT 로드.",
        "cn": "缓存文本编码器输出。延迟加载的关键：编码 → 缓存 → 释放编码器 → 加载 DiT。",
    },
    "cache_text_encoder_outputs_to_disk": {
        "en": "Save cached text encoder outputs to disk. Required for the lazy loading sequence to free VRAM before loading DiT.",
        "ko": "캐시된 텍스트 인코더 출력을 디스크에 저장. DiT 로드 전 VRAM 해제를 위한 지연 로딩 필수.",
        "cn": "将缓存的文本编码器输出保存到磁盘。延迟加载序列中加载 DiT 前释放显存所必需。",
    },
    "skip_cache_check": {
        "en": "Skip validation of cached files on startup. Faster startup when caches are known to be valid.",
        "ko": "시작 시 캐시 파일 검증 건너뛰기. 캐시가 유효함을 알 때 빠른 시작.",
        "cn": "启动时跳过缓存文件验证。当缓存已知有效时可加快启动速度。",
    },
    "use_cmmd": {
        "en": "Use CMMD (PE-Core MMD²) as the validation signal. Off by default in the GUI — CMMD adds the PE encoder + a sampling pass per held-out item, which costs extra VRAM and time. Off → falls back to the cheaper per-σ FM-MSE val pass (uninformative on Anima but free).",
        "ko": "CMMD (PE-Core MMD²)를 검증 신호로 사용. GUI 기본값은 OFF — CMMD는 VRAM과 시간 비용이 큼. OFF면 더 저렴한 σ별 FM-MSE 검증으로 대체.",
        "cn": "使用 CMMD（PE-Core MMD²）作为验证信号。GUI 默认关闭——CMMD 需要 PE 编码器 + 每个验证项的采样，额外消耗显存和时间。关闭时回退到更轻量的 σ 级 FM-MSE 验证（对 Anima 信号不明显但无成本）。",
    },
    "use_valid": {
        "en": "Hold out a small validation slice from the training set (16 images by default). When off, the whole pool is used for training and no validation pass runs. Writes/strips a {validation_split_num = 0, validation_split = 0.0} override on the variant's [[datasets]] block; base.toml is not touched.",
        "ko": "학습 셋에서 검증용 일부(기본 16장)를 분리. 끄면 전체 풀이 학습에 쓰이고 검증 패스는 실행되지 않음. 변환 파일의 [[datasets]] 블록에 오버라이드를 쓰거나 제거함; base.toml은 건드리지 않음.",
        "cn": "从训练集中分出一小部分作为验证集（默认 16 张）。关闭时全部用于训练且不执行验证。在变体的 [[datasets]] 块中写入/移除 {validation_split_num = 0, validation_split = 0.0} 覆盖；不影响 base.toml。",
    },
    "validation_split_num": {
        "en": "How many images to hold out for validation when 'use_valid' is on. Base.toml ships 16 — large enough for stable paired CMMD, small enough to barely dent the train pool. Bigger values give a quieter CMMD curve at the cost of fewer training samples. Ignored when 'use_valid' is off.",
        "ko": "'use_valid'가 켜져 있을 때 검증용으로 분리할 이미지 수. base.toml 기본값은 16. 값을 늘리면 CMMD 곡선이 매끄러워지지만 학습 샘플이 줄어듦. 'use_valid'가 꺼져 있으면 무시됨.",
        "cn": "'use_valid' 开启时用于验证的图片数量。base.toml 默认 16——足以稳定测量 paired CMMD，又几乎不影响训练池。增大可使 CMMD 曲线更平滑，但减少训练样本。'use_valid' 关闭时忽略。",
    },
    "batch_size": {
        "en": "Number of images per training micro-step. Effective batch = batch_size × gradient_accumulation_steps × num_processes. Higher values use more VRAM; start with 1 on 8 GB cards.",
        "ko": "학습 마이크로스텝당 이미지 수. 실효 배치 = batch_size × gradient_accumulation_steps × num_processes. 값이 클수록 VRAM 사용량 증가; 8GB 카드에서는 1부터 시작.",
        "cn": "每个训练微步的图片数量。有效批次 = batch_size × gradient_accumulation_steps × num_processes。值越大显存占用越高；8 GB 显卡建议从 1 开始。",
    },
    # Paths
    "pretrained_model_name_or_path": {
        "en": "Path to the base DiT model weights (.safetensors).",
        "ko": "기본 DiT 모델 가중치 경로 (.safetensors).",
        "cn": "基础 DiT 模型权重路径（.safetensors）。",
    },
    "qwen3": {
        "en": "Path to the Qwen3 text encoder weights for text-to-image conditioning.",
        "ko": "텍스트-투-이미지 컨디셔닝용 Qwen3 텍스트 인코더 가중치 경로.",
        "cn": "用于文生图条件输入的 Qwen3 文本编码器权重路径。",
    },
    "vae": {
        "en": "Path to the VAE model for image encoding/decoding.",
        "ko": "이미지 인코딩/디코딩용 VAE 모델 경로.",
        "cn": "用于图片编解码的 VAE 模型路径。",
    },
    "output_dir": {
        "en": "Directory for saving trained LoRA checkpoints.",
        "ko": "학습된 LoRA 체크포인트 저장 디렉토리.",
        "cn": "保存训练好的 LoRA 检查点的目录。",
    },
    "output_name": {
        "en": "Base filename for saved checkpoints (epoch number is appended automatically).",
        "ko": "저장되는 체크포인트의 기본 파일명 (에폭 번호 자동 추가).",
        "cn": "保存检查点的基础文件名（epoch 编号自动追加）。",
    },
    "save_model_as": {
        "en": "Checkpoint format. safetensors: recommended (fast, safe).",
        "ko": "체크포인트 형식. safetensors: 권장 (빠르고 안전).",
        "cn": "检查点格式。safetensors：推荐（快速、安全）。",
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
        "cn": (
            "原始图片和 .txt 标签所在目录。预处理按钮将此路径传给 "
            "resize_images.py（输出缩放后的 PNG）和 cache_text_embeddings.py"
            "（缓存标签）。如需并行使用多个数据集，可按预设/方法覆盖。"
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
        "cn": (
            "预处理写入 VAE 对齐 PNG 的目录。训练时也会解析为数据集子集的 "
            "image_dir（通过 base.toml 的 {resized_image_dir} 模板替换），"
            "因此修改此值会同时影响预处理和训练。"
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
        "cn": (
            "预处理写入 VAE 潜变量（.npz）和文本编码器"
            "（_anima_te.safetensors）缓存的目录。训练时也会解析为数据集子集的 "
            "cache_dir。"
        ),
    },
    "path_pattern": {
        "en": (
            "fnmatch glob applied to each image's path relative to its subset's "
            "image_dir. Lets one variant TOML train on a slice of the full "
            "dataset without re-running preprocessing. `|` separates "
            "alternatives — OR-combine multiple patterns. Examples: `*` keeps "
            "everything; `char_a/*` keeps only files under the char_a/ "
            "subfolder; `char_a/*|char_b/*` keeps either folder; "
            "`*portrait*` keeps anything whose path contains 'portrait'. "
            "Validation enumeration and the image-count threshold honour the "
            "filtered pool."
        ),
        "ko": (
            "각 이미지의 image_dir 기준 상대 경로에 적용되는 fnmatch 글롭. "
            "전처리를 다시 돌리지 않고도 하나의 variant TOML로 전체 데이터셋의 "
            "일부만 학습할 수 있습니다. `|`로 여러 패턴을 OR 결합할 수 있습니다. "
            "예: `*`는 전체 사용, `char_a/*`는 char_a/ 하위만, "
            "`char_a/*|char_b/*`는 두 폴더 모두, `*portrait*`는 경로에 "
            "'portrait'이 포함된 파일만. 검증 데이터 열거와 이미지 개수 "
            "임계값도 필터링된 풀을 기준으로 동작합니다."
        ),
        "cn": (
            "应用于每张图片相对于其子集 image_dir 的路径的 fnmatch 通配符。"
            "无需重新预处理即可让单个变体 TOML 训练数据集的子集。`|` 分隔"
            "多个模式进行 OR 组合。示例：`*` 保留全部；`char_a/*` 仅保留 "
            "char_a/ 子目录下文件；`char_a/*|char_b/*` 保留两个目录；"
            "`*portrait*` 保留路径含 'portrait' 的文件。验证枚举和图片数量"
            "阈值均基于筛选后的数据池。"
        ),
    },
    "drop_lowres_images": {
        "en": (
            "When true, the preprocess auto-chain skips source images below "
            "the `min_pixels` threshold so they never enter the resize / VAE "
            "/ TE caches. Applied to both preprocess/resize_images.py and "
            "preprocess/cache_text_embeddings.py via the same `--min_pixels` "
            "argument. Uncheck to keep every image regardless of size."
        ),
        "ko": (
            "체크 시, 학습 자동 체인의 전처리가 `min_pixels` 임계값 미만의 "
            "원본 이미지를 건너뛰어 리사이즈 / VAE / TE 캐시에 들어가지 "
            "않도록 합니다. preprocess/resize_images.py와 "
            "preprocess/cache_text_embeddings.py 양쪽에 동일한 "
            "`--min_pixels` 인자로 전달됩니다. 체크 해제하면 크기와 무관하게 "
            "모든 이미지를 사용합니다."
        ),
        "cn": (
            "为 true 时，预处理自动链会跳过低于 `min_pixels` 阈值的原始图片，"
            "使其不进入缩放 / VAE / TE 缓存。通过相同的 `--min_pixels` 参数"
            "应用于 preprocess/resize_images.py 和 "
            "preprocess/cache_text_embeddings.py。取消勾选则无论大小全部保留。"
        ),
    },
    "min_pixels": {
        "en": (
            "Pixel-count threshold used by `drop_lowres_images` (default "
            "500_000 = 0.5MP). Forwarded verbatim to "
            "preprocess/resize_images.py and cache_text_embeddings.py as "
            "`--min_pixels`. Ignored when `drop_lowres_images = false`. "
            "Set to 0 (with the flag on) to disable the filter without "
            "flipping the checkbox."
        ),
        "ko": (
            "`drop_lowres_images`에서 사용하는 픽셀 수 임계값 (기본값 "
            "500_000 = 0.5MP). preprocess/resize_images.py와 "
            "cache_text_embeddings.py에 `--min_pixels`로 그대로 전달됩니다. "
            "`drop_lowres_images = false`일 때는 무시됩니다. 체크박스를 "
            "끄지 않고 필터만 비활성화하려면 0으로 설정하세요."
        ),
        "cn": (
            "`drop_lowres_images` 使用的像素数阈值（默认 500_000 = 0.5MP）。"
            "原样转发给 preprocess/resize_images.py 和 cache_text_embeddings.py "
            "的 `--min_pixels` 参数。`drop_lowres_images = false` 时忽略。"
            "设为 0（保持开关开启）可在不关闭开关的情况下禁用过滤。"
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


def field_help_en(key: str) -> str | None:
    """Return the English help string for *key*, or None."""
    entry = FIELD_HELP.get(key)
    return entry.get("en") if entry else None


def field_help_cn(key: str) -> str | None:
    """Return the Chinese help string for *key*, or None."""
    entry = FIELD_HELP.get(key)
    return entry.get("cn") if entry else None


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
        "cn": (
            "文本编码器缓存时每张图片生成的标签变体数量。v0 为原始标签；"
            "v1..v(N-1) 为智能洗牌版本（保留 @artist 前缀和 'On the …' / "
            "'In the …' 锚点段落）。数据加载器以 20% 概率选 v0，其余均匀"
            "选 v1..v(N-1)——仅当方法配置中 use_shuffled_caption_variants=true"
            "时生效。设为 0 则仅缓存原始标签。"
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
        "cn": (
            "仅对 v1..v(N-1) 洗牌变体应用的逐标签丢弃概率。第一个 @artist "
            "标记及之前的标签永不丢弃——画师标签结构重要，且评级/角色/作品"
            "前缀顺序敏感。洗牌变体数 ≤ 0 时忽略。典型值：0.05–0.15。"
            "值越高 LoRA 对缺失标签的泛化能力越强，但可能稀释学习信号。"
        ),
    },
    "run_sam_mask": {
        "en": (
            "Whether to run SAM3 bubble segmentation during 'Run masking'. "
            "Uncheck to skip SAM entirely — useful when SAM produces false "
            "positives on your dataset or when you only want MIT's text "
            "detection. If both SAM and MIT are unchecked, the Run button "
            "is blocked."
        ),
        "ko": (
            "'마스킹 실행' 시 SAM3 말풍선 분할을 실행할지 여부. 체크 해제 "
            "하면 SAM을 완전히 건너뜁니다 — 데이터셋에서 SAM이 오탐을 "
            "많이 내거나 MIT의 텍스트 검출만 원할 때 유용합니다. SAM과 "
            "MIT 둘 다 해제하면 실행 버튼이 차단됩니다."
        ),
        "cn": (
            "是否在“运行遮罩”时执行 SAM3 气泡分割。取消勾选可完全跳过 "
            "SAM——当 SAM 在数据集上产生较多误检或只需 MIT 文字检测时有用。"
            "SAM 和 MIT 均取消时，运行按钮将被禁用。"
        ),
    },
    "run_mit_mask": {
        "en": (
            "Whether to run MIT/ComicTextDetector text segmentation during "
            "'Run masking'. Uncheck to skip MIT — useful for non-manga "
            "datasets where its prior doesn't help, or to iterate faster on "
            "SAM-only configurations. If both SAM and MIT are unchecked, "
            "the Run button is blocked."
        ),
        "ko": (
            "'마스킹 실행' 시 MIT/ComicTextDetector 텍스트 분할을 "
            "실행할지 여부. 체크 해제하면 MIT를 건너뜁니다 — 만화가 아닌 "
            "데이터셋이나 SAM 전용 구성을 빠르게 시험할 때 유용합니다. "
            "SAM과 MIT 둘 다 해제하면 실행 버튼이 차단됩니다."
        ),
        "cn": (
            "是否在“运行遮罩”时执行 MIT/ComicTextDetector 文字分割。"
            "取消可跳过 MIT——适用于非漫画数据集（其先验无帮助）或仅用 "
            "SAM 快速迭代。SAM 和 MIT 均取消时，运行按钮将被禁用。"
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
        "cn": (
            "SAM3 在每张图片中查找的文本提示。每行一个。默认值针对漫画风格"
            "对话框/文字气泡调优。如数据集有其他需遮罩的区域（如音效、"
            "说明框），可添加自定义提示。保存至 configs/sam_mask.yaml——"
            "由 preprocess/generate_masks.py 直接读取。"
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
        "cn": (
            "保留 SAM3 检测结果的最低置信度。范围 0.0–1.0。值越低产生更多"
            "遮罩（误检也更多）；值越高越严格。默认 0.5。若 SAM 漏检真实"
            "气泡则降低；若遮罩了无关区域则提高。"
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
        "cn": (
            "阈值处理后对每个 SAM 遮罩应用的二值膨胀像素数。值越大遮罩边缘"
            "向外扩展——当分割结果小于实际文字气泡边界时有用。默认 5。"
            "设为 0 禁用膨胀。"
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
        "cn": (
            "MIT/ComicTextDetector 文字分割器的置信度阈值。范围 0.0–1.0。"
            "与 SAM 阈值独立——MIT 使用专门在漫画文字区域训练的不同模型。"
            "默认 0.8。若 MIT 漏检面板内文字则降低；若捕获非文字伪影则提高。"
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
        "cn": (
            "对每个 MIT 遮罩应用的二值膨胀像素数。作用与 SAM 膨胀相同但"
            "独立调节——MIT 通常围绕单个字形分割出紧凑边界，适度的膨胀值"
            "可将其合并为按气泡的连通区域。默认 5。"
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


def preprocess_guide() -> str:
    return _guide("preprocess")


# ── Method guide dispatch ─────────────────────────────────────
# Methods that can't be baked into a plain DiT via scripts/merge_to_dit.py
# (router is layer-local / hook-only / not a weight delta) — render the
# "not mergeable" callout above their guide.
_NOT_MERGEABLE = frozenset({"postfix", "hydralora", "reft", "fera"})
_KNOWN_METHODS = frozenset({"lora", "tlora", "postfix", "hydralora", "reft", "fera"})


def method_guide(method: str) -> str | None:
    """Right-panel default HTML for *method*, or None if no guide is registered."""
    if method not in _KNOWN_METHODS:
        return None
    parts = [_guide("_apply_note")]
    if method in _NOT_MERGEABLE:
        parts.append(_guide("_not_mergeable"))
    parts.append(_guide(method))
    return "".join(parts)
