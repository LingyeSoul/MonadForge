"""en / cn strings for the MOSS-Image GUI, over ``gui.lines.strings``.

Field labels/help: English falls back to the request field's own ``help``
metadata (``library.moss.requests``); Chinese covers the basic (non-advanced)
fields, the rest fall back to English.
"""

from __future__ import annotations

# Line-specific keys over the shared table in ``gui.lines.strings``.
UI: dict[str, dict[str, str]] = {
    "en": {
        "window_title": "MOSS-Image LoRA (alpha)",
        "model_missing": "Model not found: {path}\nRun `make download-model ARGS=moss-base` "
        "(ARGS=moss-sft for the SFT DiT), set Advanced → model_dir, or set "
        "ANIMA_MOSS_MODEL_DIR in .env.",
        "scan_stale": "{stale} captions changed since their text cache — the next "
        "preprocessing run re-encodes them.",
        "scan_dupes": "{n} images share a file name with one in another subfolder — "
        "their cache files are prefixed with the folder name.",
        "tab_i2i": "i2i",
        "i2i_task": "Task",
        "i2i_no_tasks": "No i2i task configs under configs/moss/i2i/.",
        "i2i_guide": "An i2i LoRA learns to turn a reference image into a target: the "
        "reference rides every forward as clean latent tokens. Stage builds the "
        "pairs, Preprocess caches each target with its reference (and caption), Train "
        "fits the LoRA, Test renders new references with it. Each form starts from "
        "configs/moss/i2i/<task>/; only your changes are saved.",
        "i2i_task_colorize": "colorize: B&W manga page → colour. Staging mangafies the "
        "colour corpus; the page enters the text encoder through its vision tower, "
        "with its OCR'd lines in the caption, and as a half-size reference. Test with "
        "an empty prompt: the page alone.",
        "i2i_task_sanitize": "sanitize: removes text and speech bubbles. Staging mines "
        "in-artist near-twin pairs (texted / clean); the texted page is a full-size "
        "reference. Test with the page's caption minus its text tags.",
        "i2i_stage_group": "1 · Stage",
        "i2i_stage_hint": "Builds the task's pair tree under post_image_dataset/i2i/ "
        "from the dataset (make i2i-stage TASK={task}; knobs in the [staging] table "
        "of configs/easycontrol/{task}.toml). EasyControl reads the same tree.",
        "run_stage": "Stage",
        "i2i_cache_group": "2 · Preprocess",
        "i2i_train_group": "3 · Train",
        "i2i_chain": "Stage, preprocess, then train",
        "i2i_test_group": "4 · Test",
        "i2i_test_hint": "Renders each reference with the LoRA, shown beside it. "
        "ref_image is one image or a folder of images; ref_scale and ref_vision must "
        "match the cache the LoRA trained on.",
        "i2i_staged": "{n} staged pairs.",
        "i2i_not_staged": "Not staged yet: {path} is missing. Run Stage first.",
        "i2i_stage_next": "Staging done — preprocessing.",
        "i2i_need_ref": "Set ref_image to an image or a folder of images.",
        "i2i_need_prompt": "With ref_vision off, the test needs a prompt: the page's "
        "caption.",
        "test_ref": "reference",
    },
    "cn": {
        "window_title": "MOSS-Image LoRA（alpha）",
        "model_missing": "找不到模型：{path}\n请运行 `make download-model ARGS=moss-base`"
        "（SFT 模型另需 ARGS=moss-sft），或在「高级 → model_dir」中设置，"
        "或在 .env 中设置 ANIMA_MOSS_MODEL_DIR。",
        "scan_stale": "{stale} 条标注在文本缓存之后被修改——下次预处理会自动重新编码。",
        "scan_dupes": "有 {n} 张图片与其他子文件夹中的图片同名——"
        "它们的缓存文件名会加上文件夹前缀。",
        "tab_i2i": "图生图",
        "i2i_task": "任务",
        "i2i_no_tasks": "configs/moss/i2i/ 下没有图生图任务配置。",
        "i2i_guide": "图生图 LoRA 学习把参考图变成目标图：参考图以干净的潜变量 token "
        "参与每次前向。「准备配对」生成配对，「预处理」缓存每个目标及其参考图（和标注），"
        "「训练」拟合 LoRA，「测试」用它处理新的参考图。各表单从 "
        "configs/moss/i2i/<任务>/ 开始，只保存你的修改。",
        "i2i_task_colorize": "colorize：黑白漫画页 → 彩色。准备配对时把彩色语料漫画化；"
        "漫画页经视觉塔进入文本编码器（标注中带有 OCR 识别出的文字行），"
        "并作为半尺寸参考图。测试时提示词留空：只输入漫画页。",
        "i2i_task_sanitize": "sanitize：去除文字和对话框。准备配对时挖掘同一画师的"
        "近似图对（有字 / 无字）；有字的页面作为全尺寸参考图。"
        "测试时提示词用该页的标注，去掉文字相关标签。",
        "i2i_stage_group": "1 · 准备配对",
        "i2i_stage_hint": "从数据集在 post_image_dataset/i2i/ 下生成该任务的配对目录"
        "（make i2i-stage TASK={task}；参数在 configs/easycontrol/{task}.toml 的 "
        "[staging] 表中）。EasyControl 读取同一目录。",
        "run_stage": "准备配对",
        "i2i_cache_group": "2 · 预处理",
        "i2i_train_group": "3 · 训练",
        "i2i_chain": "准备配对、预处理后训练",
        "i2i_test_group": "4 · 测试",
        "i2i_test_hint": "用 LoRA 处理每张参考图，并与参考图并排显示。ref_image "
        "为一张图片或图片文件夹；ref_scale 和 ref_vision 须与 LoRA 训练所用的缓存一致。",
        "i2i_staged": "已准备 {n} 个配对。",
        "i2i_not_staged": "尚未准备配对：找不到 {path}。请先运行「准备配对」。",
        "i2i_stage_next": "配对准备完成——开始预处理。",
        "i2i_need_ref": "请把 ref_image 设为一张图片或图片文件夹。",
        "i2i_need_prompt": "ref_vision 关闭时，测试需要提示词：该页的标注。",
        "test_ref": "参考图",
    },
}

# Field labels + help, keyed by request field name. English is not listed:
# its label is the field name and its help the request's own metadata.
FIELDS_CN: dict[str, tuple[str, str]] = {
    "src": (
        "图片文件夹",
        "包含图片和同名 .txt 标注的文件夹，包括子文件夹"
        "（默认：Anima 的 resized 目录，其中是修订后的标注；"
        "该目录中只有标注、没有图片时，使用 image_dataset/ 中的原图）",
    ),
    "out": ("缓存文件夹", "每张图片写入两个缓存文件（文本嵌入 + 潜变量）"),
    "resolution": (
        "分辨率",
        "像素面积上限的边长——1024 表示按图片原始比例最多约 1024² 像素，"
        "两边均为 16 的倍数；不会放大",
    ),
    "caption_format": (
        "标注格式",
        "文本编码器编码的内容：layout_json = 把 Anima 标注映射为 MOSS 的 layout JSON 提示词"
        "（需要 danbooru 标签 CSV），text = 原样使用 .txt。生成时请使用相同格式",
    ),
    "overwrite": ("覆盖", "即使缓存是最新的也重新编码"),
    "skip_text": ("跳过文本", "跳过文本编码器"),
    "skip_latents": ("跳过潜变量", "跳过 VAE 编码"),
    "model_dir": (
        "模型文件夹",
        "包含 MOSS-Image-Base/ 和 MOSS-Image-SFT/ 的文件夹"
        "（默认：$ANIMA_MOSS_MODEL_DIR 或 models/moss_image）",
    ),
    "cache": ("缓存文件夹", "预处理写入的缓存文件夹"),
    "output": ("输出", "LoRA 输出路径（同目录下会写入 .json 运行报告）"),
    "model": ("模型", "使用哪个 DiT（base / sft）"),
    "epochs": ("轮数", "遍历缓存的次数"),
    "max_steps": ("最大步数", "达到此步数后停止（0 = 由轮数决定）"),
    "algo": (
        "适配器",
        "lora，或 LyCORIS 的 lokr（Kronecker 积）/ loha（两组秩 r 矩阵的 Hadamard 积）",
    ),
    "rank": ("秩", "LoRA 秩（lokr：较大 Kronecker 因子的秩）"),
    "alpha": (
        "alpha",
        "LoRA alpha（lora 默认 128，lokr / loha 默认等于秩）；缩放 = alpha / 秩",
    ),
    "factor": (
        "lokr 因子",
        "仅 lokr：较小 Kronecker 因子的边长上限（-1 = 接近平方的拆分）",
    ),
    "down_init": (
        "down 初始化",
        "仅 lora：kaiming = 默认随机初始化；weight_svd = 用基础权重前 r 个右奇异向量 / √3 初始化（Anima 的 SVD-Down）",
    ),
    "channel_scaling_alpha": (
        "通道缩放 α",
        "仅 lora：按 mean|x[c]|^α 重新平衡各输入通道的 AdamW 步长（Anima 的 channel scaling）；"
        "输出不变，保存为普通 LoRA。0 = 关闭，0.5 = Anima 默认",
    ),
    "dc_isolate": (
        "DC 隔离",
        "仅 lora：把校准得到的输入直流方向（各目标的 token 均值 x̄）放进 down 的一行，"
        "其余行保持与它正交，使只有一列 up 学习与 token 无关的偏置。初始输出不变，"
        "保存为普通 LoRA。默认开启（配合 lr 5e-5）",
    ),
    "lr": ("学习率", "AdamW 学习率"),
    "warmup_ratio": ("预热比例", "线性预热占总步数的比例"),
    "grad_accum": (
        "梯度累积",
        "每次优化器更新累积的样本数；步数、轮数和损失日志仍按样本计，预热按更新计",
    ),
    "save_every_epochs": ("每 N 轮保存", "每 N 轮额外保存一次（0 = 只保存最终结果）"),
    "caption_dropout": (
        "标注丢弃率",
        "以空提示词嵌入（CFG 的无条件分支）训练一步的概率",
    ),
    "plain_caption_rate": (
        "Anima 标注比例",
        "以原始 Anima 格式标注代替其 layout JSON 训练一步的概率；"
        "仅 caption_format=layout_json 的缓存带有原始标注",
    ),
    "lora_dtype": (
        "LoRA 精度",
        "保存的 LoRA 文件精度。训练时保持 fp32 主权重，秩矩阵乘法按 bf16 进行",
    ),
    "grad_checkpointing": ("梯度检查点", "按块的激活检查点"),
    "seed": ("随机种子", "随机数种子"),
    "n": ("插画配对数", "用于训练的插画配对数（准备好的漫画化页 + resized 中的彩色目标）"),
    "manga": ("漫画页数", "追加的 manga_color 漫画页数（0 = 只用插画）"),
    "holdout": ("留出插画数", "留作评估的插画配对数"),
    "manga_holdout": ("留出漫画作品数", "留作评估的漫画作品数，每部一页；其任何版本都不参与训练"),
    "ref_scale": (
        "参考图缩放",
        "参考潜变量的边长 = 目标边长除以此值，取 16 的倍数（生成时用相同的值）",
    ),
    "variants": (
        "标注变体数",
        "稀疏标注数 + 1：embeds_v1 … embeds_v{variants-1} 为去掉部分标签的颜色标注（至少 2）",
    ),
    "tag_dropout": ("标签丢弃比例", "稀疏标注去掉的颜色标注标签比例"),
}

# GenerateRequest fields whose name also exists on another request with a
# different meaning.
FIELDS_CN_GENERATE: dict[str, tuple[str, str]] = {
    "prompt": ("提示词", "提示词（设置 prompts_file 或 example 时忽略）"),
    "negative_prompt": ("反向提示词", "反向提示词（管线默认为空）"),
    "prompt_format": (
        "提示词格式",
        "layout_json = 把 Anima 格式的提示词映射为 MOSS 的 layout JSON（与缓存的标注格式一致）；"
        "text = 原样使用，适用于已是 JSON 的提示词或以 text 格式缓存训练的 LoRA",
    ),
    "lora": (
        "LoRA",
        "要测试的 LoRA（留空 = 训练输出）；Base 上训练的 LoRA 也可用于 SFT",
    ),
    "lora_multiplier": ("LoRA 强度", "LoRA 强度；0.0 即基础模型"),
    "width": ("宽度", "16 的倍数"),
    "height": ("高度", "16 的倍数"),
    "steps": ("步数", "去噪步数"),
    "cfg": ("CFG", "CFG 强度；1.0 = 不做无条件前向"),
    "sampler": (
        "采样器",
        "er_sde = 同一 σ 网格上的 ER-SDE 随机采样（线条更锐利，不增加前向）；euler = 管线默认的 ODE",
    ),
    "uncond_skip": (
        "无条件分支跳过的块",
        "逗号分隔的 DiT 块编号（'' = 普通 CFG）；管线默认跳过第 22 块",
    ),
    "seed": ("随机种子", "第一个提示词的种子（每个提示词 +1）"),
    "out_dir": ("输出文件夹", "图片和 manifest.json 保存在这里"),
    "spectrum": ("Spectrum", "大部分步数预测最后一个块的输出，而不是运行全部 42 个块"),
    "dave": (
        "DAVE",
        "同提示词多样性：前几步在一组块上削弱图像 token 的逐通道均值。"
        "'' = 关闭，'auto' = 内置 MOSS 掩码，'lo-hi' = 第 lo..hi 块，或 dave_alpha.npz 路径",
    ),
    "dave_strength": ("DAVE 强度", "在池内块上去除的均值比例（0 = 关闭，1 = 全部）"),
    "ref_image": (
        "参考图",
        "图生图参考：一张图片，或图片文件夹（每张图片 × 每个提示词一个任务）。"
        "输出保持参考图比例，面积上限为 宽 × 高；LoRA 中记录的 ref_timestep / ref_isolate 会生效",
    ),
    "ref_scale": (
        "参考图缩放",
        "设置 ref_image 时：参考潜变量的边长 = 输出边长除以此值，取 16 的倍数。"
        "须与 LoRA 训练所用的缓存一致（sanitize 为 1，colorize 为 2）",
    ),
    "ref_vision": (
        "参考图进视觉塔",
        "设置 ref_image 时：参考图同时经视觉塔进入文本编码器，位于提示词之前，"
        "与 colorize 缓存的编码方式一致。此时提示词留空即只输入该页",
    ),
}
