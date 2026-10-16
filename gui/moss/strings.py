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
    },
    "cn": {
        "window_title": "MOSS-Image LoRA（alpha）",
        "model_missing": "找不到模型：{path}\n请运行 `make download-model ARGS=moss-base`"
        "（SFT 模型另需 ARGS=moss-sft），或在「高级 → model_dir」中设置，"
        "或在 .env 中设置 ANIMA_MOSS_MODEL_DIR。",
        "scan_stale": "{stale} 条标注在文本缓存之后被修改——下次预处理会自动重新编码。",
        "scan_dupes": "有 {n} 张图片与其他子文件夹中的图片同名——"
        "它们的缓存文件名会加上文件夹前缀。",
    },
}

# Field labels + help, keyed by request field name. English is not listed:
# its label is the field name and its help the request's own metadata.
FIELDS_CN: dict[str, tuple[str, str]] = {
    "src": (
        "图片文件夹",
        "包含图片和同名 .txt 标注的文件夹，包括子文件夹"
        "（默认：Anima 的 resized 目录，其中是修订后的标注）",
    ),
    "out": ("缓存文件夹", "每张图片写入两个缓存文件（文本嵌入 + 潜变量）"),
    "resolution": (
        "分辨率",
        "像素面积上限的边长——1024 表示按图片原始比例最多约 1024² 像素，"
        "两边均为 16 的倍数；不会放大",
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
    "rank": ("秩", "LoRA 秩"),
    "alpha": ("alpha", "LoRA alpha（默认等于秩）"),
    "lr": ("学习率", "AdamW 学习率"),
    "warmup_ratio": ("预热比例", "线性预热占总步数的比例"),
    "save_every_epochs": ("每 N 轮保存", "每 N 轮额外保存一次（0 = 只保存最终结果）"),
    "caption_dropout": (
        "标注丢弃率",
        "以空提示词嵌入（CFG 的无条件分支）训练一步的概率",
    ),
    "lora_dtype": (
        "LoRA 精度",
        "保存的 LoRA 文件精度。训练时保持 fp32 主权重，秩矩阵乘法按 bf16 进行",
    ),
    "grad_checkpointing": ("梯度检查点", "按块的激活检查点"),
    "seed": ("随机种子", "随机数种子"),
}

# GenerateRequest fields whose name also exists on another request with a
# different meaning.
FIELDS_CN_GENERATE: dict[str, tuple[str, str]] = {
    "prompt": ("提示词", "提示词（设置 prompts_file 或 example 时忽略）"),
    "negative_prompt": ("反向提示词", "反向提示词（管线默认为空）"),
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
        "euler = 管线默认的 ODE；er_sde = 同一 σ 网格上的 ER-SDE 随机采样（线条更锐利，不增加前向）",
    ),
    "uncond_skip": (
        "无条件分支跳过的块",
        "逗号分隔的 DiT 块编号（'' = 普通 CFG）；管线默认跳过第 22 块",
    ),
    "seed": ("随机种子", "第一个提示词的种子（每个提示词 +1）"),
    "out_dir": ("输出文件夹", "图片和 manifest.json 保存在这里"),
    "spectrum": ("Spectrum", "大部分步数预测最后一个块的输出，而不是运行全部 42 个块"),
}
