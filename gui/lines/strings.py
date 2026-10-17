"""en / cn strings shared by the sidecar-line windows (``gui.lines.window``).

Kept out of ``gui.i18n``: these windows are en/cn only and their keys never
reach the main GUI. Each line adds its own keys (``window_title``,
``model_missing``, ``scan_stale``, ``scan_dupes``) through ``LineSpec.ui``.
"""

from __future__ import annotations

LANGUAGES = {"en": "English", "cn": "中文"}

UI: dict[str, dict[str, str]] = {
    "en": {
        "tab_preprocess": "Preprocess",
        "tab_train": "Train",
        "run_cache": "Run preprocessing",
        "run_train": "Train",
        "run_chain": "Preprocess, then train",
        "stop": "Stop",
        "advanced": "Advanced",
        "browse": "Browse…",
        "auto": "auto",
        "language": "Language",
        "model": "Model folder: {path}",
        "guide_preprocess": "Preprocessing encodes every image in the image folder "
        "once — its .txt caption through the text encoder, the image through the "
        "VAE — and writes both to the cache folder. Images that are already cached "
        "are skipped, so after adding images only the new ones are encoded. "
        "Training reads the cache folder only.",
        "guide_train": "Training fits a LoRA on the cached samples; the base model "
        "stays frozen. \"Preprocess, then train\" runs both in order and starts "
        "training only if preprocessing succeeds. Test renders the prompt with the "
        "LoRA and without it at the same seed, so the pair differs by the adapter "
        "only.",
        "click_field": "Click a field label to see its explanation here.",
        "scan": "{pairs}/{images} images have captions · text cached {text} · "
        "latents cached {latents}",
        "scan_no_src": "Pick a folder of images with .txt captions.",
        "cache_counts": "Cache folder: {samples} samples ready to train "
        "(text {text} · latents {latents})",
        "train_ready": "{n} cached samples ready.",
        "train_empty": "No cached samples in this folder yet — run preprocessing first.",
        "bad_value": "{field}: '{value}' is not a valid number.",
        "running": "Running: {label} ({job})",
        "finished": "{label} finished: {state}",
        "chain_next": "Preprocessing done — starting training.",
        "chain_cache": "Training will read the preprocessing output: {path}",
        "reattached": "Re-attached to running job {job}.",
        "idle": "Idle",
        "test_group": "Test",
        "run_test": "Test",
        "test_lora_placeholder": "train output: {path}",
        "no_lora": "No LoRA at {path}.\nTrain first, or set lora in the Test panel.",
        "test_lora": "LoRA ×{m}",
        "test_base": "base model",
        "test_hint": "Renders the prompt with the LoRA and without it (same seed), "
        "so the pair differs by the adapter only.",
        "test_next": "LoRA pass done — rendering the base model.",
    },
    "cn": {
        "tab_preprocess": "预处理",
        "tab_train": "训练",
        "run_cache": "开始预处理",
        "run_train": "开始训练",
        "run_chain": "预处理后训练",
        "stop": "停止",
        "advanced": "高级",
        "browse": "浏览…",
        "auto": "自动",
        "language": "语言",
        "model": "模型文件夹：{path}",
        "guide_preprocess": "预处理会把图片文件夹中的每张图片编码一次——.txt 标注经过"
        "文本编码器，图片经过 VAE——并把结果写入缓存文件夹。已有缓存的图片会被跳过，"
        "所以新增图片后只会编码新图片。训练只读取缓存文件夹。",
        "guide_train": "训练在缓存样本上拟合一个 LoRA，基础模型保持冻结。"
        "「预处理后训练」会依次运行两步，只有预处理成功才开始训练。"
        "测试会用同一个种子分别在加载和不加载 LoRA 的情况下生成，"
        "两张图的差别只来自适配器。",
        "click_field": "点击字段标签可在此处查看说明。",
        "scan": "{images} 张图片中 {pairs} 张有标注 · 已缓存文本 {text} · "
        "已缓存潜变量 {latents}",
        "scan_no_src": "请选择包含图片和 .txt 标注的文件夹。",
        "cache_counts": "缓存文件夹：{samples} 个样本可用于训练"
        "（文本 {text} · 潜变量 {latents}）",
        "train_ready": "已缓存 {n} 个样本，可以训练。",
        "train_empty": "此文件夹中还没有缓存样本——请先运行预处理。",
        "bad_value": "{field}：「{value}」不是有效的数字。",
        "running": "运行中：{label}（{job}）",
        "finished": "{label} 已结束：{state}",
        "chain_next": "预处理完成——开始训练。",
        "chain_cache": "训练将读取预处理输出：{path}",
        "reattached": "已重新连接到正在运行的任务 {job}。",
        "idle": "空闲",
        "test_group": "测试",
        "run_test": "测试",
        "test_lora_placeholder": "训练输出：{path}",
        "no_lora": "找不到 LoRA：{path}\n请先训练，或在测试面板中设置 lora。",
        "test_lora": "LoRA ×{m}",
        "test_base": "基础模型",
        "test_hint": "用同一个种子分别在加载和不加载 LoRA 的情况下生成，"
        "两张图的差别只来自适配器。",
        "test_next": "LoRA 生成完成——开始生成基础模型。",
    },
}
