# 构建你自己的 EasyControl 适配器

本指南说明如何为 Anima 添加一个**新的 EasyControl 控制任务**，供你自己使用。这是一个本地适配器，存放在 `easycontrol_adapters/<your_task>/` 下，不提交到 git。我们以示例 **colorize** (`easycontrol_adapters/colorization/`) 为例逐步讲解，并明确指出针对你自己的任务需要修改哪些地方。

记住一件事：

> **你不是在写模型代码。** 网络、前向传播、`b_cond` 门控、推理缓存——这些都已内置，由所有控制任务共享。每个任务唯一不同的地方是**如何构建参考图像**。本指南的其余内容都只是围绕这一点的配管工作。

---

## 0. 思路——EasyControl 适配器是什么

EasyControl 使用一张**参考图像**来引导生成。参考图像经过 VAE 编码成 *cond token*，与正在生成的图像并行流动。每一步，模型同时看到两者。（完整架构见 `docs/experimental/easycontrol.md`。）

默认的 EasyControl 将**同一张图像**同时用作参考和目标——因此它学到的只是复制。控制任务打破了这一点：将每个目标与**不同的**参考配对，让模型学习 `reference → target`，而不是复制。

| | 默认 EasyControl | 控制任务（如 colorize） |
|---|---|---|
| 目标（你想要的） | 图像 X | 彩色图像 X |
| 参考（提示） | 图像 X（相同） | X 的*变换*版本（X 的黑白漫画） |
| 学到什么 | 复制 | manga → color |
| 文本 | 完整标注 | （可选）较短的标注 |

colorize 的做法——你可以复用的思路：**真实的黑白漫画没有彩色版本可供学习**，所以你无法收集 `(黑白, 彩色)` 配对。因此反转方向——把你已有的彩色图像作为目标（那是你想要的结果），并从每张图像**生成**黑白参考（线稿 + 网点，用算法实现）。关键是生成的黑白图像要看起来像你在推理时实际输入的黑白图像。如果你要构建 `depth → image` 或 `pose → image`，那么"生成参考"这一步就是在训练图像上运行深度估计器或姿态检测器。

所以你的全部工作是：**编写一个函数 `target_image → reference_image`，缓存其输出，让数据集指向它，再加上一个 config 和一行名称注册。**

---

## 1. 你需要改动的四个地方

添加一个名为 `<task>` 的适配器，需要创建或编辑以下四个地方：

| # | 内容 | colorize 版本 | 作用 |
|---|---------|-------------------|--------------|
| 1 | `easycontrol_adapters/<task>/` 项目 | `colorization/`（`mangafy*.py`、`color_caption.py`、`prep.py`） | 构建并缓存参考图像（以及可选的较短文本缓存） |
| 2 | `configs/datasets/<task>.toml` | `configs/easycontrol/colorize.toml` | 通过 **cond_cache_dir** 将每个目标与参考配对的数据集 |
| 3 | `configs/methods/<task>.toml`（+ `configs/gui-methods/<task>.toml`） | `configs/easycontrol/colorize.toml` | config——指向数据集，设置 LR / epochs / `network_args` |
| 4 | `scripts/tasks/{training,inference}.py` | `_EASYADAPTERS = {"colorize"}` + 分支 | 让 `EASYADAPTER=<task>` 在 `make easycontrol*` 命令中正常工作 |

我们按顺序逐一讲解。这里没有任何内容需要改动 `networks/`。

---

## 2. 第一件事——适配器项目（`easycontrol_adapters/<task>/`）

真正的工作在这里。它做两件事：**构建参考图像**和**缓存它**。（可选地，第三件事：构建任务专属的文本缓存。）

### 2a. 构建参考的函数

这是一个普通函数：接受彩色图像 RGB `uint8 (H,W,3)` 和一个种子，返回**相同尺寸**的参考图像 RGB `uint8 (H,W,3)`。（相同尺寸很重要——见 §3。）对相同种子必须给出相同输出，这样重跑和并行 worker 才能保持一致。

在 colorize 中这是 `mangafy.py::mangafy_array`（及其 GPU 版本 `mangafy_gpu.py::mangafy_array_gpu`）：

```python
# easycontrol_adapters/colorization/prep.py
Screener = Callable[[np.ndarray, int], np.ndarray]  # (img_rgb, seed) → cond_rgb
```

值得照搬的四点：

- **从文件名确定性地生成种子。** colorize 使用 `zlib.crc32(stem)`——*不是* Python 的 `hash()`，后者每个进程不同，会导致并行 worker 结果不一致。你仍然可以增加多样性（colorize 按页面抖动网点角度）——只要从种子派生即可保持可复现性。
- **延迟导入重量级依赖。** colorize 有三种引擎（`cv2` / `gpu` / `sd`），只有当某页实际路由到该引擎时才加载 3.5 GB 的 SD 模型。如果你的构建器需要某个模型（深度网络、线条提取器），请只在使用时才导入。
- **有无需下载的备用路径是很大的优势。** colorize 的 `cv2`/`gpu` 引擎不需要任何下载，因此可以在全新检出的仓库上直接进行 prep 和训练，无需额外的下载步骤。如果可以，请为你的任务也提供这样的备用路径。
- **原子写入文件。** colorize 的 `_save_png_atomic` 先写临时文件再重命名。没有这一步，被中断的运行可能留下半写的 PNG，而"如果存在则跳过"的检查会永远信任那个文件。这是真实会出现的 bug——请复制这个模式。

如果你的参考图像**已经在磁盘上**（真实的深度图、真实的草图），可以完全跳过构建步骤，直接缓存它们。只有在需要从目标派生参考时才需要构建。

### 2b.（可选）较短的文本缓存

colorize 不仅改变参考——还**将标注精简为颜色词**（`color_caption.py::filter_to_colors`）。这背后的道理值得理解：参考（线稿 + 网点）已经编码了*关于形状和布局的所有信息*，所以文本还需要说的只是黑白图无法表达的那一件事——**颜色**。将标注精简为颜色词，使每个剩余的词都是模型无法从参考中获取的信息，从而形成强有力的 `prompt → color` 关联，而不是靠埋在长标注中的几个颜色词进行微弱引导。

对你的任务问同样的问题：**参考已经确定了什么，还有什么留给文本来决定？** `pose → image` 参考固定了姿态但没有固定服装或场景——你可能会保留*完整*标注。`depth → image` 参考固定了布局但没有固定身份或颜色。colorize 的"将标注精简为仍然模糊的内容"是一种可以考虑的*模式*，不是规则——很多适配器保留原始标注并完全跳过文本缓存（只需在数据集中省略 `text_cache_dir`——见 §4）。

如果你确实要精简标注，注意 colorize 的两个独立旋钮（不要混淆它们）：

- **`caption_dropout_rate`** — 自动着色的*下限（floor）*。约 5% 的训练步骤会完全丢弃标注，训练模型在没有提示时的行为。保持它**较低**（`0.05`）；过高的值会过度训练无条件路径，使提示效果变弱。
- **`use_shuffled_caption_variants`** — 完整对部分的*平衡*。文本缓存包含多个版本（v0 = 完整颜色集合，v1+ = 每个词以约一半概率丢弃后的打乱版本），加载器以 20% 的概率选 v0、80% 的概率选 v1+，这样像"pink hair"这样的局部提示也能生效。

### 2c. `prep.py`——缓存构建器

三个阶段，每个都是**幂等的**（跳过已完成的工作，重跑安全）：

1. **构建** — 遍历 `--src`（`post_image_dataset/resized`）下的所有彩色图像，运行构建函数，将参考 PNG 写入一个镜像源目录结构的 `--staging` 文件夹。
2. **编码** — 通过 `library.preprocess.cache_latents` 将暂存的参考图像 VAE 编码到 `--cond_cache_dir`，按每张图像的**原生尺寸**编码，使参考 latent 与目标 latent 形状一致。与普通缓存相同的 `{stem}_{WxH}_anima.npz` 格式。
3. **（可选）文本** — 通过 `library.preprocess.cache_text_embeddings`（带 `caption_transform=`，以及 `caption_shuffle_variants` / `caption_tag_dropout_rate`）将标注经过过滤器重新编码到 `--text_cache_dir`。

使用现有的库辅助函数——`library.preprocess.{cache_latents, cache_text_embeddings, tqdm_progress}` 和 `library.preprocess._dataset.walk_images`。不要自己写编码循环；`prep.py` 和 `scripts/preprocess/*.py` 一样，只是这些函数上方的薄壳。

colorize 处理了两个正确性陷阱，照搬它的结构即可免费获得：

- **Stem 必须匹配。** 文本阶段从标注主目录（`image_dataset/`，与 `resized/` 布局相同）读取 `.txt` 标注，使生成的文本缓存文件名与加载器查找的内容（基于 `image_dir=post_image_dataset/resized`）匹配。如果缓存文件名与目标 stem 不匹配，加载器会静默地不进行配对。
- **uncond sidecar。** colorize 的文本阶段在共享的 `T5("")` 空提示 sidecar 缺失时会重新创建它。如果你构建了文本缓存并使用了 caption dropout，也要这样做（`library.inference.uncond.stage_uncond_sidecar_with_models`）。

---

## 3. 不能违反的规则——参考和目标的 token 数必须匹配

DiT 运行在 Anima 的**原生形状分桶**上（两个 token 数家族，4032 和 4200；见 CLAUDE.md 和 `docs/experimental/easycontrol.md` 中的"Cond token count"）。**没有填充旋钮**——参考以其 latent 实际的 token 数运行。

这正是 §2a 要求参考与输入**尺寸相同**、§2c 要求按**原生尺寸**编码的原因：两者都做到，参考 latent 就会自动落入与目标相同的桶，一切正常工作。如果你想要较小的参考（完全可以——越小越省内存、越快），请在**图像**层面、编码之前缩小，这样 latent 仍然落在真实的桶上。不要试图在网络内部限制 token 数。

---

## 4. 第二件事——数据集（`configs/datasets/<task>.toml`）

这个文件使参考与目标不同。它是一个普通数据集（`[general]` + `[[datasets]]` + `[[datasets.subsets]]`），只多了**一个额外旋钮**：`cond_cache_dir`（以及可选的 `text_cache_dir`）。

colorize（`configs/easycontrol/colorize.toml`），带注释：

```toml
[general]
caption_extension = '.txt'
keep_tokens = 3

[[datasets]]
batch_size = 1
validation_split = 0.005
validation_seed = 42

  [[datasets.subsets]]
  image_dir = 'post_image_dataset/resized'        # the COLOR targets
  cache_dir = 'post_image_dataset/lora'           # target latents + text — REUSED, not rebuilt
  cond_cache_dir = 'post_image_dataset/easycontrol/colorize/cond'   # ← the reference latents (from prep.py)
  text_cache_dir = 'post_image_dataset/easycontrol/colorize/text'   # ← color-only text cache (from prep.py)
  recursive = true
  flip_aug = false        # latents can't be flipped after the fact, and there's no flipped reference
  num_repeats = 1
```

各重定向的作用：

- **`cond_cache_dir`** — 使这成为控制任务的唯一旋钮。加载器在这里通过 stem 将每个目标与参考 latent 匹配。这是 EasyControl 双流前向传播所使用的参考。
- **`text_cache_dir`** — **只**重定向文本缓存（latent 仍来自 `cache_dir`）。如果你保留完整标注，可以完全省略它——这样加载器会使用共享文本缓存，你也可以跳过 `prep.py` 的文本阶段。
- **`flip_aug = false`** — 必须。翻转后的目标需要翻转后的参考 latent，而你没有缓存那个。保持翻转关闭。

注意 colorize**复用**共享的 `post_image_dataset/lora` 缓存作为目标 latent 和文本——`make preprocess` 已经构建了它们，没有任何东西被重新编码。你的适配器只需添加*参考*缓存（以及可能的较短文本缓存）。

---

## 5. 第三件事——方法 config（`configs/methods/<task>.toml`）

这几乎是 `configs/easycontrol/easycontrol.toml` 的副本。唯一的结构性改动是 `dataset_config` 指向你的数据集；其余都是超参数。

colorize（`configs/easycontrol/colorize.toml`），重要的几行：

```toml
dataset_config = "configs/easycontrol/colorize.toml"   # ← your dataset from §4

network_module = "networks.methods.easycontrol"     # SHARED — same network as plain EasyControl

network_dim = 32
network_alpha = 32
network_args = [
    "b_cond_init=-6.0",     # how strongly the reference starts out (see below)
    "cond_scale=1.0",
    "apply_ffn_lora=1",     # 0 → drop the FFN LoRA, about half the trainable params
]

output_name = "anima_colorize_full"   # ← checkpoint name; the inference selector looks for this

use_easycontrol = true
easycontrol_drop_p = 0.0              # reference dropout for image-CFG; default 0.1, colorize wants 0
masked_loss = false

caption_dropout_rate = 0.05           # auto-color floor (§2b)
use_shuffled_caption_variants = true  # full-vs-partial balance (§2b)
easycontrol_cond_noise_max = 0.02     # small — too much noise erases the line art
learning_rate = 2e-5
max_train_epochs = 3
blocks_to_swap = 0                    # recommended for EasyControl
gradient_checkpointing = true
unsloth_offload_checkpointing = true
```

为你的任务需要考虑的旋钮：

- **`b_cond_init`** — 训练开始时参考的影响程度。`-10` 表示参考在 step 0 时几乎不起作用（模型一开始像普通 DiT 一样运行，然后逐渐学会依赖参考）；`docs/experimental/easycontrol.md` 中的"Step-0 baseline equivalence"解释了原因。colorize 将其放宽到 `-6`，让参考更早发挥作用——对于参考信号强的任务这是合理的。无论哪种情况，它都是可学习的。
- **`easycontrol_cond_noise_max`** — 训练过程中添加到参考上的噪声量（σ 从 `U(0, max)` 采样，以 `cond + σ·ε` 形式应用）。`0` 表示将参考视为完美的蓝图；较大的值会将其降级为粗略的"提示"，迫使文本承载缺失的细节。colorize 使用 `0.02`（非常小——线稿*本身就是*信号）。默认的 easycontrol.toml 使用 `0.3`。
- **`easycontrol_drop_p`** — 用于图像 CFG 的参考 dropout 频率。colorize 使用 `0`（总是需要参考）；默认值是 `0.1`。
- **`output_name`** — 必须唯一；推理步骤通过此名称找到最新的检查点（§6）。

你也可以添加 `configs/gui-methods/<task>.toml`——一个独立版本（无切换块），带有 `[variant]` 块（`family = "easycontrol"`、`label`、`description`、`order`），使其出现在 GUI 的 EasyControl 下拉菜单中。参见 `configs/gui-methods/easycontrol.toml`。如果你只从命令行运行，可以跳过这一步。

---

## 6. 第四件事——让 `EASYADAPTER=<task>` 工作

`make easycontrol*` 命令通过 `EASYADAPTER` 环境变量进行切换。三处小改动可以让 `EASYADAPTER=<task>` 使用你的 config、prep 和检查点。

**在 `scripts/tasks/training.py` 中：**

1. 将名称添加到允许列表：
   ```python
   _EASYADAPTERS = {"colorize", "<task>"}   # was {"colorize"}
   ```
   （`_easyadapter()` 会对照此集合进行校验，拼写错误时报错。）

2. 在 `cmd_easycontrol_preprocess` 中将预处理路由到你的 `prep.py`：
   ```python
   adapter = _easyadapter()
   if adapter == "colorize":
       run([PY, "easycontrol_adapters/colorization/prep.py", *extra]); return
   if adapter == "<task>":
       run([PY, "easycontrol_adapters/<task>/prep.py", *extra]); return
   ```

3. 训练本身**无需**改动——`cmd_easycontrol` 已经调用了 `train(_easyadapter() or "easycontrol", extra)`，所以只要名称进入允许列表，`EASYADAPTER=<task>` 就会自动运行 `configs/methods/<task>.toml`。

4. （仅当你的构建器需要下载时）在 `cmd_easycontrol_download` 中为你的权重获取任务添加一个分支。

**在 `scripts/tasks/inference.py`**（`cmd_test_easycontrol`）中：选择器目前硬编码了 colorize。将几个 colorize 专属的值针对你的任务进行泛化——检查点名称、输出文件夹、备用参考文件夹和空提示默认值：

```python
adapter = (os.environ.get("EASYADAPTER") or "").strip()
is_colorize = adapter == "colorize"
weight_name = "anima_colorize" if is_colorize else "anima_easycontrol"
out_sub     = "colorize"       if is_colorize else "easycontrol"
ref_fallback_dir = (ROOT/"post_image_dataset"/"resized") if is_colorize else (ROOT/"easycontrol-dataset")
```

在旁边添加你的 `adapter == "<task>"` 分支（权重名必须与你 config 的 `output_name` 匹配）。如果你的任务像 colorize 一样需要空提示默认值和来自特定文件夹的参考，也请复制下方的 `is_colorize` 分支。

---

## 7. 运行它

```bash
# 1. Build the reference cache (build + VAE-encode). Idempotent.
make easycontrol-preprocess EASYADAPTER=<task>
#    Check a few first:
python easycontrol_adapters/<task>/prep.py --limit 8
#    Eyeball the staged reference PNGs under post_image_dataset/<task>_staging/

# 2. Train (DiT frozen, adapter only).
make easycontrol EASYADAPTER=<task>

# 3. Inference — give it a real, in-distribution reference image.
REF_IMAGE=path/to/condition.png make test-easycontrol EASYADAPTER=<task>
#    Steer with text:  ... ARGS='--prompt "..."'
```

前置条件：先运行一次 `make preprocess`，使共享的目标 latent 和文本缓存存在于 `post_image_dataset/lora`（你的适配器会复用它们）。

### 推理技巧（来自 colorize 的经验）

- **提供真实的分布内参考图像。** 推理时参考图像原样经过 VAE 编码——没有构建步骤。colorize 输入真实的带网点黑白页面；普通的灰度照片不在分布内，效果会更差。你的构建器*模仿的*是什么，推理时就期望接收什么。
- **`--easycontrol_image_match_size`** — 选择与参考图像宽高比匹配的 token 桶，避免竖版页面被压扁。colorize 强制开启此项。
- **`--easycontrol_scale`**（`EC_SCALE=`，对参考的遵循程度）——训练默认值为 `1.0`；如果参考影响过强则调高（1.1–1.2），想要更宽松的输出则调低（0.7–0.8）。
- **`--guidance_scale`** — 与你的文本设置共同作用。colorize：空提示 → 低 CFG（1.0–1.5，没有目标可以推向）；文本提示 → 较高（3.0–4.5，正是这让提示真正起作用）。

---

## 8. 检查清单

- [ ] `easycontrol_adapters/<task>/`，包含一个确定性、原子写入的构建器
      （`(img, seed) → reference`，相同尺寸）和一个幂等的 `prep.py`
      （构建 → 编码 → 可选文本）。
- [ ] 参考按**原生尺寸**编码，使 token 数与目标桶匹配（§3）。
- [ ] `configs/datasets/<task>.toml`，带 `cond_cache_dir`（+ 可选的
      `text_cache_dir`）、`flip_aug = false`，并复用共享的目标缓存。
- [ ] `configs/methods/<task>.toml` → 指向数据集，唯一的
      `output_name`，`network_module = "networks.methods.easycontrol"`，
      `use_easycontrol = true`。（可选 GUI 变体。）
- [ ] `EASYADAPTER=<task>` 已添加到 `_EASYADAPTERS` + 一个预处理分支
      （training.py）+ inference.py 中泛化的检查点/输出/备用路径。
- [ ] 在完整运行前，先用 `--limit 8` 的暂存批次目视确认过。

---

## 9. 进一步阅读

- **`easycontrol_adapters/colorization/README.md`** — 完整的 colorize 设计笔记
  （标注策略、网点频段、Phase B 路线图）。上文所有内容的参考实现。
- **`docs/experimental/easycontrol.md`** — 网络本身：双流前向传播、`b_cond` step-0
  基准、推理缓存、内存使用、限制。改动 `network_args` 前请先阅读。
- **`networks/methods/easycontrol.py`** — `EasyControlNetwork` 和打了补丁的
  `Block.forward`。对于新适配器，你**不应该**需要编辑它；如果你觉得需要，请重新确认
  任务的真正区别是否确实在于参考图像。
- **`networks/CLAUDE.md`** — 逐模块映射和分发规则。
