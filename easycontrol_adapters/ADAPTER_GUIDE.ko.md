# 나만의 EasyControl 어댑터 만들기

이 가이드는 Anima에 **새로운 EasyControl 컨트롤 태스크**를 개인 용도로 추가하는 방법을 안내합니다. git에 기여하는 것이 아니라 `easycontrol_adapters/<your_task>/` 아래에 두는 로컬 어댑터입니다. 대표 예제인 **colorize** (`easycontrol_adapters/colorization/`)를 단계별로 따라가면서, 자신의 태스크를 위해 정확히 무엇을 바꿔야 하는지 짚어 줍니다.

기억할 것은 단 하나:

> **모델 코드를 작성하는 것이 아닙니다.** 네트워크, 포워드 패스, `b_cond` 게이트, 인퍼런스 캐시 — 이 모든 것은 이미 들어 있고 모든 컨트롤 태스크가 공유합니다. 태스크마다 다른 것은 **레퍼런스 이미지를 어떻게 만드느냐**뿐입니다. 이 가이드의 나머지 내용은 모두 그것을 둘러싼 배관입니다.

---

## 0. 아이디어 — EasyControl 어댑터란 무엇인가

EasyControl은 **레퍼런스 이미지**를 사용해 생성을 안내합니다. 레퍼런스는 VAE를 거쳐 *cond 토큰*으로 변환되고, 생성 중인 이미지와 나란히 흘러갑니다. 각 스텝마다 모델은 두 가지를 모두 봅니다. (전체 아키텍처: `docs/experimental/easycontrol.md`.)

기본 EasyControl은 **동일한 이미지**를 레퍼런스이자 타깃으로 사용합니다 — 그래서 그냥 복사하는 것을 학습합니다. 컨트롤 태스크는 그것을 바꿉니다: 각 타깃을 **다른** 레퍼런스와 짝지어, 모델이 복사 대신 `reference → target`을 학습하도록 합니다.

| | 기본 EasyControl | 컨트롤 태스크 (예: colorize) |
|---|---|---|
| 타깃 (원하는 것) | 이미지 X | 컬러 이미지 X |
| 레퍼런스 (힌트) | 이미지 X (동일) | X의 *변환* 버전 (X의 흑백 망가) |
| 학습 내용 | 복사 | manga → color |
| 텍스트 | 전체 캡션 | (선택) 짧은 캡션 |

colorize의 방식 — 재사용할 수 있는 아이디어입니다: **실제 흑백 망가에는 컬러 버전이 없으므로** `(흑백, 컬러)` 쌍을 수집해 만들 수 없습니다. 그래서 방향을 뒤집습니다 — 이미 보유한 컬러 이미지를 타깃으로 삼고(그것이 원하는 결과이므로), 각 이미지로부터 흑백 레퍼런스를 **직접 만듭니다**(선화 + 스크린톤, 알고리즘으로). 핵심은 만든 흑백이 인퍼런스 시 실제로 넣을 흑백처럼 보여야 한다는 것입니다. `depth → image`나 `pose → image`를 만든다면, "레퍼런스 만들기" 단계는 학습 이미지에 대해 실행하는 depth estimator나 pose detector가 될 것입니다.

즉, 할 일은: **함수 `target_image → reference_image`를 작성하고, 그 출력을 캐시하고, 데이터셋이 그것을 가리키게 하고, config와 한 줄짜리 이름 등록을 추가하는 것**입니다.

---

## 1. 건드리는 네 가지

`<task>`라는 이름의 어댑터를 추가하려면 정확히 다음 네 가지를 만들거나 편집합니다:

| # | 항목 | colorize 버전 | 역할 |
|---|---------|-------------------|--------------|
| 1 | `easycontrol_adapters/<task>/` 프로젝트 | `colorization/` (`mangafy*.py`, `color_caption.py`, `prep.py`) | 레퍼런스 이미지를 만들고 캐시 (선택적으로 짧은 텍스트 캐시도) |
| 2 | `configs/datasets/<task>.toml` | `configs/easycontrol/colorize.toml` | 각 타깃을 레퍼런스와 **cond_cache_dir**로 짝짓는 데이터셋 |
| 3 | `configs/methods/<task>.toml` (+ `configs/gui-methods/<task>.toml`) | `configs/easycontrol/colorize.toml` | config — 데이터셋을 가리키고 LR / epochs / `network_args` 설정 |
| 4 | `scripts/tasks/{training,inference}.py` | `_EASYADAPTERS = {"colorize"}` + 분기 | `EASYADAPTER=<task>`가 `make easycontrol*` 명령에서 작동하게 함 |

순서대로 살펴봅니다. 어느 것도 `networks/`를 건드리지 않습니다.

---

## 2. 항목 1 — 어댑터 프로젝트 (`easycontrol_adapters/<task>/`)

실제 작업이 여기에 있습니다. 두 가지 일을 합니다: **레퍼런스 이미지 만들기**와 **캐시하기**. (선택적으로 세 번째로, 태스크별 텍스트 캐시를 구축합니다.)

### 2a. 레퍼런스를 만드는 함수

일반 함수입니다: 컬러 이미지를 RGB `uint8 (H,W,3)`과 seed로 받아, **같은 크기**의 레퍼런스 이미지를 RGB `uint8 (H,W,3)`으로 반환합니다. (같은 크기가 중요합니다 — §3 참조.) 동일한 seed에 대해 동일한 출력을 내야 하므로, 재실행과 병렬 워커가 모두 일치합니다.

colorize에서 이는 `mangafy.py::mangafy_array`(및 GPU 버전 `mangafy_gpu.py::mangafy_array_gpu`)입니다:

```python
# easycontrol_adapters/colorization/prep.py
Screener = Callable[[np.ndarray, int], np.ndarray]  # (img_rgb, seed) → cond_rgb
```

복사할 가치가 있는 네 가지:

- **이름으로부터 결정론적으로 seed를 만드세요.** colorize는 `zlib.crc32(stem)`을 사용합니다 — 프로세스마다 달라서 병렬 워커가 불일치하게 만드는 Python의 `hash()`가 *아닙니다*. 다양성을 추가하는 것은 괜찮습니다(colorize는 스크린톤 각도를 페이지마다 조금씩 다르게 합니다) — seed에서 파생하기만 하면 재현 가능합니다.
- **무거운 것은 지연 import하세요.** colorize에는 세 개의 엔진(`cv2` / `gpu` / `sd`)이 있고, 페이지가 실제로 그쪽으로 라우팅될 때만 3.5 GB짜리 SD 모델을 로드합니다. 빌더가 모델(depth net, line extractor)을 필요로 한다면, 사용할 때만 import하세요.
- **다운로드 없는 폴백은 큰 장점입니다.** colorize의 `cv2`/`gpu` 엔진은 다운로드가 전혀 필요 없어서, 새 체크아웃에서 추가 다운로드 단계 없이 prep + 학습이 가능합니다. 가능하다면 이런 폴백을 만들어 두세요.
- **파일을 원자적으로 쓰세요.** colorize의 `_save_png_atomic`은 임시 파일에 쓴 뒤 이름을 바꿉니다. 이것 없이는 중단된 실행이 반쯤 쓰인 PNG를 남길 수 있고, "이미 있으면 건너뛰기" 검사가 그 파일을 영원히 신뢰하게 됩니다. 실제로 발생하는 버그입니다 — 이 패턴을 복사하세요.

레퍼런스가 **이미 디스크에 있다면**(실제 depth map, 실제 스케치 등), 빌드 단계를 완전히 건너뛰고 그것을 직접 캐시하면 됩니다. 타깃으로부터 레퍼런스를 파생해야 할 때만 빌드가 필요합니다.

### 2b. (선택) 짧은 텍스트 캐시

colorize는 레퍼런스만 바꾸는 게 아니라 **캡션을 색상 단어로만 축소**하기도 합니다(`color_caption.py::filter_to_colors`). 이유를 이해할 가치가 있습니다: 레퍼런스(선화 + 스크린톤)가 이미 *형태와 레이아웃에 관한 모든 것*을 인코딩하므로, 텍스트가 여전히 말해야 할 것은 흑백이 표현할 수 없는 단 하나 — **색**뿐입니다. 캡션을 색상 단어로 축소하면 남은 단어가 모두 모델이 레퍼런스에서 얻을 수 없는 정보가 되어, 긴 캡션 속에 묻힌 색상 단어로 약하게 스티어링하는 대신 강한 `prompt → color` 연결이 생깁니다.

자신의 태스크에 같은 질문을 해보세요: **레퍼런스가 이미 결정하는 것은 무엇이고, 텍스트가 결정해야 할 것은 무엇인가?** `pose → image` 레퍼런스는 포즈는 고정하지만 의상이나 배경은 고정하지 않습니다 — 아마 *전체* 캡션을 유지할 것입니다. `depth → image` 레퍼런스는 레이아웃은 고정하지만 정체성이나 색상은 고정하지 않습니다. colorize의 "캡션을 여전히 모호한 것으로 축소한다"는 방식은 고려할 *패턴*이지 규칙이 아닙니다 — 많은 어댑터가 캡션을 그대로 두고 텍스트 캐시를 완전히 건너뜁니다(데이터셋에서 `text_cache_dir`를 생략하기만 하면 됩니다 — §4 참조).

캡션을 축소한다면, colorize의 두 개의 독립적인 노브에 유의하세요(혼동하지 마세요):

- **`caption_dropout_rate`** — 자동 채색의 *하한(floor)*. 약 5%의 학습 스텝에서 캡션을 통째로 드롭하여, 프롬프트를 주지 않았을 때의 동작을 학습시킵니다. **낮게** 유지하세요(`0.05`); 높은 값은 무조건부 경로를 과하게 학습시켜 프롬프트를 약하게 만듭니다.
- **`use_shuffled_caption_variants`** — 전체 대 부분의 *균형*. 텍스트 캐시는 여러 버전을 가집니다(v0 = 전체 색상 집합, v1+ = 각 단어가 약 절반의 확률로 드롭된 셔플 버전). 로더는 v0를 20%, v1+를 80%로 추출하므로, "pink hair" 같은 부분 프롬프트만으로도 잘 동작합니다.

### 2c. `prep.py` — 캐시 빌더

세 단계, 각각 **멱등(idempotent)**(이미 완료된 작업은 건너뛰므로 재실행해도 안전합니다):

1. **빌드** — `--src`(`post_image_dataset/resized`) 아래 모든 컬러 이미지를 순회하며 빌더 함수를 실행하고, 소스 레이아웃을 미러링하는 `--staging` 폴더에 레퍼런스 PNG를 씁니다.
2. **인코드** — 스테이징된 레퍼런스를 `library.preprocess.cache_latents`로 `--cond_cache_dir`에 VAE 인코딩합니다. 각 이미지의 **원본 크기**에서 인코딩하므로 레퍼런스 레이턴트가 타깃 레이턴트와 같은 형태가 됩니다. 일반 캐시와 동일한 `{stem}_{WxH}_anima.npz` 포맷입니다.
3. **(선택) 텍스트** — 캡션을 필터를 통해 `--text_cache_dir`로 재인코딩합니다. `library.preprocess.cache_text_embeddings`에 `caption_transform=`(및 `caption_shuffle_variants` / `caption_tag_dropout_rate`)를 주어 수행합니다.

기존 라이브러리 헬퍼를 사용하세요 — `library.preprocess.{cache_latents, cache_text_embeddings, tqdm_progress}` 및 `library.preprocess._dataset.walk_images`. 직접 인코드 루프를 짜지 마세요; `prep.py`는 `scripts/preprocess/*.py`처럼 이들 위에 얹힌 얇은 셸입니다.

colorize가 처리하는, 그 구조를 복사하면 공짜로 물려받는 두 가지 정확성 함정:

- **Stem이 일치해야 합니다.** 텍스트 단계는 캡션 마스터(`image_dataset/`, `resized/`와 같은 레이아웃)에서 `.txt` 캡션을 읽으므로, 결과 텍스트 캐시 파일명이 로더가 찾는 것(`image_dir=post_image_dataset/resized` 기준)과 일치합니다. 캐시 파일명이 타깃 stem과 맞지 않으면 로더가 아무 말 없이 짝짓기를 못 합니다.
- **uncond 사이드카.** colorize의 텍스트 단계는 공유 `T5("")` 빈 프롬프트 사이드카가 없으면 다시 만듭니다. 텍스트 캐시를 만들고 캡션 드롭아웃을 사용한다면 똑같이 하세요(`library.inference.uncond.stage_uncond_sidecar_with_models`).

---

## 3. 지켜야 할 규칙 — 레퍼런스와 타깃의 토큰 카운트가 일치해야 한다

DiT는 Anima의 **원본 형태 버킷팅**으로 동작합니다(두 개의 토큰 카운트 패밀리, 4032와 4200; CLAUDE.md 및 `docs/experimental/easycontrol.md` "Cond token count" 참조). **패딩 노브가 없습니다** — 레퍼런스는 레이턴트의 실제 토큰 카운트로 실행됩니다.

그래서 §2a에서 레퍼런스가 입력과 **같은 크기**여야 한다고 하고, §2c에서 **원본 크기**로 인코딩한다고 하는 것입니다: 두 가지를 모두 지키면 레퍼런스 레이턴트가 자동으로 타깃과 같은 버킷에 들어가고, 모든 것이 그냥 작동합니다. 더 작은 레퍼런스를 원한다면(완전히 괜찮습니다 — 더 작으면 메모리와 속도 모두 이득), **이미지** 수준에서, 인코딩 전에 줄이세요. 그러면 레이턴트가 여전히 실제 버킷에 들어갑니다. 네트워크 안에서 토큰 카운트를 제한하려 하지 마세요.

---

## 4. 항목 2 — 데이터셋 (`configs/datasets/<task>.toml`)

이 파일이 레퍼런스를 타깃과 다르게 만듭니다. 평범한 데이터셋(`[general]` + `[[datasets]]` + `[[datasets.subsets]]`)에 **추가 노브 하나**를 더한 것입니다: `cond_cache_dir` (그리고 선택적으로 `text_cache_dir`).

colorize(`configs/easycontrol/colorize.toml`), 주석 포함:

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

각 리다이렉트가 하는 일:

- **`cond_cache_dir`** — 이것이 컨트롤 태스크로 만들어 주는 유일한 노브입니다. 로더는 여기서 각 타깃을 stem으로 레퍼런스 레이턴트와 매칭합니다. EasyControl의 2-스트림 포워드가 레퍼런스로 사용하는 것입니다.
- **`text_cache_dir`** — **텍스트 캐시만** 리다이렉트합니다(레이턴트는 여전히 `cache_dir`에서 옵니다). 전체 캡션을 유지한다면 완전히 생략하세요 — 그러면 로더가 공유 텍스트 캐시를 사용하고 `prep.py`의 텍스트 단계를 건너뛸 수 있습니다.
- **`flip_aug = false`** — 필수입니다. 뒤집힌 타깃에는 뒤집힌 레퍼런스 레이턴트가 필요한데, 그것은 캐시하지 않았습니다. 플립은 꺼 두세요.

colorize가 타깃 레이턴트와 텍스트에 **공유** `post_image_dataset/lora` 캐시를 재사용한다는 점에 유의하세요 — `make preprocess`가 이미 만들어 놓았고, 아무것도 다시 인코딩하지 않습니다. 어댑터는 *레퍼런스* 캐시(그리고 어쩌면 짧은 텍스트 캐시)만 추가합니다.

---

## 5. 항목 3 — 메서드 config (`configs/methods/<task>.toml`)

`configs/easycontrol/easycontrol.toml`의 거의 복사본입니다. 유일한 구조적 변경은 자신의 데이터셋을 가리키는 `dataset_config`이며, 나머지는 하이퍼파라미터입니다.

colorize(`configs/easycontrol/colorize.toml`), 중요한 라인들:

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

자신의 태스크를 위해 고민할 노브들:

- **`b_cond_init`** — 학습 시작 시 레퍼런스가 얼마나 영향을 미치는가. `-10`은 step 0에서 레퍼런스가 거의 기여하지 않음을 뜻합니다(모델이 처음에는 기본 DiT처럼 작동하다가 점점 레퍼런스에 의존하는 법을 배웁니다). `docs/experimental/easycontrol.md` "Step-0 baseline equivalence"에서 이유를 설명합니다. colorize는 레퍼런스가 더 빨리 작동하도록 `-6`으로 완화합니다 — 레퍼런스가 강한 태스크에서는 그럴 여유가 있습니다. 어느 쪽이든 학습 가능합니다.
- **`easycontrol_cond_noise_max`** — 학습 중 레퍼런스에 추가하는 노이즈 양(σ는 `U(0, max)`에서 샘플, `cond + σ·ε`로 적용). `0`은 레퍼런스를 완벽한 청사진으로 취급합니다; 높은 값은 레퍼런스를 "대략적인 힌트"로 만들어 텍스트가 나머지 디테일을 담도록 강제합니다. colorize는 `0.02`를 사용합니다(아주 작음 — 선화 *자체*가 신호입니다). 기본 easycontrol.toml은 `0.3`을 사용합니다.
- **`easycontrol_drop_p`** — image-CFG를 위한 레퍼런스 드롭아웃 빈도. colorize는 `0`을 사용합니다(항상 레퍼런스를 원합니다); 기본값은 `0.1`입니다.
- **`output_name`** — 고유해야 합니다; 인퍼런스 단계가 이 이름으로 최신 체크포인트를 찾습니다(§6).

`configs/gui-methods/<task>.toml`도 추가할 수 있습니다 — `[variant]` 블록(`family = "easycontrol"`, `label`, `description`, `order`)을 가진 독립형 버전(토글 블록 없음)으로, GUI의 EasyControl 드롭다운에 나타납니다. `configs/gui-methods/easycontrol.toml`을 참조하세요. CLI에서만 실행한다면 건너뛰세요.

---

## 6. 항목 4 — `EASYADAPTER=<task>` 작동시키기

`make easycontrol*` 명령은 `EASYADAPTER` 환경 변수로 전환됩니다. 세 가지 작은 편집으로 `EASYADAPTER=<task>`가 자신의 config, prep, 체크포인트를 사용하게 됩니다.

**`scripts/tasks/training.py`에서:**

1. 허용목록에 이름을 추가합니다:
   ```python
   _EASYADAPTERS = {"colorize", "<task>"}   # was {"colorize"}
   ```
   (`_easyadapter()`가 이 집합에 대해 검증하고 오타에 에러를 냅니다.)

2. `cmd_easycontrol_preprocess`에서 전처리를 자신의 `prep.py`로 라우팅합니다:
   ```python
   adapter = _easyadapter()
   if adapter == "colorize":
       run([PY, "easycontrol_adapters/colorization/prep.py", *extra]); return
   if adapter == "<task>":
       run([PY, "easycontrol_adapters/<task>/prep.py", *extra]); return
   ```

3. 학습 자체는 **편집이 필요 없습니다** — `cmd_easycontrol`이 이미 `train(_easyadapter() or "easycontrol", extra)`를 호출하므로, 이름이 허용목록에 들어가면 `EASYADAPTER=<task>`가 자동으로 `configs/methods/<task>.toml`을 실행합니다.

4. (빌더가 다운로드를 필요로 하는 경우에만) `cmd_easycontrol_download`에 자신의 weight-fetch 태스크를 가리키는 분기를 추가하세요.

**`scripts/tasks/inference.py`**(`cmd_test_easycontrol`)에서: selector가 현재 colorize를 하드코딩하고 있습니다. 자신의 태스크에 맞게 몇 가지 colorize 전용 값을 일반화하세요 — 체크포인트 이름, 출력 폴더, 폴백 레퍼런스 폴더, 빈 프롬프트 기본값:

```python
adapter = (os.environ.get("EASYADAPTER") or "").strip()
is_colorize = adapter == "colorize"
weight_name = "anima_colorize" if is_colorize else "anima_easycontrol"
out_sub     = "colorize"       if is_colorize else "easycontrol"
ref_fallback_dir = (ROOT/"post_image_dataset"/"resized") if is_colorize else (ROOT/"easycontrol-dataset")
```

자신의 `adapter == "<task>"` 케이스를 나란히 추가하세요(weight 이름은 config의 `output_name`과 일치해야 합니다). 자신의 태스크가 colorize처럼 빈 프롬프트 기본값과 특정 폴더의 레퍼런스를 원한다면, 아래쪽의 `is_colorize` 분기들도 그대로 따라 하세요.

---

## 7. 실행하기

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

사전 준비: `make preprocess`를 한 번 실행하여 공유 타깃 레이턴트와 텍스트 캐시가 `post_image_dataset/lora`에 존재해야 합니다(어댑터가 이를 재사용합니다).

### 인퍼런스 팁 (colorize에서 배운 것들)

- **실제 in-distribution 레퍼런스를 입력하세요.** 인퍼런스 시 레퍼런스는 있는 그대로 VAE 인코딩됩니다 — 빌드 단계가 없습니다. colorize는 실제 스크린톤이 입혀진 흑백 페이지를 입력합니다; 밋밋한 그레이스케일 사진은 분포를 벗어나 품질이 떨어집니다. 빌더가 *흉내 내려 했던* 것이 곧 인퍼런스가 받기를 기대하는 것입니다.
- **`--easycontrol_image_match_size`** — 레퍼런스의 종횡비에 맞는 토큰 버킷을 선택하여, 세로로 긴 페이지가 찌그러지지 않게 합니다. colorize는 이를 강제로 켭니다.
- **`--easycontrol_scale`** (`EC_SCALE=`, 레퍼런스를 얼마나 따를지) — 학습 기본값은 `1.0`; 레퍼런스가 너무 강하게 보이면 올리고(1.1–1.2), 더 느슨한 출력을 원하면 내리세요(0.7–0.8).
- **`--guidance_scale`** — 텍스트 설정과 함께 작용합니다. colorize: 빈 프롬프트 → 낮은 CFG(1.0–1.5, 밀어붙일 것이 없음); 텍스트 프롬프트 → 높게(3.0–4.5, 프롬프트가 실제로 작용하게 만드는 것).

---

## 8. 체크리스트

- [ ] `easycontrol_adapters/<task>/`에 결정론적이고 원자적으로 쓰는 빌더
      (`(img, seed) → reference`, 같은 크기)와 멱등 `prep.py`
      (build → encode → 선택적 text).
- [ ] 레퍼런스를 **원본 크기**로 인코딩하여 토큰 카운트가 타깃 버킷과 일치하도록(§3).
- [ ] `configs/datasets/<task>.toml`에 `cond_cache_dir`(+ 선택적
      `text_cache_dir`), `flip_aug = false`, 공유 타깃 캐시 재사용.
- [ ] `configs/methods/<task>.toml` → 데이터셋을 가리키고, 고유한
      `output_name`, `network_module = "networks.methods.easycontrol"`,
      `use_easycontrol = true`. (선택적 GUI 변형.)
- [ ] `EASYADAPTER=<task>`를 `_EASYADAPTERS`에 추가 + 전처리 분기
      (training.py) + inference.py의 일반화된 checkpoint/output/fallback.
- [ ] 전체 실행 전에 `--limit 8` 스테이징 배치를 눈으로 확인.

---

## 9. 더 읽을 곳

- **`easycontrol_adapters/colorization/README.md`** — colorize 설계 노트 전문
  (캡션 정책, 스크린톤 밴드, Phase B 로드맵). 위 모든 것의 레퍼런스 구현.
- **`docs/experimental/easycontrol.md`** — 네트워크 자체: 2-스트림 포워드,
  `b_cond` step-0 벤치, 인퍼런스 캐시, 메모리 사용, 한계. `network_args`를
  건드리기 전에 읽으세요.
- **`networks/methods/easycontrol.py`** — `EasyControlNetwork`와 패치된
  `Block.forward`. 새 어댑터를 위해 이것을 편집할 **필요는 없을** 것입니다; 편집이 필요하다고 생각된다면, 태스크의 실제 차이가 레퍼런스 이미지에 있는 것인지 다시 확인하세요.
- **`networks/CLAUDE.md`** — 모듈별 맵과 디스패치 규칙.
