# MOSS-Image LoRA 가이드북

GUI에서 MOSS-Image용 LoRA를 학습하고 테스트하는 가이드입니다. `make gui` 로 열고 상단 바의 **모델** 드롭다운을 **MOSS-Image** 로 바꾸면 됩니다 (`make gui-moss` 는 같은 패널을 별도 창으로 엽니다). 패널은 영어로 표시되며, GUI 언어가 중국어일 때만 중국어로 표시됩니다.

MOSS 라인은 **알파** 단계입니다. 지금까지 실제 스타일 학습은 한 번(30장짜리 작가 세트)만 측정했고, 플래그와 기본값은 바뀔 수 있습니다. 아래 수치는 모두 그 학습과 16 GB RTX 5070 Ti에서 돌린 벤치마크에서 나온 것입니다.

---

## 목차

1. [MOSS-Image란](#1-moss-image란)
2. [설치](#2-설치)
3. [전처리, 학습, 테스트](#3-전처리-학습-테스트)
4. [권장 설정](#4-권장-설정)
5. [테스트](#5-테스트)
6. [문제 해결](#6-문제-해결)

- [부록. CLI](#부록-cli)

---

## 1. MOSS-Image란

[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)(OpenMOSS, Apache-2.0)는 Anima와 별개의 모델이며, 이 저장소에서 독립된 라인으로 학습합니다.

| | Anima | MOSS-Image |
|---|---|---|
| 디퓨전 모델 | Anima DiT | 5B single-stream DiT (42 블록) |
| 텍스트 인코더 | Qwen3 0.6B | Qwen3.5-9B |
| VAE | QwenImage VAE | FLUX.2 VAE |
| 캐시 | `post_image_dataset/lora/` | `post_image_dataset/moss/` |
| LoRA | `output/ckpt/` | `output/ckpt_moss/` |
| 설정 | variant TOML + Hardware 프리셋 | 패널의 입력 필드 |

- 캐시, 설정, LoRA는 서로 호환되지 않습니다. MOSS LoRA는 Anima에서 로드되지 않고, Anima LoRA도 MOSS에서 로드되지 않습니다. Anima의 variant, 프리셋, `make lora` 도 여기에는 적용되지 않습니다.
- MOSS는 DiT 두 개를 제공합니다. **Base** 는 사전학습 모델이고, **SFT** 는 이를 파인튜닝한 모델로 광택 있는 가챠풍 그림체를 냅니다. 학습은 Base에서 하세요. Base에서 학습한 LoRA는 SFT에서도 동작하며, SFT 특유의 광택을 대부분 지우고 Base + LoRA에 가까운 결과를 냅니다.

---

## 2. 설치

| 팩 | 내용 | 크기 |
|---|---|---|
| `moss-base` | DiT, 텍스트 인코더, VAE, 토크나이저 | 약 27 GB |
| `moss-sft` (선택) | SFT DiT만 포함, 텍스트 인코더와 VAE는 Base 것을 사용 | 약 10 GB |

GUI의 모델 다운로드 창에서 받거나, 다음 명령을 실행합니다.

```bash
make download-model ARGS=moss-base
make download-model ARGS=moss-sft
```

가중치는 `models/moss_image/MOSS-Image-Base/` 와 `models/moss_image/MOSS-Image-SFT/` 에 저장됩니다. 다른 위치에 두려면 `.env` 에 `ANIMA_MOSS_MODEL_DIR` 을 설정하거나, 패널의 **Advanced** 에서 `model_dir` 을 지정합니다 (두 `MOSS-Image-*` 폴더를 담은 폴더). 폴더가 준비될 때까지 패널에 `Model not found: …` 가 표시됩니다.

기본 캡션 형식(`layout_json`, §3.1)에는 danbooru 태그 CSV도 필요합니다. `make download-models` 가 이미 받아오며, 따로 받으려면 `make download-model ARGS=danbooru_tags` 를 실행합니다.

---

## 3. 전처리, 학습, 테스트

패널에는 **Preprocess** 와 **Train** 두 탭이 있고, **Test** 상자는 Train 탭 아래쪽에 있습니다. 자주 쓰지 않는 필드는 **Advanced** 안에 접혀 있습니다. 각 버튼은 학습 데몬에 작업을 제출합니다. 모델 드롭다운을 Anima로 되돌리거나 GUI를 닫아도 작업은 계속 실행되고, 다시 열면 실행 중인 작업에 재연결되며(`Re-attached to running job …`), **Stop** 으로 취소합니다.

### 3.1 데이터셋과 캡션

같은 이름의 `.txt` 캡션이 붙은 이미지 폴더이며, 하위 폴더도 포함됩니다. 기본 소스는 Anima의 리사이즈 폴더 `post_image_dataset/resized` (수정된 캡션 포함)라서, Anima용으로 전처리한 데이터셋을 그대로 쓸 수 있습니다. 한 세트만 학습하려면 `src` 를 해당 하위 폴더로 지정합니다.

- **Anima식 태그 캡션으로 학습과 프롬프트 모두 동작합니다.** MOSS 자체는 자연어로 학습되었으며, 긴 캡션이나 캐릭터가 많은 캡션은 아직 검증되지 않았습니다.
- **`layout_json` (기본값)**: Anima 캡션을 MOSS의 layout-JSON 프롬프트로 변환합니다. rating/meta, 캐릭터, 시리즈, 작가는 각자의 키로 들어가고, 나머지 태그는 caption이 되며, 위치 절은 위치 정보가 붙은 subject로, OCR 텍스트 줄은 text 항목이 됩니다. 30장 세트에서 일반 태그와 같은 스타일 강도를 유지하면서, 세트의 속성이 다른 캐릭터에 새는 현상은 더 적었습니다.
- **학습과 프롬프트는 같은 형식으로**: `caption_format` (Preprocess)과 `prompt_format` (Test)이 일치해야 합니다. 둘 다 `layout_json` 이거나, 일반 캡션이면 둘 다 `text` 입니다. 프롬프트는 어느 쪽이든 Anima 태그 형식으로 입력하며, `layout_json` 이 변환해 줍니다.
- **트리거는 캡션에 적힌 그대로의 작가 태그입니다.** 예: `@channel (caststation)`. 프롬프트에도 같은 형태로 넣습니다.
- 자연어 프롬프트는 `layout_json` 을 거쳐도 하나의 caption 문자열로 들어가며, 일반 텍스트 프롬프트와 똑같이 동작합니다.
- ComfyUI에서는 **Anima Caption → MOSS Layout JSON** 노드(Anima Tagger 팩)가 텍스트 인코더 앞에서 같은 변환을 합니다.

### 3.2 전처리 (Preprocess)

모든 이미지의 Qwen3.5-9B 텍스트 임베딩과 FLUX.2 VAE 잠재를 미리 캐싱해, 학습 중에는 두 모델을 로드하지 않습니다.

| 필드 | 기본값 | 하는 일 |
|---|---|---|
| `src` | `post_image_dataset/resized` | 이미지 + 캡션 폴더 |
| `out` | `post_image_dataset/moss` | 캐시 폴더: 이미지당 파일 두 개, 하위 폴더 없이 저장 |
| `resolution` | `1024` | 픽셀 면적 상한: 원본 비율 그대로 최대 약 1024² 픽셀, 양변은 16의 배수, 업스케일하지 않음 |
| `caption_format` | `layout_json` | `layout_json` 또는 `text` (§3.1) |

**Run preprocessing** 을 누릅니다. 상태 줄에 이미지, 캡션, 캐시된 파일 수가 표시됩니다. 재실행은 증분 방식입니다. 새 이미지는 추가되고, 캐시 이후 수정된 캡션은 다시 인코딩되며(상태 줄에 개수 표시), `caption_format` 을 바꾸면 텍스트 캐시를 다시 인코딩합니다. 하위 폴더끼리 파일 이름이 겹치는 이미지는 캐시 파일 이름 앞에 폴더 이름이 붙습니다.

### 3.3 학습 (Train)

| 필드 | 기본값 | 하는 일 |
|---|---|---|
| `cache` | `post_image_dataset/moss` | 전처리 결과 폴더 |
| `output` | `output/ckpt_moss/moss_lora.safetensors` | LoRA 경로. 옆에 `.json` 실행 리포트(스텝별 loss)가 함께 저장됨 |
| `model` | `base` | 학습할 DiT |
| `epochs` | `8` | 캐시를 도는 횟수 (스텝당 이미지 1장) |
| `save_every_epochs` | `0` | N 에폭마다 `<name>_e10.safetensors` 형태로 추가 저장 |

랭크, alpha, lr, 어댑터 종류는 §4에서 다룹니다. **Train** 은 `cache` 를 읽고, **Preprocess, then train** 은 전처리를 먼저 실행한 뒤 `cache` 를 그 `out` 으로 맞춰 학습합니다.

### 3.4 테스트 (Test)

같은 시드로 프롬프트를 두 번 렌더링합니다. 한 번은 `lora_multiplier` 강도의 LoRA를 걸고, 한 번은 LoRA 없이. 두 이미지가 패널에 나란히 표시되며, 차이는 LoRA에서만 옵니다.

- `lora` 를 비워 두면 Train 탭의 `output` 을 씁니다. 다른 파일(예: `_e10` 체크포인트)을 테스트하려면 여기에 지정합니다.
- `model` 을 `sft` 로 바꾸면 Base LoRA를 SFT에서 테스트합니다.
- 결과: `output/tests/<날짜-시각>/lora/` 와 `…/base/`, 각각 `manifest.json` 포함. 테스트마다 새 폴더가 생기므로 이전 결과는 남습니다.

---

## 4. 권장 설정

측정한 학습: 한 작가(`@channel (caststation)`)의 이미지 30장, Base, LoRA 랭크 16, alpha 16, lr 1e-4(이전 기본값), `weight_svd`, 20 에폭 = 600 스텝, 약 18–21분. 시도한 변형 중 작가의 특징을 가장 많이 담으면서 다른 캐릭터로의 누출이 없었습니다. 현재 기본값(랭크 32, alpha 128, lr 1e-4, 채널 스케일링 0.5)은 이후 비교에서 나왔으며 아래에 설명합니다.

| 필드 | 기본값 | 참고 |
|---|---|---|
| `algo` | `lora` | `lokr` / `loha`: §4.1 |
| `rank` | `32` | 랭크 16에서 파라미터 37.2 M (이전 기본값) |
| `alpha` | `lora`는 `128`(랭크 32에서 scale 4), `lokr` / `loha`는 = rank | scale = alpha / rank |
| `lr` | `1e-4` | AdamW, 5 % 선형 워밍업 |
| `channel_scaling_alpha` | `0.5` | `lora` 전용; `0`이면 끔 |
| `down_init` | `weight_svd` | `weight_svd` 또는 `kaiming` |
| `epochs` | `8` | 측정한 학습은 20 |
| `caption_dropout` | `0` | 측정한 학습은 0 |

- **alpha와 lr**: AdamW에서는 scale을 올리는 것이 LoRA up 행렬의 lr을 그 배수만큼 올리는 것과 같습니다. 랭크 16에 alpha 90(scale 5.6), lr 1e-4면 up 행렬은 약 5.6e-4가 됩니다. 이 설정은 기본값과 같은 스타일을 냈지만 세트의 서명을 베끼기 시작했습니다. 샘플 하나에 작가의 워터마크와 핸들이 찍혀 나왔습니다. alpha 90에 lr 5e-5는 더 약하고 부분적인 전이만 보였습니다. 그 비교에서는 alpha = rank, lr 1e-4가 1위였고, 아래의 이후 비교 전까지 기본값이었습니다.
- **랭크 32, alpha 128: lr 1e-4 vs 2e-5 (채널 스케일링 없음)**: caststation 세트(20 에폭, `weight_svd`, layout_json)에 작가 태그를 넣은 상태에서, 복잡한 갤러리 프롬프트 두 개(10명 앙상블, 4:1 배너)의 결과가 달랐습니다. 랭크 32 / alpha 128에 lr 1e-4는 스타일을 강하게 밀어(흰 배경, 단조로운 옅은 채색) 프롬프트의 디테일을 잃었습니다(가방을 든 캐릭터가 틀림, 인형 누락). lr 2e-5에서는 디테일이 유지되고(가방 든 캐릭터 정확, 인형 있음) 스타일 변화는 더 약했으며, "훨씬 낫다"고 판단되어 채널 스케일링이 들어오기 전까지 기본값이었습니다. AdamW에서 scale 4, lr 2e-5의 실효 lr은 up 행렬 8e-5, down 행렬 2e-5입니다.
- **현재 기본값(lr 1e-4, 채널 스케일링 0.5)**: 채널 스케일링은 LoRA의 채널별 업데이트를 재조정해, 소수의 매우 큰 입력 채널(MOSS에서는 모든 FFN에 입력되는 채널 하나)이 가장 큰 몫을 차지하지 않게 합니다. 시작 시점의 출력은 그대로이며 저장되는 파일은 일반 LoRA입니다. lr 1e-4에서 강한 스타일을 유지하면서도 프롬프트를 따랐고, 채널 스케일링 없이 같은 학습을 하면 프롬프트에 없는 요소(안경, 깨진 글자 간판, 공원 대신 거리)가 추가되었습니다. 자세한 내용: `docs/moss/channel_scaling.md`. 이전 동작을 원하면 `channel_scaling_alpha 0`과 `lr 2e-5`로 설정하세요.
- **down_init**: `weight_svd` 는 LoRA의 down 행렬을 베이스 가중치의 상위 특이 벡터로 시작하고, `kaiming` 은 무작위 초기화입니다. 육안으로는 `weight_svd` 가 작가 스타일을 더 잘 유지했습니다. 두 초기화는 같은 시드에서도 다른 샘플을 내며, 그 차이는 시드를 바꾼 정도입니다.
- **epochs**: 20 에폭 중 10 에폭 시점의 결과는 더 느슨하고 다양했고, 20 에폭은 더 깔끔하고 균일했지만 정면·중앙 상반신 구도로 약간 쏠렸습니다. `save_every_epochs` 를 설정하면 한 번의 학습으로 둘을 비교할 수 있습니다.
- **sigma_\*** (Advanced): 기본값은 MOSS의 최종 사전학습 노이즈 스케줄입니다. Anima의 스케줄(`sigma_mu 0`, `sigma_s 1`, `sigma_cutoff 1.0`, `sigma_uniform 0`)은 모든 설정 중 전이가 가장 약했습니다.

### 4.1 LoKr와 LoHa

`algo` = `lokr` (크로네커 곱) 또는 `loha` (랭크 r 쌍 두 개의 아다마르 곱)이며, LyCORIS 구현입니다. 파일은 LyCORIS의 텐서 이름을 그대로 쓰고, 같은 `lora` 필드로 로드합니다. `down_init` 은 `lora` 에만 적용됩니다.

| 구성 | 필드 | 속도 (16 GB, 1024²) | 작가 세트, lr 1e-4 결과 |
|---|---|---|---|
| LoKr f8 | `rank 100000`, `factor 8` | 2.26 s/step, 여유 3.6 GB | LoRA에 근접, 얼굴의 개성은 약간 덜함 |
| LoHa r16 | `rank 16` | 2.49 s/step, 여유 1.7 GB | 과적합: 회청색으로 팔레트가 바뀌고, 세트의 의상이 다른 캐릭터에 입혀짐 |

- 둘 다 LoRA보다 메모리를 더 씁니다. **Advanced** 에서 `blocks_to_swap 24`, `activation_memory_budget 0.1` 로 설정하세요. 이전 기본값(swap 12, budget 0.15)으로는 1024²에서 메모리가 부족했습니다.
- LoKr의 `rank` 가 큰 인수 변의 절반 이상이면(위의 `rank 100000`) 그 인수는 전체 행렬이 되고 alpha는 무시됩니다(scale 1).
- LoHa: 다음으로 시도할 값은 lr 5e-5이며, 아직 측정하지 않았습니다.

### 4.2 메모리 구성 (16 GB)

기본값: `grad_checkpointing` 끔, `compile` 켬, 그리고 **Advanced** 의 `blocks_to_swap 12` (CPU에 두고 필요할 때 올리는 DiT 블록 수)와 `activation_memory_budget 0.1` (forward 결과 중 backward용으로 보관하는 비율, 나머지는 재계산).

| 구성 | 필드 | s/step | 여유 VRAM |
|---|---|---|---|
| 기본값 | swap 12, budget 0.1 (랭크 32) | 1.85 | 약 1.7 GiB |
| 이전 기본값 | swap 12, budget 0.15 (랭크 16) | 1.77 | 약 0.7 GiB |
| 여유 확보 | swap 24, budget 0.2 | 1.78 | 2.26 GiB |
| 체크포인팅 | `grad_checkpointing` 켬, swap 0 | 2.09 | 1.60 GiB |

합성 1024² 이미지 + 텍스트 300 토큰(합계 약 4.4k 토큰), LoRA 랭크 16, RTX 5070 Ti 기준입니다. 현재 기본값(랭크 32, budget 0.1, swap 12)은 약 4.5k 토큰(이미지 4096 + 텍스트 455)에서 1.85 s/step, 여유 약 1.7 GB, 최대 reserved 약 13.1 GB였고, 랭크 16에 budget 0.1은 1.82 s/step, 여유 약 2.0 GB였습니다. 실제 샘플은 최대 약 4.9k 토큰까지 나왔고, 캡션이 길수록 메모리를 더 쓰는데, `swap 24` 는 같은 속도로 그만큼의 여유를 남깁니다.

- swap 8에서 36 사이는 속도가 거의 같고(1.75–1.84 s/step), budget 0.1–0.3도 속도에 거의 영향이 없습니다. swap 40은 PCIe 전송이 연산 뒤에 숨지 못해 느려집니다(2.34 s/step). 그 아래 구간에서 swap 수는 속도가 아니라 VRAM을 정하는 값입니다.
- 최적 조합은 PCIe 링크와 GPU에 따라 다릅니다. `scripts/moss/tune_swap.py` (CLI, [부록](#부록-cli), 약 4분)가 캐시에서 가장 큰 샘플로 swap 수별 속도를 재고, 학습에 쓸 필드 값을 출력합니다.
- `blocks_to_swap -1` 은 여유 VRAM에서 swap 수를 정합니다. LoRA 기준으로 보정되어 있어 LoKr/LoHa에는 여유를 적게 잡습니다.
- `compile` 을 끄려면 `grad_checkpointing` 을 켜야 합니다. compile을 켠 학습은 3.1 대신 2.0 s/step이었고, 시작할 때 약 30초 컴파일합니다.

---

## 5. 테스트

샘플러 기본값은 MOSS 레퍼런스 파이프라인을 따릅니다. Anima에서 쓰던 샘플러 설정은 그대로 옮겨 오지 않습니다.

| 필드 | 기본값 | 하는 일 |
|---|---|---|
| `cfg` | `3.0` | CFG 스케일. `1.0` = uncond 패스 없음 |
| `uncond_skip` | `22` | uncond 패스는 22번 블록을 건너뛴 모델 (스텝당 패스는 여전히 두 번). `''` = 일반 CFG |
| `shift` | `5.0` (Advanced) | 스케줄러 shift |
| `steps` | `50` | |
| `sampler` | `er_sde` | `er_sde` 또는 `euler` |
| `width` / `height` | `1024` | 16의 배수 |
| `negative_prompt` | 비어 있음 | 파이프라인 기본값 |

- **er_sde와 euler**: `euler` 는 레퍼런스 파이프라인의 샘플러입니다. `er_sde` 는 비용이 같고 선이 더 선명하며 디테일이 더 촘촘합니다. LoRA를 걸었을 때 손도 더 깔끔하게 나왔습니다. 대신 프롬프트의 일부 디테일(노을빛, 머리색, "sitting on a bench")에서 벗어나고, 세트의 그림체도 조금 잃었습니다. 같은 시드에서도 두 샘플러는 다른 이미지를 내므로, LoRA와 베이스 비교는 한 샘플러 안에서 하세요.
- **Spectrum** (체크박스): 대부분의 스텝에서 42개 블록을 모두 돌리는 대신 모델 출력을 예측합니다. 50 스텝, 기본 `spectrum_flex_window 0.5` (Advanced)에서 2.6배 빠르고 VRAM은 0.85 GiB 더 쓰며, 그리드 크기로 보면 50 스텝 전체와 구분하기 어렵습니다. `0.25` 는 2.2배로 더 보수적이고, `1.0` 은 3.1배지만 세부 디테일이 뭉개집니다.
- **compile** (Advanced, 켬): Spectrum 여부와 관계없이 추가로 약 1.37배 빨라지며, 작업마다 몇 초 컴파일합니다. 구도는 컴파일하지 않은 결과와 같지만 세부(머리카락 가닥, 무늬)는 달라지므로, 컴파일한 이미지는 컴파일한 이미지끼리 비교하세요.

| Base, 1024², 50 스텝 | 이미지당 시간 |
|---|---|
| 일반, 컴파일 없음 | 72초 |
| 일반, 컴파일 | 52초 |
| Spectrum, 컴파일 | 20초 |

**비교 방법**: Test 버튼이 이미 LoRA 적용/미적용 이미지를 같은 시드로 렌더링합니다. LoRA 두 개나 에폭끼리 비교할 때는 프롬프트, 시드, 샘플러, compile 설정을 고정하고 `lora` 만 바꾸세요.

---

## 6. 문제 해결

| 증상 | 조치 |
|---|---|
| 학습 중 메모리 부족 | Advanced: `blocks_to_swap 24`, 그다음 `activation_memory_budget 0.1`. 또는 `tune_swap.py` 실행. LoKr/LoHa는 24 / 0.1 필요. `grad_checkpointing` 켬 + `blocks_to_swap 0` 은 swap 없이 들어가는 더 느린 구성 |
| 스타일이 너무 약함 | 에폭을 늘립니다 (측정한 학습은 20). 프롬프트에 캡션과 같은 철자의 작가 태그가 있는지, `prompt_format` 이 `caption_format` 과 같은지 확인 |
| 스타일이 너무 강함, 세트의 서명·워터마크가 나옴, 모든 이미지 구도가 같음 | 앞 에폭의 `_e<N>` 체크포인트를 테스트하거나 `lora_multiplier` 를 낮춥니다. 서명을 베끼기 시작한 것은 실효 lr을 올렸을 때였음; `lr 2e-5`를 시도해 보세요 |
| 세트의 의상이나 색이 다른 캐릭터에 샘 | `caption_format layout_json` 사용 (기본값, 일반 태그보다 누출이 적었음). LoHa는 LoRA나 LoKr로 교체 |
| `caption_format` 을 바꿨음 | 전처리를 다시 실행: 텍스트 캐시는 다시 인코딩되고 잠재는 재사용됨. 상태 줄 개수에는 형식 변경이 반영되지 않음. `prompt_format` 을 맞추고 다시 학습 |
| `layout_json needs the danbooru tag CSV` | `make download-model ARGS=danbooru_tags` |
| Test에서 `No LoRA at …` | 먼저 학습하거나 Test 상자에서 `lora` 를 지정 |

---

## 부록. CLI

GUI가 제출하는 작업과 같으며, 데몬으로 직접 실행할 수도 있습니다. 각 스크립트의 `--help` 에 모든 플래그가 나옵니다. `train.py --config configs/moss/lora.toml` 로 플래그를 TOML 파일에서 읽을 수 있습니다(기본값이 들어 있으니 복사해서 레시피로 쓰세요). 명령줄에 직접 준 플래그가 우선합니다.

```bash
make daemon-run ARGS="--stall-timeout 900 scripts/moss/cache.py --src 'post_image_dataset/resized/<set>' --out 'post_image_dataset/moss/<set>'"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --cache 'post_image_dataset/moss/<set>' --epochs 20"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/lora.toml"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/generate.py --lora output/ckpt_moss/moss_lora.safetensors --spectrum"
make daemon-run ARGS="scripts/moss/tune_swap.py --cache post_image_dataset/moss/<set>"
```

LoKr / LoHa:

```bash
make daemon-run ARGS="scripts/moss/train.py --cache 'post_image_dataset/moss/<set>' --epochs 20 --algo lokr --rank 100000 --factor 8 --blocks_to_swap 24 --activation_memory_budget 0.1"
make daemon-run ARGS="scripts/moss/train.py --cache 'post_image_dataset/moss/<set>' --epochs 20 --algo loha --rank 16 --lr 5e-5 --blocks_to_swap 24 --activation_memory_budget 0.1"
```

더 자세한 측정 결과: [`docs/experimental/moss_lora.md`](../experimental/moss_lora.md).
