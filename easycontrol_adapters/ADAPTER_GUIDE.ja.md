# 自前の EasyControl アダプターを作る

このガイドでは、Anima に**新しい EasyControl コントロールタスク**を自分用に追加する方法を説明します。git へのコントリビューションとしてではなく、`easycontrol_adapters/<your_task>/` の下に置くローカルアダプターとして追加します。正準的な例である **colorize** (`easycontrol_adapters/colorization/`) をステップごとに辿り、自分のタスク向けに何を変えればよいかを正確に示します。

一つだけ覚えておくことがあります:

> **モデルコードを書くわけではありません。** ネットワーク、フォワードパス、`b_cond` ゲート、推論キャッシュ — これらはすべて出荷済みで、あらゆるコントロールタスクで共有されています。タスクごとに変わるのは **参照画像をどう作るか** だけです。このガイドの残りはすべてその周りの配管です。

---

## 0. アイデア — EasyControl アダプターとは何か

EasyControl は **参照画像** を使って生成を導きます。参照は VAE を通じて *cond トークン* に変換され、生成中の画像と並んで流れていきます。各ステップでモデルは両方を見ます。(完全なアーキテクチャは `docs/experimental/easycontrol.md`。)

デフォルトの EasyControl は**同じ画像**を参照とターゲットの両方に使います — つまりただコピーすることを学習します。コントロールタスクはこれを変えます: 各ターゲットを**別の**参照とペアにすることで、モデルがコピーではなく `reference → target` を学習するようになります。

| | デフォルト EasyControl | コントロールタスク (例: colorize) |
|---|---|---|
| ターゲット (ほしいもの) | 画像 X | カラー画像 X |
| 参照 (ヒント) | 画像 X (同一) | X の*変換*バージョン (X の白黒マンガ) |
| 学習すること | コピー | manga → color |
| テキスト | 完全なキャプション | (任意) 短いキャプション |

colorize のやり方 — 再利用できるアイデアです: **実際の白黒マンガにはカラーバージョンがないため**、`(白黒, カラー)` ペアを集めて作ることはできません。そこで方向を反転させます — すでに手元にあるカラー画像をターゲットとし (それが欲しい結果なので)、各画像から白黒の参照を**作ります** (線画 + スクリーントーン、アルゴリズムで)。肝心なのは、作った白黒が推論時に実際に入力するものに見えるようにすることです。`depth → image` や `pose → image` を構築するなら、「参照を作る」ステップは学習画像に対して実行する深度推定器やポーズ検出器になります。

つまりあなたの仕事は: **関数 `target_image → reference_image` を書き、その出力をキャッシュし、データセットをそれに向け、config と一行の名前登録を追加すること** です。

---

## 1. 触れる四つのもの

`<task>` という名前のアダプターを追加するには、以下の四つを正確に作成/編集します:

| # | もの | colorize バージョン | 役割 |
|---|---------|-------------------|--------------|
| 1 | `easycontrol_adapters/<task>/` プロジェクト | `colorization/` (`mangafy*.py`, `color_caption.py`, `prep.py`) | 参照画像を作ってキャッシュする (任意で短いテキストキャッシュも) |
| 2 | `configs/datasets/<task>.toml` | `configs/easycontrol/colorize.toml` | 各ターゲットを参照と **cond_cache_dir** でペアにするデータセット |
| 3 | `configs/methods/<task>.toml` (+ `configs/gui-methods/<task>.toml`) | `configs/easycontrol/colorize.toml` | config — データセットを指し、LR / epochs / `network_args` を設定 |
| 4 | `scripts/tasks/{training,inference}.py` | `_EASYADAPTERS = {"colorize"}` + 分岐 | `EASYADAPTER=<task>` を `make easycontrol*` コマンドで動くようにする |

順番に見ていきます。どれも `networks/` を触りません。

---

## 2. もの 1 — アダプタープロジェクト (`easycontrol_adapters/<task>/`)

本当の作業はここにあります。二つの仕事をします: **参照画像を作る**ことと、**キャッシュする**ことです。(任意で三つ目として、タスク固有のテキストキャッシュを構築します。)

### 2a. 参照を作る関数

普通の関数です: カラー画像を RGB `uint8 (H,W,3)` と seed で受け取り、**同じサイズ**の参照画像を RGB `uint8 (H,W,3)` で返します。(同じサイズが重要です — §3 参照。) 同じ seed に対して同じ出力を返すことで、再実行と並列ワーカーがすべて一致します。

colorize ではこれが `mangafy.py::mangafy_array` (と GPU バージョン `mangafy_gpu.py::mangafy_array_gpu`) です:

```python
# easycontrol_adapters/colorization/prep.py
Screener = Callable[[np.ndarray, int], np.ndarray]  # (img_rgb, seed) → cond_rgb
```

コピーする価値のある四つのポイント:

- **ファイル名から決定論的に seed を作ってください。** colorize は `zlib.crc32(stem)` を使います — プロセスごとに変わって並列ワーカーが食い違う Python の `hash()` *ではありません*。バリエーションを加えてもかまいません (colorize はスクリーントーンの角度をページごとに変えています) — seed から導出するだけで再現性が保たれます。
- **重いものは遅延インポートしてください。** colorize には三つのエンジン (`cv2` / `gpu` / `sd`) があり、ページが実際にそちらへルーティングされた場合にのみ 3.5 GB の SD モデルをロードします。ビルダーがモデル (深度ネット、線抽出器) を必要とするなら、使うときだけインポートしてください。
- **ダウンロードなしのフォールバックは大きな利点です。** colorize の `cv2`/`gpu` エンジンはダウンロードが一切不要で、クリーンなチェックアウトで追加ダウンロードなしに prep + 学習ができます。可能ならこのようなフォールバックを用意してください。
- **ファイルは原子的に書き込んでください。** colorize の `_save_png_atomic` は一時ファイルに書いてから名前を変えます。これがないと中断された実行が半分書かれた PNG を残す可能性があり、「あれば飛ばす」チェックがそのファイルを永遠に信じてしまいます。実際に起きるバグです — このパターンをコピーしてください。

参照が**すでにディスクにある場合** (本物の深度マップ、本物のスケッチ)、ビルドステップをまるごとスキップしてそれらを直接キャッシュするだけで済みます。ターゲットから参照を導出しなければならない場合にのみビルドが必要です。

### 2b. (任意) 短いテキストキャッシュ

colorize は参照を変えるだけでなく、**キャプションを色の単語だけに絞り込み**もします (`color_caption.py::filter_to_colors`)。その理由は理解する価値があります: 参照 (線画 + スクリーントーン) がすでに *形とレイアウトのすべて* をエンコードしているので、テキストがまだ言うべきことは白黒では表せない一つのもの — **色** — だけです。キャプションを色の単語に絞ると、残ったすべての単語がモデルが参照から得られない情報になり、長いキャプションに埋もれた色の単語による弱いステアリングの代わりに、強い `prompt → color` の結びつきが生まれます。

自分のタスクについて同じ問いを立ててください: **参照がすでに決めることは何で、テキストが決めるべき曖昧なことは何か?** `pose → image` 参照はポーズを固定しますが、服装や設定は固定しません — おそらく *完全な* キャプションを保持するでしょう。`depth → image` 参照はレイアウトを固定しますが、アイデンティティや色は固定しません。colorize の「まだ曖昧なことへキャプションを絞り込む」やり方は考慮すべき *パターン* であってルールではありません — 多くのアダプターはキャプションをそのままにしてテキストキャッシュを完全にスキップします (データセットから `text_cache_dir` を省くだけです — §4)。

キャプションを絞り込む場合、colorize の二つの独立したノブに注意してください (混同しないでください):

- **`caption_dropout_rate`** — 自動着色の *フロア (下限)*。約 5% の学習ステップでキャプションをまるごとドロップし、空プロンプトのときの動作を学習させます。**低く** 保ってください (`0.05`); 高い値は無条件パスを過学習させてプロンプトを弱くします。
- **`use_shuffled_caption_variants`** — 完全対部分の *バランス*。テキストキャッシュは複数のバージョンを持ちます (v0 = 完全な色のセット、v1+ = 各単語が約半分の確率でドロップされたシャッフル版)。ローダーは v0 を 20%、v1+ を 80% 引くので、"pink hair" だけのような部分プロンプトでも機能します。

### 2c. `prep.py` — キャッシュビルダー

三つのステージ、それぞれ **べき等 (idempotent)** (完了済みの作業はスキップするので、再実行しても安全です):

1. **ビルド** — `--src` (`post_image_dataset/resized`) 以下のすべてのカラー画像を走査し、ビルダー関数を実行して、ソースのレイアウトをミラーリングした `--staging` フォルダに参照 PNG を書き込みます。
2. **エンコード** — ステージングされた参照を `library.preprocess.cache_latents` で `--cond_cache_dir` へ VAE エンコードします。各画像の **ネイティブサイズ** で行うことで、参照 latent がターゲット latent と同じ形になります。通常のキャッシュと同じ `{stem}_{WxH}_anima.npz` フォーマットです。
3. **(任意) テキスト** — キャプションをフィルターを通して `--text_cache_dir` へ再エンコードします。`library.preprocess.cache_text_embeddings` に `caption_transform=` (と `caption_shuffle_variants` / `caption_tag_dropout_rate`) を与えて行います。

既存のライブラリヘルパーを使ってください — `library.preprocess.{cache_latents, cache_text_embeddings, tqdm_progress}` と `library.preprocess._dataset.walk_images`。エンコードループを自前で書かないこと; `prep.py` は `scripts/preprocess/*.py` と同じく、これらの上の薄いシェルです。

colorize が対処している、その構造をコピーすれば無償で受け継げる二つの正確性の落とし穴:

- **Stem が一致していなければなりません。** テキストステージはキャプションマスター (`image_dataset/`、`resized/` と同じレイアウト) から `.txt` キャプションを読むので、生成されるテキストキャッシュのファイル名がローダーの `image_dir=post_image_dataset/resized` を参照する形と一致します。キャッシュのファイル名がターゲットの stem と合わなければ、ローダーは黙ってペアにしません。
- **uncond サイドカー。** colorize のテキストステージは共有の `T5("")` 空プロンプトサイドカーがなければ再作成します。テキストキャッシュを構築してキャプションドロップアウトを使うなら同じことをしてください (`library.inference.uncond.stage_uncond_sidecar_with_models`)。

---

## 3. 破ってはいけない唯一のルール — 参照とターゲットのトークン数が一致していなければならない

DiT は Anima の **ネイティブ形状バケッティング** (二つのトークン数ファミリー、4032 と 4200; CLAUDE.md と `docs/experimental/easycontrol.md` の "Cond token count" を参照) で動作します。**パディングのノブはありません** — 参照は latent の実際のトークン数で実行されます。

これこそが §2a で参照を入力と **同じサイズ** にするよう言い、§2c で **ネイティブサイズ** でエンコードすると言う理由です: 両方を守れば参照 latent が自動的にターゲットと同じバケットに入り、すべてが機能します。より小さい参照が欲しければ (まったく問題ありません — 小さいほどメモリと速度が節約できます)、**画像** レベルでエンコードの前に縮小してください。そうすれば latent は依然として実際のバケットに入ります。ネットワーク内でトークン数を制限しようとしないでください。

---

## 4. もの 2 — データセット (`configs/datasets/<task>.toml`)

このファイルが参照をターゲットと別のものにします。通常のデータセット (`[general]` + `[[datasets]]` + `[[datasets.subsets]]`) に**追加のノブが一つ**付いたものです: `cond_cache_dir` (と任意で `text_cache_dir`)。

colorize (`configs/easycontrol/colorize.toml`)、コメント付き:

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

各リダイレクトの役割:

- **`cond_cache_dir`** — これがコントロールタスクを成り立たせる唯一のノブです。ローダーはここで各ターゲットを stem で参照 latent とマッチングします。EasyControl の二ストリームフォワードが参照として使うものです。
- **`text_cache_dir`** — **テキストキャッシュのみ**をリダイレクトします (latent は依然として `cache_dir` から来ます)。完全なキャプションを使うならまるごと省いてください — するとローダーが共有テキストキャッシュを使い、`prep.py` のテキストステージをスキップできます。
- **`flip_aug = false`** — 必須です。反転したターゲットには反転した参照 latent が必要ですが、それはキャッシュしていません。フリップはオフのままにしてください。

colorize がターゲット latent とテキストに**共有の** `post_image_dataset/lora` キャッシュを再利用していることに注意してください — `make preprocess` がすでに構築済みで、何も再エンコードしません。あなたのアダプターは *参照* キャッシュ (と場合によっては短いテキストキャッシュ) だけを追加します。

---

## 5. もの 3 — メソッド config (`configs/methods/<task>.toml`)

`configs/easycontrol/easycontrol.toml` のほぼコピーです。唯一の構造的な変更は自分のデータセットを指す `dataset_config` だけで、残りはハイパーパラメータです。

colorize (`configs/easycontrol/colorize.toml`)、重要な行:

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

自分のタスクについて考えるべきノブ:

- **`b_cond_init`** — 学習開始時に参照がどれだけ影響するか。`-10` は step 0 で参照がほとんど寄与しないことを意味します (モデルは最初は通常の DiT のように動き、そこから参照に頼る方法を学習します); `docs/experimental/easycontrol.md` の "Step-0 baseline equivalence" がその理由を説明しています。colorize は参照が早く効くよう `-6` に緩めています — 参照が強いタスクならそれで十分です。どちらにせよ学習可能です。
- **`easycontrol_cond_noise_max`** — 学習中に参照に加えるノイズの量 (σ は `U(0, max)` からサンプリング、`cond + σ·ε` として適用)。`0` は参照を完璧な設計図として扱います; 大きな値は参照を大雑把な「ヒント」に劣化させ、テキストが残りの詳細を担うよう強制します。colorize は `0.02` を使います (非常に小さい — 線画*こそが*信号です)。デフォルトの easycontrol.toml は `0.3` を使います。
- **`easycontrol_drop_p`** — image-CFG のために参照をどれくらいの頻度でドロップするか。colorize は `0` を使います (常に参照が欲しい); デフォルトは `0.1` です。
- **`output_name`** — 一意でなければなりません; 推論ステップがこの名前で最新のチェックポイントを探します (§6)。

任意で `configs/gui-methods/<task>.toml` も追加できます — `[variant]` ブロック (`family = "easycontrol"`、`label`、`description`、`order`) を持つ独立したバージョン (トグルブロックなし) で、GUI の EasyControl ドロップダウンに表示されます。`configs/gui-methods/easycontrol.toml` を参照してください。CLI からのみ実行するならスキップしてください。

---

## 6. もの 4 — `EASYADAPTER=<task>` を動くようにする

`make easycontrol*` コマンドは `EASYADAPTER` 環境変数で切り替えます。三つの小さな編集で `EASYADAPTER=<task>` が自分の config、prep、チェックポイントを使うようになります。

**`scripts/tasks/training.py` で:**

1. 許可リストに名前を追加します:
   ```python
   _EASYADAPTERS = {"colorize", "<task>"}   # was {"colorize"}
   ```
   (`_easyadapter()` がこのセットに対して検証し、タイポでエラーになります。)

2. `cmd_easycontrol_preprocess` で前処理を自分の `prep.py` にルーティングします:
   ```python
   adapter = _easyadapter()
   if adapter == "colorize":
       run([PY, "easycontrol_adapters/colorization/prep.py", *extra]); return
   if adapter == "<task>":
       run([PY, "easycontrol_adapters/<task>/prep.py", *extra]); return
   ```

3. 学習自体は **編集不要です** — `cmd_easycontrol` はすでに `train(_easyadapter() or "easycontrol", extra)` を呼んでいるので、名前が許可リストに入りさえすれば `EASYADAPTER=<task>` が自動的に `configs/methods/<task>.toml` を実行します。

4. (ビルダーがダウンロードを必要とする場合のみ) `cmd_easycontrol_download` に自分の重みフェッチタスクを指す分岐を追加します。

**`scripts/tasks/inference.py`** (`cmd_test_easycontrol`) で: セレクターは現在 colorize をハードコードしています。自分のタスク向けにいくつかの colorize 専用の値を汎用化してください — チェックポイント名、出力フォルダ、フォールバック参照フォルダ、空プロンプトのデフォルト:

```python
adapter = (os.environ.get("EASYADAPTER") or "").strip()
is_colorize = adapter == "colorize"
weight_name = "anima_colorize" if is_colorize else "anima_easycontrol"
out_sub     = "colorize"       if is_colorize else "easycontrol"
ref_fallback_dir = (ROOT/"post_image_dataset"/"resized") if is_colorize else (ROOT/"easycontrol-dataset")
```

自分の `adapter == "<task>"` ケースを隣に追加してください (weight 名は config の `output_name` と一致させてください)。自分のタスクが colorize のように空プロンプトのデフォルトと特定フォルダからの参照を使いたいなら、下方の `is_colorize` 分岐も真似てください。

---

## 7. 実行する

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

前提条件: `make preprocess` を一度実行し、共有のターゲット latent とテキストキャッシュが `post_image_dataset/lora` に存在するようにしてください (あなたのアダプターはそれらを再利用します)。

### 推論のヒント (colorize から得た知見)

- **本物の in-distribution な参照を入力してください。** 推論時、参照はそのまま VAE エンコードされます — ビルドステップはありません。colorize は本物のスクリーントーン付き白黒ページを入力します; ありきたりなグレースケール写真は分布外で品質が落ちます。ビルダーが *模倣しようとした* ものが、推論が受け取ることを期待するものです。
- **`--easycontrol_image_match_size`** — 参照のアスペクト比に合うトークンバケットを選び、縦長のページがつぶれないようにします。colorize はこれを強制オンにします。
- **`--easycontrol_scale`** (`EC_SCALE=`、参照にどれだけ従うか) — 学習デフォルトは `1.0`; 参照が強すぎると感じたら上げ (1.1–1.2)、より緩い出力を求めるなら下げます (0.7–0.8)。
- **`--guidance_scale`** — テキストの設定と連動します。colorize: 空プロンプト → 低い CFG (1.0–1.5、押し向ける先がない); テキストプロンプト → 高め (3.0–4.5、プロンプトを実際に効かせるのがこれ)。

---

## 8. チェックリスト

- [ ] `easycontrol_adapters/<task>/` に決定論的で原子的書き込みのビルダー
      (`(img, seed) → reference`、同じサイズ) と、べき等な `prep.py`
      (build → encode → 任意の text)。
- [ ] 参照を **ネイティブサイズ** でエンコードし、トークン数がターゲットのバケットと一致すること (§3)。
- [ ] `configs/datasets/<task>.toml` に `cond_cache_dir` (+ 任意の
      `text_cache_dir`)、`flip_aug = false`、共有ターゲットキャッシュを再利用。
- [ ] `configs/methods/<task>.toml` → データセットを指し、一意な
      `output_name`、`network_module = "networks.methods.easycontrol"`、
      `use_easycontrol = true`。(任意で GUI バリアント。)
- [ ] `EASYADAPTER=<task>` を `_EASYADAPTERS` に追加 + 前処理の分岐
      (training.py) + inference.py での汎用化されたチェックポイント/出力/フォールバック。
- [ ] 完全実行の前に `--limit 8` のステージングバッチを目視で確認済み。

---

## 9. さらに読むには

- **`easycontrol_adapters/colorization/README.md`** — colorize の設計ノート全文
  (キャプションポリシー、スクリーントーンの帯、Phase B ロードマップ)。上記すべてのリファレンス実装。
- **`docs/experimental/easycontrol.md`** — ネットワーク自体: 二ストリームフォワード、
  `b_cond` step-0 ベンチ、推論キャッシュ、メモリ使用量、制限事項。`network_args` を
  触れる前に読んでください。
- **`networks/methods/easycontrol.py`** — `EasyControlNetwork` とパッチされた
  `Block.forward`。新しいアダプターのためにこれを編集する**必要はない**はずです; 必要だと思ったら、タスクの本当の違いが参照画像にあるのかを再確認してください。
- **`networks/CLAUDE.md`** — モジュール別マップとディスパッチルール。
