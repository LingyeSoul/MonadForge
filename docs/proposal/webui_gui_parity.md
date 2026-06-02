# WebUI Parity Plan — Close GUI Gaps

Three gaps remain between the upstream PySide6 GUI and the WebUI. This plan is written for execution by other sessions — each task is self-contained with exact file paths, patterns to follow, and verification steps.

**Prerequisites:** `uv sync` already done. Dev server: `python -m webui` (backend) + `cd webui/frontend && npm run dev` (frontend).

---

## Task 1: Register `soft_tokens` + distill methods in `_METHOD_ORDER`

**Severity:** Low | **Effort:** ~5 min | **Files:** 1

### Problem

`webui/services/config_service.py` line 93-104 defines `_METHOD_ORDER` with 10 entries but is missing `soft_tokens`. Upstream's `gui/__init__.py` `_METHOD_ORDER` includes it. SPD and Turbo are distill methods (sectioned TOML, not flat) — they need a different treatment (see Task 2).

### Change

In `webui/services/config_service.py`, add `"soft_tokens"` to `_METHOD_ORDER`:

```python
_METHOD_ORDER = (
    "lora",
    "tlora",
    "hydralora",
    "reft",
    "postfix",
    "fera",
    "chimera",
    "soft_tokens",   # <-- ADD
    "ip_adapter",
    "easycontrol",
    "controlnet",
)
```

### Verify

1. Start dev server, open ConfigEditor
2. Method dropdown should now show "soft_tokens"
3. Selecting it should load `configs/methods/soft_tokens.toml` and display its fields

---

## Task 2: SPD/Turbo Distill Config Editor + Training Launch

**Severity:** High | **Effort:** ~2-3 hours | **Files:** 5-6 new/modified

### Problem

The upstream GUI's `distill_tab.py` provides a structured editor for sectioned TOML configs (`configs/methods/spd.toml` and `turbo.toml`) plus train/stop buttons. The WebUI has:
- No API for reading/writing sectioned TOML
- No frontend view for distill config editing
- `exp-spd` not registered in `_COMMAND_DESCRIPTIONS`
- `exp-turbo` registered but no UI launcher
- Guide HTML content already exists in `webui/explanations/guides/{en,cn,ja,ko}/spd.html` and `turbo.html`

### Architecture

Follow the existing patterns exactly:
- **Backend:** `APIRouter()` in a new `webui/api/distill.py`, thin functions delegating to `webui/services/distill_service.py`
- **Frontend:** New `webui/frontend/src/views/DistillView.vue` using `<script setup>`, `ref()`/`reactive()`, Pinia task store
- **TOML read/write:** Use `tomlkit` (already in deps) to preserve inline comments on save round-trip

### Step 2.1: Register task commands

**File:** `webui/api/tasks.py`

Add to `_COMMAND_DESCRIPTIONS` dict (around line 60):

```python
"exp-spd": "SPD distillation training",
```

(`exp-turbo` is already registered at line 60.)

### Step 2.2: Backend — distill service

**File:** `webui/services/distill_service.py` (NEW)

```python
"""Service for reading/writing sectioned distill TOML configs (spd, turbo)."""
```

Functions to implement:

```python
def list_distill_methods() -> list[dict]:
    """Return available distill methods with metadata.

    Scans configs/methods/ for known distill TOML files (spd.toml, turbo.toml).
    Returns list of {"key": "spd", "label": "SPD", "config_path": "configs/methods/spd.toml",
    "task_command": "exp-spd", "sections": [...]}.
    """

def read_distill_config(method: str) -> dict:
    """Read a sectioned distill TOML and return its structure.

    Returns {"sections": [{"name": "network"|"root", "fields": [{"key": str, "value": any, "type": str, "comment": str}]}]}.
    Uses tomlkit to parse, preserving comments for the "comment" field.
    type is one of: "int", "float", "str", "bool", "list".
    "root" section name for top-level keys (no [section] header).
    """

def save_distill_config(method: str, updates: dict[str, dict[str, any]]) -> None:
    """Save field values back to the sectioned TOML, preserving comments.

    updates is {"section_name": {"key": new_value, ...}, ...}.
    Uses tomlkit to parse + mutate + dump, so inline # comments survive.
    Raises ValueError if method is not a known distill method.
    """
```

Implementation notes:
- `_DISTILL_METHODS` dict maps method key to config path + task command:
  ```python
  _DISTILL_METHODS = {
      "spd": {"path": "configs/methods/spd.toml", "task": "exp-spd", "label": "SPD"},
      "turbo": {"path": "configs/methods/turbo.toml", "task": "exp-turbo", "label": "Turbo"},
  }
  ```
- Use `ROOT = Path(__file__).resolve().parent.parent.parent` to resolve config paths (same as config_service.py)
- `read_distill_config` iterates `doc.body` from tomlkit: `(key, item)` pairs where `key is None` means comment/whitespace, `isinstance(item, Table)` means section
- Type detection: `isinstance(v, bool)` (check before int since bool is int subclass), `isinstance(v, int)`, `isinstance(v, float)`, `isinstance(v, str)`, `isinstance(v, list)`
- `save_distill_config` mutates the tomlkit document in-place then writes back

### Step 2.3: Backend — distill API endpoints

**File:** `webui/api/distill.py` (NEW)

```python
"""API endpoints for SPD/Turbo distill config editing."""
```

Endpoints:

```python
router = APIRouter(prefix="/distill", tags=["distill"])

class DistillMethodSummary(BaseModel):
    key: str
    label: str
    config_path: str
    task_command: str

class DistillField(BaseModel):
    key: str
    value: Any
    type: str  # "int" | "float" | "str" | "bool" | "list"
    comment: str = ""

class DistillSection(BaseModel):
    name: str  # "root" or section name like "network"
    fields: list[DistillField]

class DistillConfigResponse(BaseModel):
    method: str
    sections: list[DistillSection]

class DistillSaveRequest(BaseModel):
    updates: dict[str, dict[str, Any]]  # {"section": {"key": value}}

@router.get("/methods", response_model=list[DistillMethodSummary])
def list_methods():
    """List available distill methods."""

@router.get("/config", response_model=DistillConfigResponse)
def get_config(method: str = Query(...)):
    """Read a distill method's sectioned TOML config."""

@router.put("/config")
def save_config(body: DistillSaveRequest, method: str = Query(...)):
    """Save field values to a distill method's TOML config."""
```

### Step 2.4: Register router

**File:** `webui/app.py` (or wherever routers are registered — check existing pattern)

Add:
```python
from webui.api.distill import router as distill_router
app.include_router(distill_router, prefix="/api")
```

### Step 2.5: Frontend — DistillView.vue

**File:** `webui/frontend/src/views/DistillView.vue` (NEW)

Follow `PreprocessView.vue` patterns exactly. Structure:

```vue
<template>
  <v-container fluid>
    <!-- Method selector -->
    <v-row>
      <v-col cols="12">
        <v-card variant="tonal">
          <v-card-title>
            <v-icon start>mdi-flask</v-icon>
            {{ t('distTitle') }}
          </v-card-title>
          <v-card-text>
            <v-select v-model="selectedMethod" :items="methods" item-title="label"
                      item-value="key" :label="t('distMethod')" @update:modelValue="loadConfig" />
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <!-- Guide panel (right side, collapsible) -->
    <!-- Load from /api/docs/guide/{method} or hardcode spd.html/turbo.html URLs -->

    <!-- Config sections -->
    <v-row v-for="section in config.sections" :key="section.name">
      <v-col cols="12">
        <v-card variant="outlined">
          <v-card-title>{{ section.name === 'root' ? t('distGeneral') : section.name }}</v-card-title>
          <v-card-text>
            <v-form>
              <template v-for="field in section.fields" :key="field.key">
                <!-- Render by field.type -->
                <v-text-field v-if="field.type === 'str'" v-model="field.value" :label="field.key"
                              :hint="field.comment" persistent-hint />
                <v-text-field v-else-if="field.type === 'int'" v-model.number="field.value"
                              :label="field.key" type="number" :hint="field.comment" persistent-hint />
                <v-text-field v-else-if="field.type === 'float'" v-model.number="field.value"
                              :label="field.key" type="number" step="any" :hint="field.comment" persistent-hint />
                <v-switch v-else-if="field.type === 'bool'" v-model="field.value"
                          :label="field.key" :hint="field.comment" persistent-hint />
                <v-text-field v-else-if="field.type === 'list'" :model-value="JSON.stringify(field.value)"
                              @update:model-value="v => field.value = JSON.parse(v)"
                              :label="field.key" :hint="field.comment" persistent-hint />
              </template>
            </v-form>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <!-- Action bar -->
    <v-row>
      <v-col cols="12" class="d-flex ga-2">
        <v-btn color="warning" prepend-icon="mdi-content-save" @click="saveConfig"
               :loading="saving">{{ t('distSave') }}</v-btn>
        <v-btn color="success" prepend-icon="mdi-play-circle" @click="startTraining"
               :loading="isRunning">{{ t('distTrain') }}</v-btn>
        <v-btn color="error" prepend-icon="mdi-stop" @click="stopTraining"
               :disabled="!isRunning">{{ t('distStop') }}</v-btn>
      </v-col>
    </v-row>
  </v-container>
</template>
```

Script setup:

```typescript
const methods = ref<DistillMethodSummary[]>([])
const selectedMethod = ref('')
const config = reactive<DistillConfigResponse>({ method: '', sections: [] })
const saving = ref(false)

const taskStore = useTaskStore()

const isRunning = computed(() =>
  taskStore.tasks.some(tp => {
    const method = methods.value.find(m => m.key === selectedMethod.value)
    return method && tp.command === method.task_command && tp.state === 'running'
  })
)

onMounted(async () => {
  const res = await fetch('/api/distill/methods')
  methods.value = await res.json()
  if (methods.value.length) {
    selectedMethod.value = methods.value[0].key
    await loadConfig()
  }
})

async function loadConfig() {
  const res = await fetch(`/api/distill/config?method=${selectedMethod.value}`)
  const data = await res.json()
  Object.assign(config, data)
}

async function saveConfig() {
  saving.value = true
  try {
    // Build updates dict from config.sections
    const updates: Record<string, Record<string, any>> = {}
    for (const section of config.sections) {
      updates[section.name] = {}
      for (const field of section.fields) {
        updates[section.name][field.key] = field.value
      }
    }
    await fetch(`/api/distill/config?method=${selectedMethod.value}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ updates })
    })
  } finally {
    saving.value = false
  }
}

async function startTraining() {
  const method = methods.value.find(m => m.key === selectedMethod.value)
  if (!method) return
  await saveConfig()
  await taskStore.startTask(method.task_command)
}

function stopTraining() {
  const method = methods.value.find(m => m.key === selectedMethod.value)
  if (!method) return
  const task = taskStore.tasks.find(tp => tp.command === method.task_command && tp.state === 'running')
  if (task) taskStore.cancelTask(task.id)
}
```

### Step 2.6: Register route in frontend router

**File:** `webui/frontend/src/router/index.ts`

Add a route for `/distill` pointing to `DistillView.vue`. Check existing route registration pattern.

### Step 2.7: Add i18n keys

**Files:** `webui/frontend/src/i18n/locales/{en,cn,ja,ko}.ts` (or wherever i18n lives)

Add keys:
```typescript
distTitle: 'Distillation' / '蒸馏' / '蒸留' / '증류'
distMethod: 'Method' / '方法' / '方法' / '방법'
distGeneral: 'General' / '通用' / '一般' / '일반'
distSave: 'Save' / '保存' / '保存' / '저장'
distTrain: 'Train' / '训练' / '学習' / '학습'
distStop: 'Stop' / '停止' / '停止' / '중지'
```

### Step 2.8: Add navigation entry

**File:** wherever the sidebar/nav is defined (check `webui/frontend/src/App.vue` or layout component)

Add a nav item for the Distill page with `mdi-flask` icon.

### Verify

1. `python -m webui` + `npm run dev`
2. Navigate to Distill page
3. Method dropdown shows "SPD" and "Turbo"
4. Selecting SPD loads `configs/methods/spd.toml` — 6 sections rendered (root, network, schedule, onpolicy, optim, io)
5. Edit a field (e.g. `iterations`), click Save — verify TOML file retains comments
6. Click Train — `exp-spd` task appears in task monitor
7. Repeat for Turbo (8 sections: root, network, dmd, optim, dpdmd, sampling, mean_var, io)

---

## Task 3: Adapter Dataset State Dashboard

**Severity:** Medium | **Effort:** ~30 min | **Files:** 3

### Problem

Upstream's `adapter_tab.py` shows a dataset state dashboard before training: source image count, caption pairing status, cache coverage breakdown (latents/te/pe counts). The WebUI's `AdapterView.vue` only shows 8 preview thumbnails — no quantitative indicators.

### Step 3.1: Backend — dataset stats endpoint

**File:** `webui/api/preprocess.py` (add endpoint, or create `webui/api/adapter.py`)

Add a new endpoint that returns dataset stats for a given directory:

```python
@router.get("/adapter-stats")
def get_adapter_stats(dir: str = Query(...)):
    """Return dataset statistics for an adapter's source directory.

    Returns:
        {
            "source_count": int,          # images in dir
            "caption_count": int,         # .txt files matching images
            "cache": {
                "latents": int,           # .npz files in cache dir
                "te": int,               # _anima_te.safetensors files
                "pe": int,               # _anima_pe.safetensors files
            }
        }
    """
```

Implementation:
- Count images: `sum(1 for f in Path(dir).iterdir() if f.suffix.lower() in IMAGE_EXTS)`
- Count captions: `sum(1 for f in Path(dir).iterdir() if f.suffix == '.txt')`
- Cache counts: scan `post_image_dataset/lora/` for `{stem}_*.npz`, `{stem}_*_anima_te.safetensors`, `{stem}_*_anima_pe.safetensors` matching stems in the source dir
- Reuse the existing `count_preprocess_caches` pattern from `gui/__init__.py` (line 685-709) — it counts latents/te/pe/masks by suffix matching

### Step 3.2: Frontend — enhance AdapterView.vue

**File:** `webui/frontend/src/views/AdapterView.vue`

For each adapter card (IP-Adapter and EasyControl), add a stats section between the description/alert and the preview thumbnails:

```vue
<!-- Dataset stats (add after the v-alert, before the preview section) -->
<v-row dense class="mt-2" v-if="adapterStats[adapterKey]">
  <v-col cols="auto">
    <v-chip prepend-icon="mdi-image" variant="tonal"
            :color="adapterStats[adapterKey].source_count > 0 ? 'success' : 'default'">
      {{ t('adSourceImages') }}: {{ adapterStats[adapterKey].source_count }}
    </v-chip>
  </v-col>
  <v-col cols="auto">
    <v-chip prepend-icon="mdi-text-box" variant="tonal"
            :color="adapterStats[adapterKey].caption_count > 0 ? 'success' : 'default'">
      {{ t('adCaptions') }}: {{ adapterStats[adapterKey].caption_count }}
    </v-chip>
  </v-col>
  <v-col cols="auto">
    <v-chip prepend-icon="mdi-database" variant="tonal"
            :color="adapterStats[adapterKey].cache.latents > 0 ? 'success' : 'default'">
      {{ t('adCacheLatents') }}: {{ adapterStats[adapterKey].cache.latents }}
    </v-chip>
  </v-col>
  <v-col cols="auto">
    <v-chip prepend-icon="mdi-brain" variant="tonal"
            :color="adapterStats[adapterKey].cache.te > 0 ? 'success' : 'default'">
      {{ t('adCacheTE') }}: {{ adapterStats[adapterKey].cache.te }}
    </v-chip>
  </v-col>
  <v-col cols="auto">
    <v-chip prepend-icon="mdi-vector-square" variant="tonal"
            :color="adapterStats[adapterKey].cache.pe > 0 ? 'success' : 'default'">
      {{ t('adCachePE') }}: {{ adapterStats[adapterKey].cache.pe }}
    </v-chip>
  </v-col>
</v-row>
```

Script changes:

```typescript
const adapterStats = reactive<Record<string, AdapterStats>>({})

async function fetchStats(dir: string, key: string) {
  try {
    const res = await fetch(`/api/preprocess/adapter-stats?dir=${encodeURIComponent(dir)}`)
    if (res.ok) adapterStats[key] = await res.json()
  } catch { /* silent */ }
}

onMounted(() => {
  fetchStats('image_dataset', 'ip')
  fetchStats('easycontrol-dataset', 'ec')
  // ... existing preview fetch logic
})
```

### Step 3.3: Add i18n keys

```typescript
adSourceImages: 'Images' / '图片' / '画像' / '이미지'
adCaptions: 'Captions' / '字幕' / 'キャプション' / '캡션'
adCacheLatents: 'Latents' / '潜变量' / '潜在変数' / '잠재 변수'
adCacheTE: 'Text Enc' / '文本编码' / 'テキスト' / '텍스트'
adCachePE: 'PE' / 'PE特征' / 'PE特徴' / 'PE 특징'
```

### Verify

1. Open Adapter page
2. IP-Adapter card shows: source image count, caption count, cache breakdown (latents/te/pe)
3. Chips are green when count > 0, neutral when 0
4. Run `make preprocess` then refresh — counts should update

---

## Execution Order

1. **Task 1** (1 line change) — do first, unblocks soft_tokens in WebUI
2. **Task 2** (new feature) — largest piece, can be done as one PR
3. **Task 3** (enhancement) — independent of Task 2, can be parallelized

Tasks 1 and 3 are independent. Task 2 is self-contained. All three can be separate commits on the same branch.

---

## Post-execution Cleanup

After all three tasks are verified, the `gui/` directory can be fully removed:

```bash
git rm -r gui/
```

The PySide6 GUI is fully superseded by the WebUI. The only GUI files that would remain are `gui/tabs/distill_tab.py` and `gui/tabs/methods_tab.py` — both replaced by Task 2.
