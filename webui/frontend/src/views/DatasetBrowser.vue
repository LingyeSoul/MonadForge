<template>
  <v-container fluid class="pa-4">
    <!-- Header row -->
    <v-row align="center" class="mb-4">
      <v-col cols="auto">
        <div class="text-h5">{{ t('dsTitle') }}</div>
        <div class="text-body-2 text-medium-emphasis">{{ t('dsSubtitle') }}</div>
      </v-col>
    </v-row>

    <!-- Toolbar: directory, search, sort, view toggle -->
    <v-card variant="tonal" class="mb-4 pa-3">
      <v-row align="center" dense>
        <v-col cols="12" sm="4" md="3">
          <v-select
            v-model="directory"
            :items="directories"
            item-title="name"
            item-value="name"
            :label="t('dsDirectory')"
            variant="outlined"
            density="compact"
            hide-details
            prepend-inner-icon="mdi-folder-outline"
            :loading="loadingDirs"
            @update:model-value="onDirectoryChange"
          />
        </v-col>
        <v-col cols="12" sm="4" md="4">
          <v-text-field
            v-model="search"
            prepend-inner-icon="mdi-magnify"
            :label="t('dsSearch')"
            variant="outlined"
            density="compact"
            hide-details
            clearable
            @update:model-value="onSearchChange"
          />
        </v-col>
        <v-col cols="auto">
          <v-btn
            :icon="sortDesc ? 'mdi-sort-descending' : 'mdi-sort-ascending'"
            variant="text"
            density="compact"
            :title="sortDesc ? t('dsSortAsc') : t('dsSortDesc')"
            @click="toggleSort"
          />
        </v-col>
        <v-col cols="auto">
          <v-btn
            icon="mdi-refresh"
            variant="text"
            density="compact"
            :title="t('dsReload')"
            :loading="loadingImages"
            @click="loadImages"
          />
        </v-col>
        <v-col cols="auto">
          <v-btn-toggle v-model="viewMode" mandatory density="compact" variant="outlined">
            <v-btn value="grid" icon="mdi-view-grid" />
            <v-btn value="list" icon="mdi-view-list" />
          </v-btn-toggle>
        </v-col>
        <v-spacer />
        <v-col cols="auto">
          <span class="text-caption text-medium-emphasis">
            {{ totalLabel }}
          </span>
        </v-col>
      </v-row>
    </v-card>

    <!-- Grid view -->
    <v-row v-if="viewMode === 'grid' && images.length > 0">
      <v-col
        v-for="img in images"
        :key="img.path"
        cols="6"
        sm="4"
        md="3"
        lg="2"
      >
        <v-card
          class="mx-auto"
          variant="tonal"
          :class="{ 'border-primary': selectedImage?.path === img.path }"
          @click="selectImage(img)"
        >
          <v-img
            :src="imageUrl(img)"
            :alt="img.filename"
            aspect-ratio="1"
            cover
            class="bg-grey-darken-3"
          >
            <template #placeholder>
              <div class="d-flex align-center justify-center fill-height">
                <v-progress-circular indeterminate color="primary" size="24" />
              </div>
            </template>
            <template #error>
              <div class="d-flex align-center justify-center fill-height">
                <v-icon icon="mdi-image-broken" size="32" color="grey" />
              </div>
            </template>
            <div v-if="img.has_mask" class="mask-badge">
              <v-icon icon="mdi-checkerboard" size="14" />
            </div>
          </v-img>
          <v-card-subtitle class="text-caption text-truncate pa-2">
            {{ img.stem }}
          </v-card-subtitle>
        </v-card>
      </v-col>
    </v-row>

    <!-- List view -->
    <v-data-table
      v-if="viewMode === 'list' && images.length > 0"
      :items="images"
      :headers="listHeaders"
      item-key="path"
      hover
      density="compact"
      @click:row="onRowClick"
    >
      <template #item.preview="{ item }">
        <v-img
          :src="imageUrl(item)"
          width="48"
          height="48"
          cover
          class="rounded my-1"
        >
          <template #error>
            <v-icon icon="mdi-image-broken" size="20" color="grey" />
          </template>
        </v-img>
      </template>
      <template #item.caption="{ item }">
        <span class="text-truncate d-inline-block" style="max-width: 500px">
          {{ item.caption ? item.caption.slice(0, 120) : '—' }}
        </span>
      </template>
      <template #item.has_mask="{ item }">
        <v-icon
          v-if="item.has_mask"
          icon="mdi-checkerboard"
          size="small"
          color="warning"
        />
      </template>
    </v-data-table>

    <!-- Pagination -->
    <div v-if="totalPages > 1" class="d-flex justify-center mt-4">
      <v-pagination
        v-model="page"
        :length="totalPages"
        :total-visible="7"
        density="compact"
        rounded="circle"
        @update:model-value="onPageChange"
      />
    </div>

    <!-- Empty state -->
    <div v-if="!loadingImages && images.length === 0" class="text-center pa-12">
      <v-icon icon="mdi-image-off-outline" size="64" color="grey" class="mb-4" />
      <div class="text-h6 text-medium-emphasis">{{ t('dsNoImages') }}</div>
      <div class="text-body-2 text-medium-emphasis" v-html="t('dsNoImagesHint')" />
    </div>

    <!-- Caption editor dialog -->
    <v-dialog v-model="editorDialog" max-width="900" :persistent="isDirty">
      <v-card v-if="selectedImage">
        <v-card-title class="d-flex align-center ga-2">
          <v-icon icon="mdi-image" />
          <span class="text-truncate">{{ selectedImage.filename }}</span>
          <v-chip v-if="selectedImage.has_mask" size="small" color="warning" variant="tonal">
            {{ t('dsHasMask') }}
          </v-chip>
          <v-spacer />
          <v-btn
            v-if="selectedImage.has_mask"
            :icon="showMask ? 'mdi-eye' : 'mdi-eye-off'"
            variant="text"
            size="small"
            :title="t('dsToggleMask')"
            @click="showMask = !showMask"
          />
        </v-card-title>
        <v-card-text>
          <v-row>
            <v-col cols="12" md="6">
              <div class="image-preview-container">
                <v-img
                  :src="showMask && selectedImage.has_mask ? maskUrl(selectedImage) : imageUrl(selectedImage)"
                  class="rounded"
                  contain
                  style="max-height: 400px"
                >
                  <template #error>
                    <div class="d-flex align-center justify-center fill-height pa-4">
                      <v-icon icon="mdi-image-broken" size="48" color="grey" />
                    </div>
                  </template>
                </v-img>
              </div>
            </v-col>
            <v-col cols="12" md="6">
              <v-textarea
                v-model="editCaption"
                :label="t('dsCaption')"
                variant="outlined"
                rows="10"
                auto-grow
                :loading="saving"
                :hint="captionHint"
                persistent-hint
                class="caption-editor"
              />
              <!-- Tag preview -->
              <div v-if="editCaption" class="tag-preview mt-2">
                <div class="text-caption text-medium-emphasis mb-1">{{ t('dsTagPreview') }}</div>
                <div class="tag-preview-content">
                  <span
                    v-for="(tag, ti) in parsedTags"
                    :key="ti"
                    class="tag-chip"
                    :class="tagClass(tag)"
                  >{{ tag }}</span>
                </div>
              </div>
            </v-col>
          </v-row>
        </v-card-text>
        <v-divider />
        <v-card-actions class="pa-3">
          <v-btn
            variant="text"
            @click="openVersions"
          >
            <v-icon icon="mdi-history" class="mr-1" />
            {{ t('dsVersions') }}
          </v-btn>
          <v-btn
            variant="text"
            :disabled="!isDirty"
            @click="revertCaption"
          >
            {{ t('dsRevert') }}
          </v-btn>
          <v-spacer />
          <v-btn variant="text" @click="tryCloseEditor">
            {{ t('dsClose') }}
          </v-btn>
          <v-btn
            color="primary"
            :disabled="!isDirty"
            :loading="saving"
            @click="saveCaption"
          >
            <v-icon icon="mdi-content-save" class="mr-1" />
            {{ t('dsSaveCaption') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Unsaved changes confirmation -->
    <v-dialog v-model="discardDialog" max-width="400">
      <v-card>
        <v-card-title>{{ t('dsUnsavedTitle') }}</v-card-title>
        <v-card-text>{{ t('dsUnsavedBody') }}</v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="discardDialog = false">{{ t('dsCancel') }}</v-btn>
          <v-btn color="warning" variant="text" @click="discardAndClose">{{ t('dsDiscard') }}</v-btn>
          <v-btn color="primary" @click="saveAndClose">{{ t('dsSaveCaption') }}</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Version history dialog -->
    <v-dialog v-model="versionsDialog" max-width="1000">
      <v-card v-if="selectedImage">
        <v-card-title class="d-flex align-center">
          <v-icon icon="mdi-history" class="mr-2" />
          {{ t('dsVersionHistory') }} — {{ selectedImage.filename }}
        </v-card-title>
        <v-card-text>
          <div v-if="versions.length === 0" class="text-center text-medium-emphasis pa-8">
            {{ t('dsNoVersions') }}
          </div>
          <v-row v-else>
            <!-- Version list -->
            <v-col cols="12" md="4" style="max-height: 500px; overflow-y: auto">
              <v-list density="compact">
                <v-list-item
                  v-for="(v, i) in versions"
                  :key="i"
                  :active="selectedVersion === i"
                  @click="selectedVersion = i"
                >
                  <template #prepend>
                    <v-icon icon="mdi-clock-outline" size="small" />
                  </template>
                  <v-list-item-title class="text-caption">{{ v.ts }}</v-list-item-title>
                  <v-list-item-subtitle class="text-truncate">{{ v.text.slice(0, 80) }}</v-list-item-subtitle>
                  <template #append>
                    <v-btn size="x-small" variant="text" color="primary" @click.stop="restoreVersion(v.text)">
                      {{ t('dsRestore') }}
                    </v-btn>
                  </template>
                </v-list-item>
              </v-list>
            </v-col>

            <!-- Diff display -->
            <v-col cols="12" md="8">
              <div v-if="selectedVersion >= 0 && selectedVersion < versions.length">
                <div class="text-subtitle-2 mb-2 d-flex align-center ga-2">
                  <v-chip size="small" color="success" variant="tonal">
                    +{{ diffStats.added }} {{ t('dsDiffInsertions') }}
                  </v-chip>
                  <v-chip size="small" color="error" variant="tonal">
                    -{{ diffStats.removed }} {{ t('dsDiffDeletions') }}
                  </v-chip>
                </div>
                <pre class="diff-output"><template v-for="(line, li) in diffLines" :key="li"><span :class="diffLineClass(line)">{{ line }}</span>
</template></pre>
              </div>
              <div v-else class="text-center text-medium-emphasis pa-8">
                {{ t('dsDiffSelectVersion') }}
              </div>
            </v-col>
          </v-row>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="versionsDialog = false">{{ t('dsClose') }}</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useI18n } from '../composables/useI18n'

interface ImageItem {
  path: string
  filename: string
  stem: string
  caption: string | null
  has_mask: boolean
}

interface Directory {
  name: string
  path: string
}

interface VersionEntry {
  ts: string
  text: string
}

const { t } = useI18n()

// ── state ─────────────────────────────────────────────────────

const directories = ref<Directory[]>([])
const directory = ref('image_dataset')
const images = ref<ImageItem[]>([])
const search = ref('')
const sortDesc = ref(false)
const viewMode = ref<'grid' | 'list'>('grid')
const page = ref(1)
const pageSize = ref(50)
const total = ref(0)
const totalPages = ref(0)

const loadingDirs = ref(false)
const loadingImages = ref(false)

// Editor
const editorDialog = ref(false)
const selectedImage = ref<ImageItem | null>(null)
const editCaption = ref('')
const diskCaption = ref('') // last known on-disk value
const saving = ref(false)
const showMask = ref(false)

// Unsaved
const discardDialog = ref(false)
let _pendingClose = false

// Version history
const versionsDialog = ref(false)
const versions = ref<VersionEntry[]>([])
const selectedVersion = ref(-1)

// ── computed ──────────────────────────────────────────────────

const isDirty = computed(() => editCaption.value !== diskCaption.value)

const captionHint = computed(() => {
  if (!isDirty.value) return ''
  const added = editCaption.value.length - diskCaption.value.length
  if (added > 0) return `+${added} chars`
  return `${added} chars`
})

const totalLabel = computed(() => {
  if (total.value === 0) return t('dsNoImages')
  if (totalPages.value <= 1) return `${total.value} images`
  return `${total.value} images · p${page.value}/${totalPages.value}`
})

const listHeaders = computed(() => [
  { title: '', key: 'preview', width: '60px', sortable: false },
  { title: t('dsFilename'), key: 'filename' },
  { title: t('dsCaption'), key: 'caption', sortable: false },
  { title: t('dsMask'), key: 'has_mask', width: '60px', sortable: false },
])

// ── URL helpers ───────────────────────────────────────────────

function imageUrl(img: ImageItem): string {
  return `/api/images/file/${encodeURIComponent(img.path)}?directory=${encodeURIComponent(directory.value)}`
}

function maskUrl(img: ImageItem): string {
  return `/api/images/mask-file/${encodeURIComponent(img.path)}?directory=${encodeURIComponent(directory.value)}`
}

// ── data loading ──────────────────────────────────────────────

async function loadDirectories() {
  loadingDirs.value = true
  try {
    const res = await fetch('/api/images/directories')
    directories.value = await res.json()
    if (directories.value.length > 0 && !directories.value.find(d => d.name === directory.value)) {
      directory.value = directories.value[0].name
    }
  } catch {
    directories.value = [{ name: 'image_dataset', path: '' }]
  } finally {
    loadingDirs.value = false
  }
}

async function loadImages() {
  loadingImages.value = true
  try {
    const params = new URLSearchParams({
      directory: directory.value,
      search: search.value,
      sort_desc: String(sortDesc.value),
      page: String(page.value),
      page_size: String(pageSize.value),
    })
    const res = await fetch(`/api/images?${params}`)
    const data = await res.json()
    images.value = data.items || []
    total.value = data.total || 0
    totalPages.value = data.pages || 0
  } catch {
    images.value = []
    total.value = 0
    totalPages.value = 0
  } finally {
    loadingImages.value = false
  }
}

function onDirectoryChange() {
  page.value = 1
  loadImages()
}

function onSearchChange() {
  page.value = 1
  loadImages()
}

function toggleSort() {
  sortDesc.value = !sortDesc.value
  page.value = 1
  loadImages()
}

function onPageChange(p: number) {
  page.value = p
  loadImages()
}

// ── image selection + caption editor ──────────────────────────

function selectImage(img: ImageItem) {
  selectedImage.value = img
  editCaption.value = img.caption || ''
  diskCaption.value = img.caption || ''
  showMask.value = false
  editorDialog.value = true
}

function onRowClick(_event: unknown, { item }: { item: ImageItem }) {
  selectImage(item)
}

function revertCaption() {
  editCaption.value = diskCaption.value
}

async function saveCaption() {
  if (!selectedImage.value || !isDirty.value) return
  saving.value = true
  try {
    const res = await fetch(
      `/api/images/caption/${encodeURIComponent(selectedImage.value.path)}?directory=${encodeURIComponent(directory.value)}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: editCaption.value }),
      },
    )
    const data = await res.json()
    diskCaption.value = data.content
    // Update the in-memory image list too
    const idx = images.value.findIndex(i => i.path === selectedImage.value!.path)
    if (idx >= 0) images.value[idx].caption = data.content
  } catch {
    // error — caption stays dirty
  } finally {
    saving.value = false
  }
}

function tryCloseEditor() {
  if (isDirty.value) {
    _pendingClose = true
    discardDialog.value = true
  } else {
    editorDialog.value = false
  }
}

async function saveAndClose() {
  await saveCaption()
  discardDialog.value = false
  if (_pendingClose) {
    editorDialog.value = false
    _pendingClose = false
  }
}

function discardAndClose() {
  discardDialog.value = false
  editorDialog.value = false
  _pendingClose = false
}

// ── version history ───────────────────────────────────────────

async function openVersions() {
  if (!selectedImage.value) return
  try {
    const res = await fetch(
      `/api/images/versions/${encodeURIComponent(selectedImage.value.path)}?directory=${encodeURIComponent(directory.value)}`,
    )
    versions.value = await res.json()
    selectedVersion.value = -1
    versionsDialog.value = true
  } catch {
    versions.value = []
    versionsDialog.value = true
  }
}

function restoreVersion(text: string) {
  editCaption.value = text
  versionsDialog.value = false
}

// ── Tag preview ───────────────────────────────────────────────

const parsedTags = computed(() => {
  if (!editCaption.value) return []
  return editCaption.value.split(',').map(s => s.trim()).filter(Boolean)
})

function tagClass(tag: string): string {
  if (/^by\s/i.test(tag)) return 'tag-artist'
  if (/^(on the|in the)\s/i.test(tag)) return 'tag-section'
  return 'tag-plain'
}

// ── Diff computation ──────────────────────────────────────────

const diffLines = computed(() => {
  if (selectedVersion.value < 0 || selectedVersion.value >= versions.value.length) return []
  const oldText = versions.value[selectedVersion.value].text
  const newText = diskCaption.value
  return computeUnifiedDiff(oldText, newText)
})

const diffStats = computed(() => {
  let added = 0
  let removed = 0
  for (const line of diffLines.value) {
    if (line.startsWith('+')) added++
    else if (line.startsWith('-')) removed++
  }
  return { added, removed }
})

function diffLineClass(line: string): string {
  if (line.startsWith('@@')) return 'diff-hunk'
  if (line.startsWith('+')) return 'diff-add'
  if (line.startsWith('-')) return 'diff-del'
  return 'diff-ctx'
}

function computeUnifiedDiff(oldText: string, newText: string): string[] {
  const oldLines = oldText.split('\n')
  const newLines = newText.split('\n')
  const result: string[] = []

  // Simple LCS-based diff
  const m = oldLines.length
  const n = newLines.length
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0))

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = oldLines[i - 1] === newLines[j - 1]
        ? dp[i - 1][j - 1] + 1
        : Math.max(dp[i - 1][j], dp[i][j - 1])
    }
  }

  // Backtrack to build diff
  const ops: { type: string; line: string }[] = []
  let i = m
  let j = n
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && oldLines[i - 1] === newLines[j - 1]) {
      ops.unshift({ type: ' ', line: oldLines[i - 1] })
      i--
      j--
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      ops.unshift({ type: '+', line: newLines[j - 1] })
      j--
    } else {
      ops.unshift({ type: '-', line: oldLines[i - 1] })
      i--
    }
  }

  // Format as unified diff with context
  const contextLines = 3
  let opIdx = 0
  while (opIdx < ops.length) {
    // Find next change
    if (ops[opIdx].type === ' ') {
      // Check if this is near a change (within contextLines)
      let nearChange = false
      for (let k = Math.max(0, opIdx - contextLines); k <= Math.min(ops.length - 1, opIdx + contextLines); k++) {
        if (ops[k].type !== ' ') { nearChange = true; break }
      }
      if (nearChange) {
        result.push(` ${ops[opIdx].line}`)
      } else if (opIdx === 0 || (opIdx > 0 && ops[opIdx - 1].type !== ' ')) {
        result.push(` ${ops[opIdx].line}`)
      }
    } else {
      result.push(`${ops[opIdx].type}${ops[opIdx].line}`)
    }
    opIdx++
  }

  return result
}

// ── lifecycle ─────────────────────────────────────────────────

onMounted(async () => {
  await loadDirectories()
  await loadImages()
})
</script>

<style scoped>
.mask-badge {
  position: absolute;
  top: 4px;
  right: 4px;
  background: rgba(0, 0, 0, 0.6);
  border-radius: 4px;
  padding: 2px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.caption-editor :deep(.v-field) {
  font-family: monospace;
  font-size: 13px;
}

/* Tag preview */
.tag-preview-content {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  font-size: 12px;
  line-height: 1.6;
}

.tag-chip {
  padding: 1px 6px;
  border-radius: 4px;
  border: 1px solid;
  font-size: 11px;
}

.tag-plain {
  border-color: #888;
  color: #ccc;
}

.tag-artist {
  border-color: #d4a017;
  color: #d4a017;
}

.tag-section {
  border-color: #5e8eb0;
  color: #5e8eb0;
}

/* Diff display */
.diff-output {
  font-family: monospace;
  font-size: 12px;
  line-height: 1.5;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
  padding: 8px 12px;
  max-height: 450px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
}

.diff-add {
  background: rgba(72, 199, 142, 0.15);
  color: #9ad17a;
}

.diff-del {
  background: rgba(224, 122, 122, 0.15);
  color: #e07a7a;
}

.diff-hunk {
  color: #7aa6da;
}

.diff-ctx {
  color: #aaa;
}
</style>
