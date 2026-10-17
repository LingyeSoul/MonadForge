<template>
  <v-card variant="tonal" class="curation-inspector">
    <v-card-title class="d-flex align-center flex-wrap ga-2 text-subtitle-1">
      <span class="path-label">{{ item.rel }}</span><v-spacer />
      <v-chip v-if="item.excluded" color="warning" size="small">{{ tr('excluded') }}</v-chip>
      <v-btn size="small" variant="outlined" :disabled="busy || dirty" @click="emit('exclude', !item.excluded)">{{ tr(item.excluded ? 'restore' : 'exclude') }}</v-btn>
    </v-card-title>
    <v-card-text>
      <div class="d-flex flex-wrap ga-2 align-center mb-3">
        <v-btn-toggle v-model="view" mandatory density="compact" variant="outlined">
          <v-btn value="image" size="small" @click="showMask = false; showOcr = false">{{ tr('original') }}</v-btn>
          <v-btn value="resized" size="small" :disabled="!item.resized">{{ tr('resized') }}</v-btn>
        </v-btn-toggle>
        <v-checkbox v-model="showMask" :disabled="!item.mask" :label="tr('mask')" density="compact" hide-details />
        <v-checkbox v-model="showOcr" :disabled="!item.ocr.length" :label="tr('ocr')" density="compact" hide-details />
      </div>
      <div class="image-frame">
        <div v-if="base" class="image-canvas" :style="{ aspectRatio: `${base.width || 1} / ${base.height || 1}` }">
          <img class="base-image" :src="curationFileUrl(base.path)" :alt="item.name" />
          <img v-if="showMask && item.mask && maskAligned" class="mask-image" :src="curationFileUrl(item.mask.path)" alt="" :style="{ opacity: opacity / 100 }" />
          <svg v-if="showOcr && item.ocr.length && ocrAligned" class="ocr-layer" :viewBox="`0 0 ${base.width || 1} ${base.height || 1}`">
            <g v-for="(line, index) in item.ocr" :key="index" @mouseenter="hoveredOcr = index" @mouseleave="hoveredOcr = -1">
              <polygon :points="points(line.box)" :class="{ filtered: !line.usable, hovered: hoveredOcr === index }"><title>{{ line.text }}</title></polygon>
              <text :x="line.box[0] || 0" :y="Math.max(16, (line.box[1] || 0) - 5)" :font-size="Math.max(16, (base.width || 600) / 45)">{{ line.seq }}</text>
            </g>
          </svg>
        </div>
        <v-icon v-else icon="mdi-image-off" size="48" />
      </div>
      <v-slider v-if="showMask" v-model="opacity" :label="tr('opacity')" min="0" max="100" step="5" density="compact" hide-details class="mt-3" />
      <p class="text-caption text-medium-emphasis mt-2">{{ base?.width }} × {{ base?.height }} · {{ base?.path }}</p>
      <v-alert v-if="(showMask && !maskAligned) || (showOcr && !ocrAligned)" type="info" variant="tonal" density="compact" class="mt-2">{{ tr('overlayBasis') }}</v-alert>
      <v-expansion-panels v-if="item.ocr.length" class="mt-3" variant="accordion">
        <v-expansion-panel :title="`${tr('ocr')} · ${item.ocr.length}`"><v-expansion-panel-text>
          <div v-for="(line, index) in item.ocr" :key="index" class="ocr-row pa-2" :class="{ 'bg-surface-light': hoveredOcr === index }" @mouseenter="hoveredOcr = index" @mouseleave="hoveredOcr = -1">
            <v-chip size="x-small" :color="line.usable ? 'success' : 'warning'" class="mr-2">{{ line.seq }} · {{ tr(line.usable ? 'includedText' : 'ignoredText') }}</v-chip>
            <span>{{ line.text }}</span><span class="text-caption text-medium-emphasis ml-2">{{ Math.round(line.score * 100) }}%</span>
          </div>
        </v-expansion-panel-text></v-expansion-panel>
      </v-expansion-panels>
      <div class="text-subtitle-2 mt-5 mb-2">{{ tr('versions') }}</div>
      <div class="d-flex flex-wrap ga-2 mb-3" data-testid="curation-version-ladder">
        <v-btn v-for="entry in item.versions" :key="entry.kind" size="small" :variant="versionKind === entry.kind ? 'flat' : 'outlined'" :color="versionKind === entry.kind ? 'primary' : undefined" :prepend-icon="entry.editable ? 'mdi-pencil-outline' : entry.exists ? 'mdi-file-document-outline' : 'mdi-file-outline'" @click="selectVersion(entry.kind)">{{ entry.kind }}</v-btn>
      </div>
      <p class="text-caption text-medium-emphasis path-label mb-3">{{ version?.note }} {{ version?.path }}</p>
      <v-alert v-if="!version?.editable" type="info" variant="tonal" density="compact" class="mb-3">{{ tr('readOnly') }}</v-alert>
      <v-alert v-if="savedStale" type="warning" variant="tonal" density="compact" class="mb-3">{{ tr('stale') }}</v-alert>
      <v-textarea v-model="text" data-testid="curation-caption-editor" :label="tr('editor')" :readonly="!version?.editable" :disabled="busy" rows="5" auto-grow hide-details />
      <div class="d-flex flex-wrap align-center ga-2 mt-2 mb-4">
        <v-btn v-if="version?.editable" data-testid="curation-caption-save" size="small" color="primary" :disabled="!dirty || busy || parsePending" @click="save">{{ tr('save') }}</v-btn>
        <v-btn v-if="dirty" size="small" variant="text" :disabled="busy" @click="resetText">{{ tr('reset') }}</v-btn>
        <v-chip v-if="dirty" size="small" color="warning">{{ tr('dirty') }}</v-chip>
        <v-progress-circular v-if="parsePending" size="16" width="2" indeterminate />
      </div>
      <v-alert v-if="parseError" type="error" density="compact" variant="tonal">{{ parseError }}</v-alert>
      <div v-if="parsed" class="caption-structure">
        <div class="text-subtitle-2 mb-2">{{ tr('parsed') }}</div>
        <div class="d-flex flex-wrap ga-1">
          <v-chip v-for="(tag, index) in parsed.flat_tags" :key="index" size="small" :closable="canStructureEdit" @click:close="removeTag(null, index)">{{ tag }}</v-chip>
        </div>
        <div v-if="canStructureEdit" class="d-flex ga-2 mt-3">
          <v-text-field v-model="newTag" :label="tr('addTag')" density="compact" hide-details @keydown.enter.prevent="addTag(null)" />
          <v-btn icon="mdi-plus" size="small" variant="text" :aria-label="tr('addTag')" @click="addTag(null)" />
        </div>
        <v-card v-for="(clause, index) in parsed.clauses" :key="index" variant="outlined" class="pa-3 mt-3">
          <div class="d-flex align-center ga-2 mb-2"><strong class="text-body-2">{{ clause.header }}</strong><v-spacer /><v-btn v-if="canStructureEdit" icon="mdi-close" size="x-small" variant="text" :aria-label="tr('remove')" @click="removeClause(index)" /></div>
          <v-chip v-for="(tag, tagIndex) in clause.tags" :key="tagIndex" size="small" class="mr-1 mb-1" :closable="canStructureEdit" @click:close="removeTag(index, tagIndex)">{{ tag }}</v-chip>
          <div v-if="canStructureEdit" class="d-flex ga-2 mt-2"><v-text-field v-model="clauseTag[index]" :label="tr('addTag')" density="compact" hide-details @keydown.enter.prevent="addTag(index)" /><v-btn icon="mdi-plus" size="small" variant="text" :aria-label="tr('addTag')" @click="addTag(index)" /></div>
        </v-card>
        <div v-if="canStructureEdit" class="d-flex flex-wrap align-center ga-2 mt-3">
          <v-select v-model="clausePrefix" :items="['On the ', 'In the ', 'Japanese text reads as ', 'Japanese SFX reads as ']" :label="tr('clauseType')" density="compact" hide-details style="min-width: 220px" />
          <v-text-field v-if="!clausePrefix.startsWith('Japanese')" v-model="clausePosition" :label="tr('positionLabel')" density="compact" hide-details style="min-width: 150px" />
          <v-btn size="small" variant="outlined" :disabled="!clausePrefix.startsWith('Japanese') && !clausePosition.trim()" @click="addClause">{{ tr('addClause') }}</v-btn>
        </div>
      </div>
      <v-select v-model="compareKind" :items="item.versions.filter(entry => entry.exists).map(entry => entry.kind)" :label="tr('compare')" density="compact" hide-details class="mt-5" clearable />
      <div v-if="compareVersion" class="caption-diff mt-3" data-testid="curation-caption-diff"><span>{{ diff.prefix }}</span><del>{{ diff.removed }}</del><ins>{{ diff.added }}</ins><span>{{ diff.suffix }}</span></div>
    </v-card-text>
    <v-dialog v-model="discardOpen" max-width="480" persistent><v-card :title="tr('discardTitle')" :text="tr('discardBody')"><v-card-actions><v-spacer /><v-btn @click="resolveDiscard(false)">{{ tr('cancel') }}</v-btn><v-btn color="warning" @click="resolveDiscard(true)">{{ tr('discard') }}</v-btn></v-card-actions></v-card></v-dialog>
  </v-card>
</template>
<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useAppStore } from '../../stores/app'
import { captionDiff, curationFileUrl } from '../../composables/useCuration'
import { curationText } from '../../i18n/curation'
import type { CaptionClause, CurationItem, ParsedCaption } from '../../types/curation'
const props = defineProps<{ item: CurationItem; busy: boolean; parse: (text: string) => Promise<ParsedCaption>; compose: (tags: string[], clauses: CaptionClause[]) => Promise<{ text: string; parsed: ParsedCaption }> }>()
const emit = defineEmits<{ save: [kind: string, text: string]; exclude: [excluded: boolean]; dirty: [dirty: boolean] }>()
const app = useAppStore()
const tr = (key: Parameters<typeof curationText>[1]) => curationText(app.language, key)
const view = ref<'image' | 'resized'>('image')
const showMask = ref(false), showOcr = ref(false), opacity = ref(45), hoveredOcr = ref(-1)
const base = computed(() => showOcr.value && props.item.ocr_base ? props.item.ocr_base : showMask.value && props.item.mask_base ? props.item.mask_base : props.item[view.value] || props.item.image)
const maskAligned = computed(() => base.value?.path === props.item.mask_base?.path)
const ocrAligned = computed(() => base.value?.path === props.item.ocr_base?.path)
watch([showMask, showOcr], ([mask, ocr]) => { if ((mask || ocr) && props.item.resized) view.value = 'resized' })
function points(box: number[]): string {
  if (box.length === 4) { const [x1, y1, x2, y2] = box; return `${x1},${y1} ${x2},${y1} ${x2},${y2} ${x1},${y2}` }
  return Array.from({ length: Math.floor(box.length / 2) }, (_, index) => `${box[index * 2]},${box[index * 2 + 1]}`).join(' ')
}
const versionKind = ref('master'), compareKind = ref<string | null>(null), text = ref('')
const version = computed(() => props.item.versions.find(entry => entry.kind === versionKind.value))
const compareVersion = computed(() => props.item.versions.find(entry => entry.kind === compareKind.value))
const diff = computed(() => captionDiff(compareVersion.value?.text || '', text.value))
const dirty = computed(() => text.value !== (version.value?.text || ''))
watch(dirty, value => emit('dirty', value))
const savedStale = ref(false), parsed = ref<ParsedCaption | null>(null), parseError = ref(''), parsePending = ref(false)
let parseTimer: ReturnType<typeof setTimeout> | undefined
let parseIndex = 0
function resetText(): void { text.value = version.value?.text || ''; parsed.value = version.value?.parsed || null; savedStale.value = false }
watch(() => props.item, () => {
  if (!props.item.versions.some(entry => entry.kind === versionKind.value)) versionKind.value = props.item.versions.find(entry => entry.editable)?.kind || props.item.versions[0]?.kind || 'master'
  resetText()
}, { immediate: true })
watch(text, value => {
  clearTimeout(parseTimer)
  const index = ++parseIndex
  parsePending.value = true
  parseTimer = setTimeout(async () => {
    try { const data = await props.parse(value); if (index === parseIndex) { parsed.value = data; parseError.value = '' } }
    catch (cause) { if (index === parseIndex) parseError.value = String(cause) }
    finally { if (index === parseIndex) parsePending.value = false }
  }, 220)
})
onBeforeUnmount(() => { clearTimeout(parseTimer); parseIndex++; discardResolve?.(false) })
const discardOpen = ref(false)
let discardResolve: ((value: boolean) => void) | undefined
async function requestLeave(): Promise<boolean> { if (!dirty.value) return true; discardOpen.value = true; return new Promise(resolve => { discardResolve = resolve }) }
function resolveDiscard(value: boolean): void { discardOpen.value = false; discardResolve?.(value); discardResolve = undefined }
async function selectVersion(kind: string): Promise<void> { if (kind === versionKind.value || !await requestLeave()) return; versionKind.value = kind; resetText() }
function save(): void { if (version.value?.editable) { emit('save', versionKind.value, text.value); savedStale.value = true } }
defineExpose({ requestLeave })
const newTag = ref(''), clauseTag = ref<Record<number, string>>({}), clausePrefix = ref('On the '), clausePosition = ref('')
const canStructureEdit = computed(() => Boolean(version.value?.editable && !props.busy && !parsePending.value && !parseError.value))
async function composeChange(update: (next: ParsedCaption) => void): Promise<void> {
  if (!parsed.value || !canStructureEdit.value) return
  const identity = `${props.item.rel}\0${versionKind.value}`
  const baseline = text.value
  const index = ++parseIndex
  clearTimeout(parseTimer)
  const next: ParsedCaption = JSON.parse(JSON.stringify(parsed.value))
  update(next)
  parsePending.value = true
  try {
    const data = await props.compose(next.flat_tags, next.clauses)
    if (index !== parseIndex || identity !== `${props.item.rel}\0${versionKind.value}` || text.value !== baseline) return
    text.value = data.text; parsed.value = data.parsed; parseError.value = ''
  }
  catch (cause) { if (index === parseIndex) parseError.value = String(cause) }
  finally { if (index === parseIndex) parsePending.value = false }
}
function removeTag(clause: number | null, index: number): void { void composeChange(next => { (clause == null ? next.flat_tags : next.clauses[clause]!.tags).splice(index, 1) }) }
function addTag(clause: number | null): void { const value = (clause == null ? newTag.value : clauseTag.value[clause] || '').trim(); if (!value) return; void composeChange(next => { (clause == null ? next.flat_tags : next.clauses[clause]!.tags).push(value) }); if (clause == null) newTag.value = ''; else clauseTag.value[clause] = '' }
function removeClause(index: number): void { void composeChange(next => { next.clauses.splice(index, 1) }) }
function addClause(): void { void composeChange(next => { const position = clausePrefix.value.startsWith('Japanese') ? '' : clausePosition.value.trim(); next.clauses.push({ prefix: clausePrefix.value, position, header: `${clausePrefix.value}${position}`.trim(), tags: [] }) }) }
</script>
<style scoped>
.path-label { overflow-wrap: anywhere; min-width: 0; }
.image-frame { min-height: 180px; max-height: 540px; overflow: auto; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,.22); border-radius: 8px; }
.image-canvas { position: relative; width: 100%; max-height: 520px; }
.base-image { display: block; width: 100%; height: 100%; object-fit: contain; }
.mask-image, .ocr-layer { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
.mask-image { mix-blend-mode: screen; pointer-events: none; }
.ocr-layer polygon { fill: #40ddbb22; stroke: #40ddbb; stroke-width: 2px; vector-effect: non-scaling-stroke; }
.ocr-layer polygon.filtered { stroke: #ffb454; fill: #ffb45411; }
.ocr-layer polygon.hovered { fill: #40ddbb66; stroke-width: 4px; }
.ocr-layer text { fill: #fff; paint-order: stroke; stroke: #000; stroke-width: 2px; pointer-events: none; }
.caption-diff { white-space: pre-wrap; overflow-wrap: anywhere; font-family: monospace; border: 1px solid rgba(var(--v-theme-on-surface), .15); border-radius: 6px; padding: 12px; }
.caption-diff del { background: #d84c4c33; color: #ff9696; }
.caption-diff ins { background: #40aa7033; color: #81dca5; text-decoration: none; }
</style>
