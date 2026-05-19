<template>
  <v-container fluid class="pa-4">
    <v-row align="center" class="mb-4">
      <v-col>
        <div class="text-h5">{{ t('dsTitle') }}</div>
        <div class="text-body-2 text-medium-emphasis">{{ t('dsSubtitle') }}</div>
      </v-col>
      <v-col cols="auto">
        <v-text-field
          v-model="search"
          prepend-inner-icon="mdi-magnify"
          :label="t('dsSearch')"
          variant="outlined"
          density="compact"
          hide-details
          clearable
          style="width: 300px"
        />
      </v-col>
      <v-col cols="auto">
        <v-btn-toggle v-model="viewMode" mandatory density="compact" variant="outlined">
          <v-btn value="grid" icon="mdi-view-grid" />
          <v-btn value="list" icon="mdi-view-list" />
        </v-btn-toggle>
      </v-col>
    </v-row>

    <v-alert type="info" variant="tonal" class="mb-4">
      <span v-html="t('dsEndpointHint')" />
    </v-alert>

    <v-row v-if="viewMode === 'grid'">
      <v-col
        v-for="img in filteredImages"
        :key="img.filename"
        cols="6"
        sm="4"
        md="3"
        lg="2"
      >
        <v-card class="mx-auto" variant="tonal" @click="selectImage(img)">
          <v-img
            :src="`/api/images/${img.path}`"
            :alt="img.filename"
            aspect-ratio="1"
            cover
            class="bg-grey-darken-3"
          >
            <template #placeholder>
              <div class="d-flex align-center justify-center fill-height">
                <v-progress-circular indeterminate color="primary" />
              </div>
            </template>
          </v-img>
          <v-card-subtitle class="text-caption text-truncate pa-2">
            {{ img.filename }}
          </v-card-subtitle>
        </v-card>
      </v-col>
    </v-row>

    <v-data-table
      v-else
      :items="filteredImages"
      :headers="listHeaders"
      item-key="filename"
      hover
      @click:row="onRowClick"
    >
      <template #item.preview="{ item }">
        <v-img
          :src="`/api/images/${item.path}`"
          width="48"
          height="48"
          cover
          class="rounded my-1"
        />
      </template>
      <template #item.caption="{ item }">
        <span class="text-truncate d-inline-block" style="max-width: 400px">
          {{ item.caption || '—' }}
        </span>
      </template>
    </v-data-table>

    <div v-if="images.length === 0" class="text-center pa-12">
      <v-icon icon="mdi-image-off-outline" size="64" color="grey" class="mb-4" />
      <div class="text-h6 text-medium-emphasis">{{ t('dsNoImages') }}</div>
      <div class="text-body-2 text-medium-emphasis" v-html="t('dsNoImagesHint')" />
    </div>

    <v-dialog v-model="editorDialog" max-width="700">
      <v-card v-if="selectedImage">
        <v-card-title class="d-flex align-center">
          {{ selectedImage.filename }}
          <v-spacer />
          <v-chip v-if="selectedImage.has_mask" size="small" color="warning" variant="tonal">
            {{ t('dsHasMask') }}
          </v-chip>
        </v-card-title>
        <v-card-text>
          <v-row>
            <v-col cols="12" md="6">
              <v-img
                :src="`/api/images/${selectedImage.path}`"
                class="rounded"
                contain
              />
            </v-col>
            <v-col cols="12" md="6">
              <v-textarea
                v-model="editCaption"
                :label="t('dsCaption')"
                variant="outlined"
                rows="8"
                auto-grow
                :loading="savingCaption"
              />
              <div class="text-caption text-medium-emphasis">
                {{ selectedImage.width }}x{{ selectedImage.height }}
              </div>
            </v-col>
          </v-row>
        </v-card-text>
        <v-card-actions>
          <v-btn variant="text" @click="editorDialog = false">
            {{ t('dsClose') }}
          </v-btn>
          <v-spacer />
          <v-btn
            color="primary"
            :disabled="editCaption === selectedImage.caption"
            :loading="savingCaption"
            @click="saveCaption"
          >
            {{ t('dsSaveCaption') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useI18n } from '../composables/useI18n'

interface ImageInfo {
  path: string
  filename: string
  width: number
  height: number
  caption: string
  has_mask: boolean
}

const { t } = useI18n()
const images = ref<ImageInfo[]>([])
const search = ref('')
const viewMode = ref('grid')
const editorDialog = ref(false)
const selectedImage = ref<ImageInfo | null>(null)
const editCaption = ref('')
const savingCaption = ref(false)

const listHeaders = computed(() => [
  { title: '', key: 'preview', width: '60px', sortable: false },
  { title: t('dsFilename'), key: 'filename' },
  { title: t('dsCaption'), key: 'caption', sortable: false },
  { title: t('dsSize'), key: 'size', width: '100px' },
  { title: t('dsMask'), key: 'has_mask', width: '60px' },
])

const filteredImages = computed(() => {
  if (!search.value) return images.value
  const q = search.value.toLowerCase()
  return images.value.filter(img =>
    img.filename.toLowerCase().includes(q) ||
    (img.caption || '').toLowerCase().includes(q)
  )
})

function selectImage(img: ImageInfo) {
  selectedImage.value = img
  editCaption.value = img.caption || ''
  editorDialog.value = true
}

function onRowClick(_event: unknown, { item }: { item: ImageInfo }) {
  selectImage(item)
}

async function saveCaption() {
  if (!selectedImage.value) return
  savingCaption.value = true
  try {
    await fetch(`/api/images/${selectedImage.value.path}/caption`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ caption: editCaption.value }),
    })
    selectedImage.value.caption = editCaption.value
  } catch {
    // handle error
  } finally {
    savingCaption.value = false
  }
}

async function loadImages() {
  try {
    const res = await fetch('/api/images')
    const data = await res.json()
    images.value = data.images || []
  } catch {
    // API not available yet
  }
}

onMounted(loadImages)
</script>
