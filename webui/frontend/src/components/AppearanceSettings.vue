<template>
  <v-card variant="tonal">
    <v-card-title class="text-subtitle-1">
      <v-icon icon="mdi-palette-outline" class="mr-2" />
      {{ t('appearance') }}
    </v-card-title>
    <v-card-text>
      <div class="appearance-option">
        <div class="appearance-option__text">
          <span class="appearance-option__label">{{ t('appearanceGlass') }}</span>
          <p class="appearance-option__hint">{{ t('appearanceGlassHint') }}</p>
        </div>
        <v-switch
          :model-value="appStore.glass"
          hide-details
          :aria-label="t('appearanceGlass')"
          @update:model-value="onToggleGlass"
        />
      </div>

      <div class="appearance-slider">
        <div class="appearance-slider__head">
          <span class="appearance-option__label">{{ t('appearanceSurfaceOpacity') }}</span>
          <span class="appearance-slider__value">{{ appStore.surfaceOpacity }}%</span>
        </div>
        <v-slider
          :model-value="appStore.surfaceOpacity"
          :min="20"
          :max="100"
          :step="5"
          :disabled="!appStore.glass"
          hide-details
          density="compact"
          :aria-label="t('appearanceSurfaceOpacity')"
          @update:model-value="onSurfaceOpacity"
        />
      </div>

      <div class="appearance-option">
        <div class="appearance-option__text">
          <span class="appearance-option__label">{{ t('appearanceWallpaper') }}</span>
          <p class="appearance-option__hint">{{ t('appearanceWallpaperHint') }}</p>
        </div>
        <v-switch
          :model-value="appStore.wallpaper.enabled"
          hide-details
          :aria-label="t('appearanceWallpaper')"
          @update:model-value="onToggleWallpaper"
        />
      </div>

      <template v-if="appStore.wallpaper.enabled">
        <div class="appearance-slider">
          <div class="appearance-slider__head">
            <span class="appearance-option__label">{{ t('appearanceWallpaperBlur') }}</span>
            <span class="appearance-slider__value">{{ appStore.wallpaper.blur }} px</span>
          </div>
          <v-slider
            :model-value="appStore.wallpaper.blur"
            :min="0"
            :max="30"
            :step="1"
            hide-details
            density="compact"
            :aria-label="t('appearanceWallpaperBlur')"
            @update:model-value="onWallpaperBlur"
          />
        </div>

        <div class="appearance-slider">
          <div class="appearance-slider__head">
            <span class="appearance-option__label">{{ t('appearanceVeil') }}</span>
            <span class="appearance-slider__value">{{ appStore.veilOpacity }}%</span>
          </div>
          <v-slider
            :model-value="appStore.veilOpacity"
            :min="0"
            :max="90"
            :step="5"
            hide-details
            density="compact"
            :aria-label="t('appearanceVeil')"
            @update:model-value="onVeilOpacity"
          />
        </div>

        <v-text-field
          v-model="urlDraft"
          :label="t('appearanceWallpaperUrl')"
          hide-details
          class="mt-2"
          @change="applyUrl"
        />
        <div class="d-flex align-center gap-2 mt-3">
          <v-btn variant="tonal" size="small" prepend-icon="mdi-upload" @click="fileInput?.click()">
            {{ t('appearanceWallpaperUpload') }}
          </v-btn>
          <v-btn
            v-if="appStore.wallpaper.src"
            variant="text"
            size="small"
            prepend-icon="mdi-delete-outline"
            @click="clearWallpaper"
          >
            {{ t('appearanceWallpaperClear') }}
          </v-btn>
        </div>
        <input
          ref="fileInput"
          type="file"
          accept="image/*"
          class="appearance-file-input"
          @change="onFileChange"
        />
      </template>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useAppStore } from '../stores/app'
import { useNotifyStore } from '../stores/notify'
import { useI18n } from '../composables/useI18n'
import { fileToWallpaperDataUrl, verifyImageSrc } from '../utils/wallpaper'

const appStore = useAppStore()
const notifyStore = useNotifyStore()
const { t } = useI18n()

const fileInput = ref<HTMLInputElement | null>(null)
const urlDraft = ref('')

function onToggleGlass(value: unknown) {
  appStore.setGlass(value === true)
}

function onToggleWallpaper(value: unknown) {
  appStore.setWallpaper({ ...appStore.wallpaper, enabled: value === true })
}

function onSurfaceOpacity(value: unknown) {
  appStore.setSurfaceOpacity(Number(value))
}

function onWallpaperBlur(value: unknown) {
  appStore.setWallpaper({ ...appStore.wallpaper, blur: Number(value) })
}

function onVeilOpacity(value: unknown) {
  appStore.setVeilOpacity(Number(value))
}

async function applyUrl() {
  const src = urlDraft.value.trim()
  if (!src) return
  try {
    await verifyImageSrc(src)
    appStore.setWallpaper({ enabled: true, src, blur: appStore.wallpaper.blur })
  } catch {
    notifyStore.show(t('appearanceWallpaperLoadError'), 'error')
  }
}

async function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  // Reset so selecting the same file again still fires a change event
  input.value = ''
  if (!file) return
  try {
    const dataUrl = await fileToWallpaperDataUrl(file)
    urlDraft.value = ''
    appStore.setWallpaper({ enabled: true, src: dataUrl, blur: appStore.wallpaper.blur })
  } catch {
    notifyStore.show(t('appearanceWallpaperLoadError'), 'error')
  }
}

function clearWallpaper() {
  urlDraft.value = ''
  appStore.setWallpaper({ enabled: false, src: '', blur: appStore.wallpaper.blur })
}
</script>

<style scoped>
.appearance-option {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.appearance-option + .appearance-option {
  margin-top: 12px;
}
.appearance-option__text {
  min-width: 0;
}
.appearance-option__label {
  font-size: 14px;
  font-weight: 500;
}
.appearance-option__hint {
  margin: 2px 0 0;
  font-size: 12px;
  color: var(--text-muted);
}
.appearance-slider {
  margin-top: 4px;
}
.appearance-slider + .appearance-slider {
  margin-top: 4px;
}
.appearance-option + .appearance-slider {
  margin-top: 2px;
}
.appearance-slider + .appearance-option {
  margin-top: 12px;
}
.appearance-slider__head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}
.appearance-slider__value {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-secondary);
}
.appearance-file-input {
  display: none;
}
</style>
