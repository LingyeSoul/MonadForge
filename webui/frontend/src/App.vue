<template>
  <v-app class="forge-app">
    <v-navigation-drawer
      v-model="drawer"
      :rail="!mobile && rail"
      :permanent="!mobile"
      :temporary="mobile"
      :width="240"
      :rail-width="72"
      class="workspace-nav"
    >
      <router-link to="/config" class="workspace-brand" :aria-label="t('appName')">
        <img src="/logo.svg" alt="" width="32" height="32" />
        <div v-if="mobile || !rail" class="workspace-brand__text">
          <strong>MonadForge</strong>
          <span>{{ t('workspaceStudio') }}</span>
        </div>
      </router-link>
      <nav :aria-label="t('workspaceNavigation')">
        <div v-for="group in navGroups" :key="group.label" class="nav-group">
          <div v-if="mobile || !rail" class="nav-group__label">{{ t(group.label) }}</div>
          <v-list density="compact" nav>
            <v-list-item
              v-for="item in group.items"
              :key="item.to"
              :prepend-icon="item.icon"
              :title="t(item.titleKey)"
              :to="item.to"
              :aria-label="t(item.titleKey)"
              :active="route.path === item.to"
              @click="closeMobileDrawer"
            >
              <v-tooltip v-if="!mobile && rail" activator="parent" location="end">{{ t(item.titleKey) }}</v-tooltip>
            </v-list-item>
          </v-list>
        </div>
      </nav>

      <template #append>
        <v-list density="compact" nav class="workspace-nav__utilities">
          <v-list-item
            prepend-icon="mdi-book-open-page-variant-outline"
            :title="t('guidebook')"
            :aria-label="t('guidebook')"
            @click="showGuidebook = true"
          >
            <v-tooltip v-if="!mobile && rail" activator="parent" location="end">{{ t('guidebook') }}</v-tooltip>
          </v-list-item>
          <v-list-item
            prepend-icon="mdi-bug-outline"
            :title="t('reportIssue')"
            :aria-label="t('reportIssue')"
            href="https://github.com/LingyeSoul/MonadForge/issues"
            target="_blank"
            rel="noopener noreferrer"
          >
            <v-tooltip v-if="!mobile && rail" activator="parent" location="end">{{ t('reportIssue') }}</v-tooltip>
          </v-list-item>
          <v-list-item
            prepend-icon="mdi-cog-outline"
            :title="t('navSystem')"
            :aria-label="t('navSystem')"
            to="/system"
            @click="closeMobileDrawer"
          >
            <v-tooltip v-if="!mobile && rail" activator="parent" location="end">{{ t('navSystem') }}</v-tooltip>
          </v-list-item>
        </v-list>
        <div class="workspace-nav__footer" :class="{ 'is-rail': !mobile && rail }">
          <span v-if="mobile || !rail" class="workspace-local">
            <v-icon icon="mdi-laptop" size="16" />{{ t('workspaceLocal') }}
          </span>
          <v-btn
            :icon="appStore.theme === 'dark' ? 'mdi-white-balance-sunny' : 'mdi-weather-night'"
            variant="text"
            size="small"
            :aria-label="t(appStore.theme === 'dark' ? 'themeLight' : 'themeDark')"
            @click="toggleTheme"
          >
            <v-icon :icon="appStore.theme === 'dark' ? 'mdi-white-balance-sunny' : 'mdi-weather-night'" />
            <v-tooltip activator="parent" location="top">{{ t(appStore.theme === 'dark' ? 'themeLight' : 'themeDark') }}</v-tooltip>
          </v-btn>
          <v-menu location="top end">
            <template #activator="{ props }">
              <v-btn v-bind="props" icon="mdi-translate" variant="text" size="small" :aria-label="t('workspaceLanguage')">
                <v-icon icon="mdi-translate" />
                <v-tooltip activator="parent" location="top">{{ t('workspaceLanguage') }}</v-tooltip>
              </v-btn>
            </template>
            <v-list density="compact" :aria-label="t('workspaceLanguage')">
              <v-list-item v-for="language in languageOptions" :key="language.value" :title="language.label"
                :active="appStore.language === language.value"
                :append-icon="appStore.language === language.value ? 'mdi-check' : undefined"
                @click="onLangChange(language.value)" />
            </v-list>
          </v-menu>
        </div>
      </template>
    </v-navigation-drawer>

    <v-app-bar flat height="60" class="workspace-topbar">
      <v-btn :icon="mobile ? 'mdi-menu' : 'mdi-dock-left'" variant="text" size="small" class="ml-3 mr-3"
        :aria-label="t(mobile || rail ? 'workspaceExpandNav' : 'workspaceCollapseNav')"
        :aria-expanded="mobile ? drawer : !rail" @click="toggleDrawer">
        <v-icon :icon="mobile ? 'mdi-menu' : 'mdi-dock-left'" />
        <v-tooltip activator="parent" location="bottom">{{ t(mobile || rail ? 'workspaceExpandNav' : 'workspaceCollapseNav') }}</v-tooltip>
      </v-btn>
      <div class="workspace-breadcrumb">
        <span class="workspace-breadcrumb__group">{{ t(currentGroup) }}</span>
        <v-icon class="workspace-breadcrumb__group" icon="mdi-chevron-right" size="16" />
        <span>{{ t(currentPage) }}</span>
      </div>
      <v-spacer />
      <span class="workspace-engine">Anima <span>DiT</span></span>
      <v-btn icon="mdi-book-open-page-variant-outline" variant="text" size="small" class="mx-3" :aria-label="t('guidebook')" @click="showGuidebook = true">
        <v-icon icon="mdi-book-open-page-variant-outline" />
        <v-tooltip activator="parent" location="bottom">{{ t('guidebook') }}</v-tooltip>
      </v-btn>
    </v-app-bar>

    <v-main id="workspace-main">
      <router-view v-slot="{ Component }">
        <Transition :css="false" @enter="enterPage" @enter-cancelled="cancelMotion">
          <component :is="Component" />
        </Transition>
      </router-view>
    </v-main>

    <GuidebookDialog v-model="showGuidebook" />

    <v-snackbar
      v-model="snackbarOpen"
      :color="notifyStore.current?.type"
      timeout="-1"
      location="top end"
      @update:model-value="onSnackbarUpdate"
    >
      {{ notifyStore.current?.message }}
    </v-snackbar>
  </v-app>
</template>

<script setup lang="ts">
import { computed, ref, watch, onBeforeUnmount } from 'vue'
import { useDisplay, useTheme } from 'vuetify'
import { useRoute } from 'vue-router'
import { gsap } from 'gsap'
import { useAppStore } from './stores/app'
import { useNotifyStore } from './stores/notify'
import { useI18n } from './composables/useI18n'
import GuidebookDialog from './components/GuidebookDialog.vue'

const appStore = useAppStore()
const notifyStore = useNotifyStore()
const { t, setLanguage } = useI18n()
const vuetifyTheme = useTheme()
const route = useRoute()
const { smAndDown: mobile } = useDisplay()

// Store state is the single source of truth; mirror it onto Vuetify's
// global theme (theme keys 'dark'/'light' match the store values 1:1).
watch(() => appStore.theme, (name) => {
  vuetifyTheme.change(name)
}, { immediate: true })

function toggleTheme() {
  appStore.setTheme(appStore.theme === 'dark' ? 'light' : 'dark')
}

const snackbarOpen = ref(false)
const snackbarTimer = ref<ReturnType<typeof setTimeout> | null>(null)
const showGuidebook = ref(false)
let dismissing = false

function scheduleClose(timeout: number) {
  if (snackbarTimer.value) clearTimeout(snackbarTimer.value)
  snackbarTimer.value = setTimeout(() => {
    snackbarTimer.value = null
    closeAndAdvance()
  }, timeout)
}

function closeAndAdvance() {
  if (dismissing) return
  dismissing = true
  if (snackbarTimer.value) {
    clearTimeout(snackbarTimer.value)
    snackbarTimer.value = null
  }
  snackbarOpen.value = false
  // Let Vuetify finish its close animation before popping the queue
  setTimeout(() => {
    notifyStore.dismiss()
    dismissing = false
  }, 300)
}

watch(() => notifyStore.current, (item) => {
  if (item) {
    snackbarOpen.value = true
    scheduleClose(item.timeout)
  }
})

function onSnackbarUpdate(open: boolean) {
  if (!open) {
    closeAndAdvance()
  }
}

onBeforeUnmount(() => {
  if (snackbarTimer.value) clearTimeout(snackbarTimer.value)
  pageMotion?.kill()
})

const drawer = ref(!mobile.value)
const rail = ref(localStorage.getItem('monadforge-nav-rail') === 'true')
watch(rail, value => localStorage.setItem('monadforge-nav-rail', String(value)))
watch(mobile, value => { drawer.value = !value })

const navGroups = [
  { label: 'workspaceTraining', items: [
  { icon: 'mdi-cog-transfer-outline', titleKey: 'navConfig', to: '/config' },
  { icon: 'mdi-chart-line', titleKey: 'navDashboard', to: '/dashboard' },
  { icon: 'mdi-console-line', titleKey: 'navTasks', to: '/tasks' },
  ] },
  { label: 'workspaceData', items: [
  { icon: 'mdi-image-multiple-outline', titleKey: 'navDataset', to: '/dataset' },
  { icon: 'mdi-cogs', titleKey: 'navPreprocess', to: '/preprocess' },
  { icon: 'mdi-layers-triple-outline', titleKey: 'navStagedResolution', to: '/staged-resolution' },
  ] },
  { label: 'workspaceModels', items: [
  { icon: 'mdi-cube-outline', titleKey: 'navModels', to: '/models' },
  { icon: 'mdi-puzzle-outline', titleKey: 'navAdapter', to: '/adapter' },
  { icon: 'mdi-flask-outline', titleKey: 'navDistill', to: '/distill' },
  { icon: 'mdi-image-filter-center-focus-strong', titleKey: 'navSuperResolution', to: '/sr' },
  { icon: 'mdi-call-merge', titleKey: 'navMerge', to: '/merge' },
  ] },
]
const currentGroup = computed(() => navGroups.find(group => group.items.some(item => item.to === route.path))?.label ?? 'navSystem')
const currentPage = computed(() => navGroups.flatMap(group => group.items).find(item => item.to === route.path)?.titleKey ?? 'navSystem')
const languageOptions = [
  { value: 'en', label: 'English' },
  { value: 'cn', label: '简体中文' },
  { value: 'ko', label: '한국어' },
  { value: 'ja', label: '日本語' },
]

function toggleDrawer() {
  if (mobile.value) drawer.value = !drawer.value
  else rail.value = !rail.value
}

function closeMobileDrawer() {
  if (mobile.value) drawer.value = false
}

let pageMotion: gsap.core.Tween | undefined
function enterPage(el: Element, done: () => void) {
  pageMotion = gsap.fromTo(el, { opacity: 0, y: 6 }, {
    opacity: 1, y: 0,
    duration: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 0.24,
    ease: 'power2.out', clearProps: 'opacity,transform', onComplete: done,
  })
}

function cancelMotion(el: Element) {
  pageMotion?.kill()
  gsap.set(el, { clearProps: 'opacity,transform' })
}

async function onLangChange(lang: unknown) {
  if (typeof lang === 'string') {
    await setLanguage(lang)
  }
}
</script>

<style>
/* Make v-main a proper flex-height container so fill-height children work */
.v-main {
  display: flex !important;
  flex-direction: column;
  height: 100dvh;
  overflow: hidden;
}
.v-main > .v-main__wrap {
  display: flex;
  flex-direction: column;
  flex: 1 1 0;
  min-height: 0;
  overflow: hidden;
}

</style>
