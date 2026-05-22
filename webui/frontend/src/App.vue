<template>
  <v-app>
    <v-navigation-drawer
      v-model="drawer"
      :rail="rail"
      permanent
      @click="rail = false"
    >
      <v-list-item
        :title="t('appName')"
        class="monadforge-logo py-3"
        nav
      >
        <template #prepend>
          <v-avatar size="32" class="mr-2">
            <img src="/logo.svg" alt="MonadForge" />
          </v-avatar>
        </template>
        <template #append>
          <v-btn
            icon="mdi-chevron-left"
            variant="text"
            size="small"
            @click.stop="rail = !rail"
          />
        </template>
      </v-list-item>

      <v-divider />

      <v-list density="compact" nav>
        <v-list-item
          v-for="item in navItems"
          :key="item.to"
          :prepend-icon="item.icon"
          :title="t(item.titleKey)"
          :to="item.to"
          rounded="xl"
        />
      </v-list>

      <template #append>
        <v-divider />
        <v-list density="compact" nav>
          <v-list-item
            prepend-icon="mdi-book-open-page-variant-outline"
            :title="t('guidebook')"
            rounded="xl"
            @click="showGuidebook = true"
          />
          <v-list-item
            prepend-icon="mdi-bug-outline"
            :title="t('reportIssue')"
            rounded="xl"
            href="https://github.com/LingyeSoul/MonadForge/issues"
            target="_blank"
          />
          <v-list-item
            prepend-icon="mdi-cog-outline"
            :title="t('navSystem')"
            to="/system"
            rounded="xl"
          />
        </v-list>
        <div class="pa-2">
          <v-btn-toggle
            :model-value="appStore.language"
            density="compact"
            variant="outlined"
            divided
            mandatory
            class="w-100"
            @update:model-value="onLangChange"
          >
            <v-btn value="en" size="small">EN</v-btn>
            <v-btn value="cn" size="small">中</v-btn>
          </v-btn-toggle>
        </div>
      </template>
    </v-navigation-drawer>

    <v-main>
      <router-view />
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
import { ref, watch, onBeforeUnmount } from 'vue'
import { useAppStore } from './stores/app'
import { useNotifyStore } from './stores/notify'
import { useI18n } from './composables/useI18n'
import GuidebookDialog from './components/GuidebookDialog.vue'

const appStore = useAppStore()
const notifyStore = useNotifyStore()
const { t, setLanguage } = useI18n()

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
})

const drawer = ref(true)
const rail = ref(true)

const navItems = [
  { icon: 'mdi-cog-transfer-outline', titleKey: 'navConfig', to: '/config' },
  { icon: 'mdi-image-multiple-outline', titleKey: 'navDataset', to: '/dataset' },
  { icon: 'mdi-cogs', titleKey: 'navPreprocess', to: '/preprocess' },
  { icon: 'mdi-puzzle-outline', titleKey: 'navAdapter', to: '/adapter' },
  { icon: 'mdi-call-merge', titleKey: 'navMerge', to: '/merge' },
  { icon: 'mdi-console-line', titleKey: 'navTasks', to: '/tasks' },
  { icon: 'mdi-chart-line', titleKey: 'navDashboard', to: '/dashboard' },
]

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
  height: 100vh;
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
