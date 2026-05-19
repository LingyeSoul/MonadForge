<template>
  <v-app>
    <v-navigation-drawer
      v-model="drawer"
      :rail="rail"
      permanent
      @click="rail = false"
    >
      <v-list-item
        prepend-icon="mdi-anvil"
        :title="t('appName')"
        class="monadforge-logo py-3"
        nav
      >
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
            prepend-icon="mdi-cog-outline"
            :title="t('navSystem')"
            to="/system"
            rounded="xl"
          />
          <v-list-item
            :prepend-icon="taskStore.tasks.length > 0 ? 'mdi-console' : 'mdi-console-outline'"
            :title="taskStore.tasks.length > 0 ? t('navTasksCount', { n: taskStore.tasks.length }) : t('navTasks')"
            rounded="xl"
            @click="appStore.toggleTaskDrawer()"
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

    <v-navigation-drawer
      v-model="appStore.taskDrawer"
      location="right"
      width="420"
      temporary
    >
      <TaskPanel />
    </v-navigation-drawer>

    <v-main>
      <router-view />
    </v-main>
  </v-app>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useAppStore } from './stores/app'
import { useTaskStore } from './stores/task'
import { useI18n } from './composables/useI18n'
import TaskPanel from './components/TaskPanel.vue'

const appStore = useAppStore()
const taskStore = useTaskStore()
const { t, setLanguage } = useI18n()

const drawer = ref(true)
const rail = ref(true)

const navItems = [
  { icon: 'mdi-cog-transfer-outline', titleKey: 'navConfig', to: '/config' },
  { icon: 'mdi-image-multiple-outline', titleKey: 'navDataset', to: '/dataset' },
  { icon: 'mdi-cogs', titleKey: 'navPreprocess', to: '/preprocess' },
  { icon: 'mdi-puzzle-outline', titleKey: 'navAdapter', to: '/adapter' },
  { icon: 'mdi-call-merge', titleKey: 'navMerge', to: '/merge' },
]

async function onLangChange(lang: unknown) {
  if (typeof lang === 'string') {
    await setLanguage(lang)
  }
}

taskStore.fetchTasks()
</script>
