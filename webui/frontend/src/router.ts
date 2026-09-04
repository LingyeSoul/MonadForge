import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/config' },
    {
      path: '/config',
      name: 'config',
      component: () => import('./views/ConfigEditor.vue'),
    },
    {
      path: '/dataset',
      name: 'dataset',
      component: () => import('./views/DatasetBrowser.vue'),
    },
    {
      path: '/preprocess',
      name: 'preprocess',
      component: () => import('./views/PreprocessView.vue'),
    },
    {
      path: '/staged-resolution',
      name: 'staged-resolution',
      component: () => import('./views/StagedResolutionView.vue'),
    },
    {
      path: '/adapter',
      name: 'adapter',
      component: () => import('./views/AdapterView.vue'),
    },
    {
      path: '/distill',
      name: 'distill',
      component: () => import('./views/DistillView.vue'),
    },
    {
      path: '/sr',
      name: 'sr',
      component: () => import('./views/SuperResolutionView.vue'),
    },
    {
      path: '/merge',
      name: 'merge',
      component: () => import('./views/MergeView.vue'),
    },
    {
      path: '/models',
      name: 'models',
      component: () => import('./views/ModelsView.vue'),
    },
    {
      path: '/tasks',
      name: 'tasks',
      component: () => import('./views/TaskMonitorView.vue'),
    },
    {
      path: '/dashboard',
      name: 'dashboard',
      component: () => import('./views/TrainingDashboard.vue'),
    },
    {
      path: '/system',
      name: 'system',
      component: () => import('./views/SystemView.vue'),
    },
  ],
})

export default router
