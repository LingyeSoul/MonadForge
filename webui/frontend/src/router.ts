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
      path: '/adapter',
      name: 'adapter',
      component: () => import('./views/AdapterView.vue'),
    },
    {
      path: '/merge',
      name: 'merge',
      component: () => import('./views/MergeView.vue'),
    },
    {
      path: '/tasks',
      name: 'tasks',
      component: () => import('./views/TaskMonitorView.vue'),
    },
    {
      path: '/system',
      name: 'system',
      component: () => import('./views/SystemView.vue'),
    },
  ],
})

export default router
