import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import vuetify from './plugins/vuetify'
import { useAppStore } from './stores/app'
import { initI18n } from './composables/useI18n'
import './styles/main.scss'

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)
app.use(vuetify)
app.use(router)

const appStore = useAppStore()
appStore.init()

initI18n().then(() => {
  app.mount('#app')
})
