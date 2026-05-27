import '@fontsource/roboto/300.css'
import '@fontsource/roboto/400.css'
import '@fontsource/roboto/500.css'
import '@fontsource/roboto/700.css'
import '@fontsource/jetbrains-mono/400.css'
import '@fontsource/jetbrains-mono/500.css'
import '@fontsource/jetbrains-mono/600.css'
import '@mdi/font/css/materialdesignicons.css'
import 'vuetify/styles'
import { createVuetify } from 'vuetify'
import { md3 } from 'vuetify/blueprints'

const monadForgeDark = {
  dark: true,
  colors: {
    background:     '#0C0C10',
    surface:        '#1A1A22',
    'surface-bright':  '#22222C',
    'surface-variant': '#2A2A36',
    primary:        '#C75B1A',  // Ember
    secondary:      '#D4912A',  // Amber
    'on-primary':   '#FFFFFF',
    'on-secondary': '#1A1A22',
    error:          '#CF6679',
    info:           '#64B5F6',
    success:        '#4CAF50',
    warning:        '#FB8C00',
  },
  variables: {
    'border-color':          '#2A2A36',
    'border-opacity':        '0.12',
    'high-emphasis-opacity': '0.92',
    'medium-emphasis-opacity': '0.64',
    'disabled-opacity':      '0.32',
  },
}

export default createVuetify({
  blueprint: md3,
  theme: {
    defaultTheme: 'monadForgeDark',
    themes: { monadForgeDark },
  },
  defaults: {
    VTextField: { variant: 'outlined', density: 'compact' },
    VSelect: { variant: 'outlined', density: 'compact' },
    VSwitch: { color: 'primary', density: 'compact' },
    VBtn: { variant: 'flat' },
    VCard: { elevation: 0 },
    VTextarea: { variant: 'outlined', density: 'compact' },
  },
})
