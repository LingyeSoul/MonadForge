import '@fontsource/geist/latin-400.css'
import '@fontsource/geist/latin-500.css'
import '@fontsource/geist/latin-600.css'
import '@fontsource/geist/latin-700.css'
import '@fontsource/jetbrains-mono/latin-400.css'
import '@fontsource/jetbrains-mono/latin-500.css'
import '@fontsource/jetbrains-mono/latin-600.css'
import '@mdi/font/css/materialdesignicons.css'
import 'vuetify/styles'
import { createVuetify } from 'vuetify'
import { md3 } from 'vuetify/blueprints'

const dark = {
  dark: true,
  colors: {
    background:           '#141618',
    surface:              '#1C1F22',
    'surface-bright':     '#272B2F',
    'surface-variant':    '#34393E',
    'on-surface':         '#EEF0F2',
    'on-surface-variant': '#A8AFB6',
    primary:              '#EBA375',
    secondary:            '#88BDB5',
    'on-primary':         '#251C16',
    'on-secondary':       '#152522',
    error:                '#F18C96',
    info:                 '#91B9EA',
    success:              '#89C5A2',
    warning:              '#E3BC75',
    outline:              '#42484E',
    'outline-variant':    '#2A2E33',
  },
  variables: {
    'border-color':          '#A8AFB6',
    'border-opacity':        '0.18',
    'high-emphasis-opacity': '0.92',
    'medium-emphasis-opacity': '0.64',
    'disabled-opacity':      '0.32',
  },
}

const light = {
  dark: false,
  colors: {
    background:           '#F6F7F8',
    surface:              '#FFFFFF',
    'surface-bright':     '#FFFFFF',
    'surface-variant':    '#E7EBEE',
    'on-surface':         '#242A30',
    'on-surface-variant': '#626C76',
    primary:              '#9D4C23',
    secondary:            '#34756C',
    'on-primary':         '#FFFFFF',
    'on-secondary':       '#FFFFFF',
    error:                '#BA3B50',
    info:                 '#356FA8',
    success:              '#327653',
    warning:              '#956914',
    outline:              '#B7C0C8',
    'outline-variant':    '#E0E5E9',
  },
  variables: {
    'border-color':          '#626C76',
    'border-opacity':        '0.2',
    'high-emphasis-opacity': '0.92',
    'medium-emphasis-opacity': '0.64',
    'disabled-opacity':      '0.32',
  },
}

export default createVuetify({
  blueprint: md3,
  theme: {
    defaultTheme: 'dark',
    themes: { dark, light },
  },
  defaults: {
    VTextField: { variant: 'outlined', density: 'compact' },
    VSelect:    { variant: 'outlined', density: 'compact' },
    VSwitch:    { color: 'secondary', density: 'compact' },
    VBtn:       { variant: 'flat', rounded: 'sm' },
    VCard:      { elevation: 0, rounded: 'sm' },
    VTooltip:   { openDelay: 350 },
    VTextarea:  { variant: 'outlined', density: 'compact' },
    VChip:      { size: 'small', variant: 'tonal' },
  },
})
