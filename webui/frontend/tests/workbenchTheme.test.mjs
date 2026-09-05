import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'
import en from '../src/i18n/en.ts'
import cn from '../src/i18n/cn.ts'
import ja from '../src/i18n/ja.ts'
import ko from '../src/i18n/ko.ts'

const source = ts.createSourceFile('vuetify.ts',
  readFileSync(new URL('../src/plugins/vuetify.ts', import.meta.url), 'utf8'),
  ts.ScriptTarget.Latest, true)

function colorsFor(name) {
  const declarations = source.statements
    .filter(ts.isVariableStatement)
    .flatMap(statement => [...statement.declarationList.declarations])
  const theme = declarations.find(declaration => declaration.name.getText(source) === name)
  const colors = theme.initializer.properties.find(property => property.name.getText(source) === 'colors')
  return Object.fromEntries(colors.initializer.properties.map(property => [property.name.text, property.initializer.text]))
}

function luminance(hex) {
  const channels = hex.slice(1).match(/../g).map(value => parseInt(value, 16) / 255)
    .map(value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4)
  return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722
}

function contrast(a, b) {
  const values = [luminance(a), luminance(b)].sort((x, y) => y - x)
  return (values[0] + 0.05) / (values[1] + 0.05)
}

for (const name of ['dark', 'light']) {
  test(`${name} theme has readable text and action colors`, () => {
    const colors = colorsFor(name)
    for (const [foreground, background] of [
      ['on-surface', 'surface'],
      ['on-surface-variant', 'background'],
      ['on-primary', 'primary'],
      ['on-secondary', 'secondary'],
      ['primary', 'background'],
    ]) {
      const ratio = contrast(colors[foreground], colors[background])
      assert.ok(ratio >= 4.5, `${name} ${foreground}/${background}: ${ratio.toFixed(2)}`)
    }
  })
}

test('workbench labels are translated in all four languages', () => {
  const keys = [
    'workspaceStudio', 'workspaceNavigation', 'workspaceTraining', 'workspaceData',
    'workspaceModels', 'workspaceLocal', 'workspaceLanguage', 'workspaceExpandNav',
    'workspaceCollapseNav', 'cfgWorkspaceTitle', 'cfgWorkspaceMeta', 'cfgMethodLibrary',
    'cfgVariantLibrary', 'cfgSearchFields', 'cfgNoMatchingFields', 'cfgClearSearch',
    'cfgPendingChanges', 'cfgParametersCount', 'cfgOpenGuide', 'dashOpenConfig',
    'dsGridView', 'dsListView', 'dsTreeView',
  ]
  for (const [name, messages] of Object.entries({ en, cn, ja, ko })) {
    for (const key of keys) {
      assert.equal(typeof messages[key], 'string', `${name}: ${key}`)
      assert.ok(messages[key].length > 0, `${name}: ${key}`)
    }
  }
})
