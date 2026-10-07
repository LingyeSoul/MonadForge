import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import test from 'node:test'
import { parse, compileScript } from '@vue/compiler-sfc'
import { renderToString } from '@vue/server-renderer'
import * as vue from 'vue'
import ts from 'typescript'

// Compile the real SFC so Vue's generated Boolean prop casting is exercised.
// Testing the fallback expression alone would miss absent props becoming false.
const filename = fileURLToPath(new URL('../src/components/ConfigField.vue', import.meta.url))
const { descriptor } = parse(readFileSync(filename, 'utf8'), { filename })
const compiled = compileScript(descriptor, {
  id: 'config-field-regression',
  fs: { fileExists: existsSync, readFile: path => readFileSync(path, 'utf8') },
})
const { outputText } = ts.transpileModule(compiled.content, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
})

async function displayedValue(storedValue, explicitProps = {}) {
  const exports = {}
  const require = name => {
    if (name === 'vue') return vue
    if (name === '../stores/config') {
      return { useConfigStore: () => ({ getFieldValue: () => storedValue }) }
    }
    if (name === '../composables/useI18n') return { useI18n: () => ({ t: key => key }) }
    if (name.endsWith('.vue')) return { default: {} }
    throw new Error(`Unexpected import: ${name}`)
  }
  new Function('require', 'exports', outputText)(require, exports)
  const component = exports.default
  const setup = component.setup
  // Render the actual setup bindings, without needing Vuetify or a browser DOM.
  component.setup = (props, context) => {
    const bindings = setup(props, context)
    return () => JSON.stringify({
      value: bindings.currentValue.value,
      floatDisplay: bindings.floatDisplay.value,
    })
  }
  const app = vue.createSSRApp(component, { field: { key: 'test_value' }, ...explicitProps })
  const html = await renderToString(app)
  return JSON.parse(html.replaceAll('&quot;', '"'))
}

for (const value of [32, 8, 1, 0, 0.00005, true, false, 'networks.lora_anima', '', null]) {
  test(`omitted modelValue displays stored value ${JSON.stringify(value)}`, async () => {
    assert.equal((await displayedValue(value)).value, value)
  })
}

test('float fields display the stored value, including scientific notation', async () => {
  assert.equal((await displayedValue(1)).floatDisplay, '1')
  assert.equal((await displayedValue(0.00005)).floatDisplay, '5e-5')
})

for (const value of [false, true, 0, 32, '', 'explicit', null]) {
  test(`explicit modelValue ${JSON.stringify(value)} overrides the store`, async () => {
    assert.equal((await displayedValue('stored', { modelValue: value })).value, value)
  })
}

test('explicit undefined falls back to the store', async () => {
  assert.equal((await displayedValue(32, { modelValue: undefined })).value, 32)
})
