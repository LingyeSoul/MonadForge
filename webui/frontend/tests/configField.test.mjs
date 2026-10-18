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

function loadComponent(storedValue) {
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
  return exports.default
}

async function displayedValue(storedValue, explicitProps = {}) {
  const component = loadComponent(storedValue)
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

function floatEditor(t, value, nullable = false) {
  const props = vue.reactive({ field: { key: 'test_float', field_type: 'float', nullable }, modelValue: value })
  const scope = vue.effectScope()
  t.after(() => scope.stop())
  // Exercise the component's actual handlers and watcher with the same numeric
  // parent echo produced by ConfigEditor and QwenFields after an update event.
  const bindings = scope.run(() => loadComponent(value).setup(props, {
    expose() {},
    emit(event, nextValue) {
      assert.equal(event, 'update')
      props.modelValue = nextValue
    },
  }))
  return {
    props,
    display: () => bindings.floatDisplay.value,
    async input(text) { bindings.onFloatInput(text); await vue.nextTick() },
    async blur() { bindings.onFloatBlur(); await vue.nextTick() },
  }
}

test('decimal editing preserves the point and trailing zeros until blur', async t => {
  const editor = floatEditor(t, 0.25)
  for (const text of ['0.', '0.0', '0.05', '0.050']) {
    await editor.input(text)
    assert.equal(editor.display(), text)
    assert.equal(editor.props.modelValue, Number(text))
  }
  await editor.blur()
  assert.equal(editor.display(), '0.05')
})

test('scientific notation keeps incomplete input without updating the value', async t => {
  const editor = floatEditor(t, 0.25)
  for (const text of ['1', '1e', '1e-']) {
    await editor.input(text)
    assert.equal(editor.display(), text)
    assert.equal(editor.props.modelValue, 1)
  }
  await editor.input('1e-07')
  assert.equal(editor.display(), '1e-07')
  assert.equal(editor.props.modelValue, 1e-7)
  await editor.blur()
  assert.equal(editor.display(), '1e-7')
})

test('external value changes replace a pending float edit', async t => {
  const editor = floatEditor(t, 0.25)
  await editor.input('1e-')
  editor.props.modelValue = 2
  await vue.nextTick()
  assert.equal(editor.display(), '2')
})

for (const nullable of [false, true]) {
  test(`empty float input preserves nullable=${nullable} behavior on blur`, async t => {
    const editor = floatEditor(t, 0.25, nullable)
    await editor.input('')
    assert.equal(editor.display(), '')
    assert.equal(editor.props.modelValue, 0.25)
    await editor.blur()
    assert.equal(editor.props.modelValue, nullable ? null : 0.25)
    assert.equal(editor.display(), nullable ? '' : '0.25')
  })
}

test('invalid float input restores the current numeric value on blur', async t => {
  const editor = floatEditor(t, 0.25)
  await editor.input('1e-')
  assert.equal(editor.props.modelValue, 0.25)
  await editor.blur()
  assert.equal(editor.display(), '0.25')
})
