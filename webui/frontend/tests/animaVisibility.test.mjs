import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import * as vue from 'vue'
import * as pinia from 'pinia'
import ts from 'typescript'

const source = readFileSync(new URL('../src/stores/config.ts', import.meta.url), 'utf8')
function setup() {
  pinia.setActivePinia(pinia.createPinia())
  const exports = {}
  const dependencies = { vue, pinia, './app': { useAppStore: () => ({ language: 'en' }) },
    '../i18n': { getFrontendTranslations: () => ({}) } }
  const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
  new Function('require', 'exports', code)(name => dependencies[name], exports)
  return exports.useConfigStore()
}
const field = (key, value, visible_when, group = 'Architecture', is_basic = false) => ({ key, value, visible_when, group, is_basic })
const visibleKeys = store => Object.values(store.groupedAdvanced).flat().map(field => field.key)

test('Anima algorithm edits immediately hide inactive controls while retaining their values', () => {
  const store = setup()
  store.fields = [field('use_lokr', false), field('lokr_factor', 8, [{ use_lokr: [true] }]),
    field('use_glokr', false), field('decompose_both', false, [{ use_lokr: [true] }, { use_glokr: [true] }])]
  assert.deepEqual(visibleKeys(store), ['use_lokr', 'use_glokr'])
  store.setFieldValue('use_lokr', true)
  assert.ok(visibleKeys(store).includes('lokr_factor'))
  store.setFieldValue('lokr_factor', 16)
  store.setFieldValue('use_lokr', false)
  assert.ok(!visibleKeys(store).includes('lokr_factor'))
  assert.equal(store.getFieldValue('lokr_factor'), 16)
  assert.equal(store.editedValues.lokr_factor, 16)
  store.setFieldValue('use_glokr', true)
  assert.ok(visibleKeys(store).includes('decompose_both'))
  assert.ok(!visibleKeys(store).includes('lokr_factor'))
  store.setFieldValue('use_lokr', true)
  assert.equal(store.getFieldValue('lokr_factor'), 16)
})

test('the same visibility conditions apply to basic fields and remove empty advanced groups', () => {
  const store = setup()
  store.fields = [field('use_repa', false, null, 'Architecture', true),
    field('repa_weight', 0.05, [{ use_repa: [true] }], 'Alignment'),
    field('repa_layer', 8, [{ use_repa: [true] }], 'Architecture', true)]
  assert.deepEqual(store.basicFields.map(field => field.key), ['use_repa'])
  assert.deepEqual(store.groupedAdvanced, {})
  store.setFieldValue('use_repa', true)
  assert.deepEqual(store.basicFields.map(field => field.key), ['use_repa', 'repa_layer'])
  assert.deepEqual(visibleKeys(store), ['repa_weight'])
})

test('legacy network_args use the same precedence and boolean values as the trainer', () => {
  const store = setup()
  store.fields = [field('use_lokr', false), field('network_args', ['use_lokr=TRUE']),
    field('lokr_factor', 8, [{ use_lokr: [true] }])]
  assert.ok(visibleKeys(store).includes('lokr_factor'))
  store.setFieldValue('use_lokr', true)
  store.setFieldValue('network_args', ['use_lokr=false'])
  assert.ok(!visibleKeys(store).includes('lokr_factor'))
  store.setFieldValue('network_args', [])
  assert.ok(visibleKeys(store).includes('lokr_factor'))
})

test('AND conditions require both the correct module and enabled feature; missing optional flags match null', () => {
  const store = setup()
  store.fields = [field('network_module', 'networks.methods.controlnet_lllite'), field('use_lokr', true),
    field('lokr_factor', 8, [{ network_module: [null, 'networks.lora_anima'], use_lokr: [true] }]),
    field('down_init', 'weight_svd', [{ use_lokr: [null, false], use_ortho: [null, false] }])]
  assert.ok(!visibleKeys(store).includes('lokr_factor'))
  store.setFieldValue('network_module', 'networks.lora_anima')
  assert.ok(visibleKeys(store).includes('lokr_factor'))
  store.setFieldValue('use_lokr', false)
  assert.ok(visibleKeys(store).includes('down_init'))
})

test('saving an inactive field edit preserves it through the returned complete field metadata', async t => {
  const store = setup()
  store.variant = 'lora'; store.preset = 'default'
  const fields = [field('use_lokr', false), field('lokr_factor', 8, [{ use_lokr: [true] }])]
  store.fields = fields
  store.setFieldValue('lokr_factor', 16)
  let submitted
  t.mock.method(globalThis, 'fetch', async (_url, init = {}) => {
    if (init.method === 'PUT') { submitted = JSON.parse(init.body).data; return new Response('{}') }
    return new Response(JSON.stringify({ fields: [fields[0], { ...fields[1], value: 16 }] }))
  })
  await store.save()
  assert.equal(submitted.lokr_factor, 16)
  assert.ok(!visibleKeys(store).includes('lokr_factor'))
  store.setFieldValue('use_lokr', true)
  assert.equal(store.getFieldValue('lokr_factor'), 16)
  assert.ok(visibleKeys(store).includes('lokr_factor'))
})
