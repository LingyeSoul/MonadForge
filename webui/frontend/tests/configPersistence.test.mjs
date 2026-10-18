import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { parse, compileScript } from '@vue/compiler-sfc'
import * as vue from 'vue'
import * as pinia from 'pinia'
import ts from 'typescript'
const source = path => readFileSync(new URL(`../src/${path}`, import.meta.url), 'utf8')
const editorSource = compileScript(parse(source('views/ConfigEditor.vue')).descriptor, { id: 'config-persistence' }).content
function evaluate(text, deps) {
  const exports = {}
  const code = ts.transpileModule(text, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
  new Function('require', 'exports', code)(name => {
    if (name.endsWith('.vue')) return { default: {} }
    assert.ok(name in deps, `Unexpected import: ${name}`)
    return deps[name]
  }, exports)
  return exports
}
const metadata = value => ({ key: 'network_dim', value, field_type: 'int', group: 'basic', origin: 'method', read_only: false, is_virtual: false })
const response = (value, status = 200) => new Response(JSON.stringify(value), { status })
function setup(t) {
  pinia.setActivePinia(pinia.createPinia())
  const app = vue.reactive({ language: 'en' }), appModule = { useAppStore: () => app }
  const configModule = evaluate(source('stores/config.ts'), { vue, pinia, './app': appModule, '../i18n': { getFrontendTranslations: () => ({ cfgErrLoadConfig: '{error}', cfgErrSave: '{error}' }) } })
  const config = configModule.useConfigStore(), mounted = [], leaveGuards = [], listeners = new Map()
  const previousWindow = globalThis.window
  globalThis.window = { addEventListener: (name, fn) => listeners.set(name, fn), removeEventListener: name => listeners.delete(name) }
  t.after(() => { globalThis.window = previousWindow })
  let persistedRank = 32
  const calls = []
  t.mock.method(globalThis, 'fetch', async (url, init = {}) => {
    calls.push({ url, init })
    if (init.method === 'PUT') { persistedRank = JSON.parse(init.body).data.network_dim ?? persistedRank; return response({}) }
    if (url.includes('/merged?')) return response({ fields: [metadata(persistedRank)] })
    if (url.includes('/variants')) return response({ variants: ['lora'] })
    if (url.includes('/methods')) return response({ methods: ['lora'] })
    if (url.includes('/presets')) return response({ presets: ['default'] })
    if (url.includes('/variant-meta')) return response({ family: 'lora' })
    return response({})
  })
  const workspace = vue.reactive({ family: 'anima' })
  const qwen = vue.reactive({ family: 'qwen21', busy: false, loading: false, dirty: false, profiles: [], profileAlgorithms: {}, configMethod: '', configProfile: '' })
  const editorModule = evaluate(editorSource, {
    vue: { ...vue, onMounted: fn => mounted.push(fn), onBeforeUnmount() {} },
    'vue-router': { useRoute: () => ({ query: {} }), useRouter: () => ({ push: async () => {} }), onBeforeRouteLeave: fn => leaveGuards.push(fn) },
    '../stores/modelWorkspace': { useModelWorkspace: () => workspace, modelLabel: value => value },
    '../stores/qwenWorkspace': { useActiveModelWorkspace: () => vue.computed(() => qwen), useQwenWorkspace: () => qwen, useMossWorkspace: () => qwen },
    '../stores/config': configModule, '../stores/task': { useTaskStore: () => ({ loading: false }) }, '../stores/notify': { useNotifyStore: () => ({ show() {} }) }, '../stores/app': appModule,
    '../composables/useI18n': { useI18n: () => ({ t: key => key }) }, '../i18n/modelWorkspace': { mossText: () => '' }, '../utils/modelConfig': { modelAlgorithmItems: () => [] }, '../utils/configSearch': { matchesConfigSearch: () => true },
    '../composables/usePreprocessRunStorage': { readPreprocessRun: () => null, writePreprocessRun() {} },
    '../composables/useTrainingContinuation': { useTrainingContinuation: () => ({ active: false, selected: null, candidates: [], presetItems: [], exit() {}, async fetchCandidates() {} }) },
  })
  config.method = 'lora'; config.variant = 'lora'; config.fields = [metadata(32)]
  const scope = vue.effectScope()
  t.after(() => scope.stop())
  const editor = scope.run(() => editorModule.default.setup({}, { expose() {} }))
  return { config, editor, calls, listeners, leave: () => leaveGuards[0](), mount: async () => { for (const fn of mounted) await fn() } }
}
test('Anima navigation cancellation preserves fields and Extra Args; discard releases navigation', async t => {
  const { config, editor, leave, mount, listeners } = setup(t)
  await mount(); config.setFieldValue('network_dim', 99); editor.extraArgs.value = 'optimizer_type = "AdamW"'
  const blocked = leave(); editor.cancelWorkspaceChange()
  assert.equal(await blocked, false); assert.equal(config.getFieldValue('network_dim'), 99); assert.equal(editor.extraArgs.value, 'optimizer_type = "AdamW"')
  let prevented = false
  listeners.get('beforeunload')({ preventDefault() { prevented = true } }); assert.equal(prevented, true)
  const discarded = leave(); await editor.discardWorkspaceChange()
  assert.equal(await discarded, true); assert.equal(config.getFieldValue('network_dim'), 32); assert.equal(config.dirty, false); assert.equal(editor.extraArgs.value, '')
})
test('Anima navigation saves before leaving and stays put on save failure', async t => {
  const { config, editor, leave, mount, calls } = setup(t)
  await mount(); config.setFieldValue('network_dim', 48); editor.extraArgs.value = 'optimizer_type = "AdamW"'
  const allowed = leave(); await editor.saveWorkspaceChange()
  assert.equal(await allowed, true); assert.equal(config.getFieldValue('network_dim'), 48); assert.equal(config.dirty, false); assert.equal(editor.extraArgs.value, '')
  assert.equal(JSON.parse(calls.find(call => call.init.method === 'PUT').init.body).data.extra_args, 'optimizer_type = "AdamW"')
  config.setFieldValue('network_dim', 71)
  t.mock.method(globalThis, 'fetch', async () => response({ detail: 'disk unavailable' }, 503))
  const denied = leave(); await editor.saveWorkspaceChange()
  assert.equal(editor.pendingChange.value.kind, 'leave'); assert.equal(config.getFieldValue('network_dim'), 71); assert.equal(config.dirty, true)
  editor.cancelWorkspaceChange(); assert.equal(await denied, false)
})
for (const failure of ['http', 'schema']) test(`merged-config ${failure} failure preserves fields, selection and unsaved edits`, async t => {
  const { config } = setup(t); config.setFieldValue('network_dim', 71)
  t.mock.method(globalThis, 'fetch', async () => response({ detail: 'unavailable' }, failure === 'http' ? 503 : 200))
  await assert.rejects(config.fetchMerged('other', 'other'))
  assert.equal(config.variant, 'lora'); assert.equal(config.fields.length, 1); assert.equal(config.getFieldValue('network_dim'), 71); assert.equal(config.dirty, true); assert.notEqual(config.error, '')
})
test('saving acknowledges only submitted edits and keeps newer edits dirty', async t => {
  const { config } = setup(t); config.setFieldValue('network_dim', 40)
  let release, submitted
  t.mock.method(globalThis, 'fetch', async (_url, init = {}) => {
    if (init.method === 'PUT') { submitted = JSON.parse(init.body); return new Promise(resolve => { release = () => resolve(response({})) }) }
    return response({ fields: [metadata(40)] })
  })
  const pending = config.save(); config.setFieldValue('network_dim', 55); release(); await pending
  assert.equal(submitted.data.network_dim, 40); assert.equal(config.fields[0].value, 40); assert.equal(config.getFieldValue('network_dim'), 55); assert.equal(config.dirty, true)
})
test('a failed refresh after PUT cannot clear the draft or report save success', async t => {
  const { config } = setup(t); config.setFieldValue('network_dim', 40)
  t.mock.method(globalThis, 'fetch', async (_url, init = {}) => init.method === 'PUT' ? response({}) : response({ detail: 'refresh unavailable' }, 503))
  await assert.rejects(config.save(), /refresh unavailable/); assert.equal(config.getFieldValue('network_dim'), 40); assert.equal(config.dirty, true)
})
test('a failed variant switch cannot pair a new selector with the previous loaded configuration', async t => {
  const { config, editor, mount } = setup(t)
  await mount()
  t.mock.method(globalThis, 'fetch', async () => response({ detail: 'variant unavailable' }, 503))
  await editor.requestVariantChange('missing')
  assert.equal(editor.selectedVariant.value, 'lora')
  assert.equal(config.variant, 'lora')
  assert.equal(config.getFieldValue('network_dim'), 32)
  assert.match(editor.workspaceError.value, /variant unavailable/)
})
