import assert from 'node:assert/strict'
import test from 'node:test'
import { createPinia, setActivePinia } from 'pinia'
import { initialDraft, parseSchema, readDraft, resolvedPaths, resultImages } from '../src/utils/qwen21.ts'
import { useQwenWorkspace, useMossWorkspace } from '../src/stores/qwenWorkspace.ts'
import { taskModelFamily, useModelWorkspace } from '../src/stores/modelWorkspace.ts'
import { modelFieldLabel, mossText } from '../src/i18n/modelWorkspace.ts'

function field(name, value) {
  return { name, kind: typeof value === 'number' ? 'int' : 'str', value, nullable: false, choices: [], help: name, advanced: false, multiline: false }
}
function schema(family) {
  const model = field('model_dir', `/models/${family}`)
  return { train: [model, field('rank', 32)], cache: [model, field('src', `/data/${family}`)], generate: [model, field('prompt', family)], ...(family === 'moss' ? { tune: [model, field('rank', 64), field('margin_gb', 2)] } : {}) }
}
function setup(t) {
  const entries = new Map()
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: { getItem: key => entries.get(key) ?? null, setItem: (key, value) => entries.set(key, String(value)), removeItem: key => entries.delete(key) } })
  t.after(() => { if (descriptor) Object.defineProperty(globalThis, 'localStorage', descriptor); else delete globalThis.localStorage })
  setActivePinia(createPinia())
  return entries
}
const response = value => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } })

test('MOSS tuning values survive profile load, edit and serialization', () => {
  const parsed = parseSchema(schema('moss'))
  const draft = initialDraft(parsed)
  draft.tune.rank = 80
  draft.tune.margin_gb = 4
  const loaded = readDraft(parsed, JSON.stringify(draft))
  loaded.train.rank = 16
  assert.deepEqual(loaded.tune, { rank: 80, margin_gb: 4 })
  assert.deepEqual(readDraft(parsed, JSON.stringify(loaded)).tune, loaded.tune)
  assert.equal(initialDraft(parseSchema(schema('qwen21'))).tune, undefined)
})

test('model workspaces isolate profiles and route cache and training through their own API', async t => {
  setup(t)
  const calls = []
  const profiles = Object.fromEntries(['moss', 'qwen21'].map(family => [family, initialDraft(parseSchema(schema(family)))]))
  profiles.moss.tune.rank = 88
  t.mock.method(globalThis, 'fetch', async (url, init = {}) => {
    calls.push({ url, init })
    if (url === '/api/system/runtimes') return response({ runtimes: [{ name: 'default', available: true, python: '/python/default', error: null }, { name: 'curation', available: true, python: '/python/curation', error: null }, { name: 'v100', available: false, python: null, error: 'V100 is not registered' }] })
    const family = url.split('/')[2]
    if (url.endsWith('/schema')) return response(schema(family))
    if (url.endsWith('/profiles')) return response(['default'])
    if (url.endsWith('/profiles/default')) {
      if (init.method === 'PUT') profiles[family] = JSON.parse(init.body)
      return response(profiles[family])
    }
    if (url.includes('/jobs/')) return response({ task_id: `${family}-job` })
    throw new Error(`Unexpected URL ${url}`)
  })
  const moss = useMossWorkspace(), qwen = useQwenWorkspace()
  await Promise.all([moss.ensureLoaded(), qwen.ensureLoaded()])
  moss.update('train', 'rank', 8)
  assert.equal(qwen.draft.train.rank, 32)
  await moss.save()
  assert.equal(profiles.moss.tune.rank, 88)
  assert.equal(profiles.qwen21.models.model_dir, '/models/qwen21')
  const workspace = useModelWorkspace()
  await workspace.loadRuntimes()
  workspace.selectRuntime('curation')
  assert.throws(() => workspace.selectRuntime('v100'), /not registered/)
  assert.equal(workspace.runtimeProfile, 'curation')
  assert.equal(await moss.submit('train', {}), 'moss-job')
  assert.equal(await qwen.submit('cache', {}), 'qwen21-job')
  const train = calls.find(call => call.url === '/api/moss/jobs/train')
  assert.deepEqual(JSON.parse(train.init.body), { values: { model_dir: '/models/moss', rank: 8 }, runtime_profile: 'curation' })
  const cache = calls.find(call => call.url === '/api/qwen21/jobs/cache')
  assert.deepEqual(JSON.parse(cache.init.body), { values: { model_dir: '/models/qwen21', src: '/data/qwen21' }, runtime_profile: 'curation' })
})

test('concurrent submit clicks are guarded while a model action is pending', async t => {
  setup(t)
  const moss = useMossWorkspace()
  let resolve, actions = 0
  const first = moss.perform(async () => { actions++; await new Promise(done => { resolve = done }) })
  await moss.perform(async () => { actions++ })
  assert.equal(actions, 1)
  assert.equal(moss.busy, true)
  resolve()
  await first
  assert.equal(moss.busy, false)
})

test('saved invalid model selection does not blank the app and task labels stay task-bound', t => {
  const storage = setup(t)
  storage.set('monadforge-model-family', 'removed-experiment')
  const workspace = useModelWorkspace()
  assert.equal(workspace.family, 'anima')
  workspace.select('moss')
  assert.equal(storage.get('monadforge-model-family'), 'moss')
  assert.equal(taskModelFamily('qwen21-generate'), 'qwen21')
  assert.equal(taskModelFamily('moss-cache-train'), 'moss')
  assert.equal(taskModelFamily('lora'), 'anima')
})

test('MOSS outputs retain their model identity without inventing Qwen multipliers', () => {
  const moss = resultImages({ images: [{ file: 'one.png', seed: 2, prompt: 'sky', model: 'edit' }] }, 'moss')[0]
  assert.equal(moss.multiplier, null)
  assert.equal(moss.model, 'edit')
  const qwen = resultImages({ images: [{ file: 'one.png', seed: 2, prompt: 'sky', multiplier: 0 }] })[0]
  assert.equal(qwen.multiplier, 0)
  assert.deepEqual(resolvedPaths({ dit: '/m/dit', text_encoder: '/m/te', vae: '/m/vae', tokenizer: '/m/tok' }, 'moss'), { dit: '/m/dit', text_encoder: '/m/te', vae: '/m/vae', tokenizer: '/m/tok' })
  assert.equal(modelFieldLabel('moss', 'cn', 'algo', 'qwenField_algo'), '适配器算法')
  assert.match(mossText('cn', 'cacheHint'), /MOSS/)
  assert.doesNotMatch(mossText('cn', 'cacheHint'), /RGBA/)
})

test('unavailable saved V100 runtime blocks submission without falling back', async t => {
  const storage = setup(t)
  storage.set('monadforge-runtime-profile', 'v100')
  const calls = []
  t.mock.method(globalThis, 'fetch', async url => {
    calls.push(url)
    return response({ runtimes: [{ name: 'default', available: true, python: '/python/default', error: null }, { name: 'v100', available: false, python: null, error: 'V100 missing' }] })
  })
  const workspace = useModelWorkspace()
  await assert.rejects(useMossWorkspace().submit('generate', {}), /V100 missing/)
  assert.equal(workspace.runtimeProfile, 'v100')
  assert.deepEqual(calls, ['/api/system/runtimes'])
})
