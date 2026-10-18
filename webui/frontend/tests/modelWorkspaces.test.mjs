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

function configWorkspace(t) {
  const storage = setup(t)
  const schemas = Object.fromEntries(['qwen21', 'moss'].map(family => {
    const value = schema(family)
    value.train.push({ ...field('algo', 'lora'), choices: ['lora', 'lokr', 'loha'] })
    return [family, value]
  }))
  const documents = Object.fromEntries(Object.entries(schemas).map(([family, value]) => {
    const defaultDraft = initialDraft(parseSchema(value))
    const kr = structuredClone(defaultDraft)
    kr.train.algo = 'lokr'
    kr.train.rank = 48
    return [family, { default: defaultDraft, kr }]
  }))
  const calls = []
  const failures = new Set()
  t.mock.method(globalThis, 'fetch', async (url, init = {}) => {
    calls.push({ url, init })
    if (failures.has(`${init.method ?? 'GET'} ${url}`)) return new Response(JSON.stringify({ detail: 'profile unavailable' }), { status: 503 })
    const [, , family, resource, name] = url.split('/')
    if (resource === 'schema') return response(schemas[family])
    if (resource === 'profiles' && !name) return response(Object.keys(documents[family]))
    if (resource === 'profiles' && name) {
      if (init.method === 'PUT') documents[family][name] = JSON.parse(init.body)
      return response(documents[family][name])
    }
    throw new Error(`Unexpected URL ${url}`)
  })
  return { storage, documents, calls, failures, qwen: useQwenWorkspace(), moss: useMossWorkspace() }
}

test('configuration catalog reads method ownership without selecting or replacing the working draft', async t => {
  const { qwen, calls } = configWorkspace(t)
  await qwen.ensureLoaded()
  qwen.update('train', 'rank', 99)
  const currentDraft = qwen.draft
  await qwen.loadProfileCatalog()
  assert.deepEqual(qwen.profileAlgorithms, { default: 'lora', kr: 'lokr' })
  assert.equal(qwen.configMethod, '')
  assert.equal(qwen.configProfile, '')
  assert.equal(qwen.draft, currentDraft)
  assert.equal(qwen.draft.train.rank, 99)
  assert.equal(qwen.dirty, true)
  qwen.selectConfigMethod('lokr')
  assert.equal(qwen.configMethod, 'lokr')
  assert.equal(qwen.configProfile, '')
  assert.equal(qwen.draft, currentDraft)
  assert.equal(calls.some(call => call.init.method === 'PUT'), false)
  assert.throws(() => qwen.selectConfigMethod('unknown'), /Unsupported/)
})

test('method selection accepts only matching profiles and isolates each model workspace', async t => {
  const { qwen, moss, calls } = configWorkspace(t)
  await Promise.all([qwen.loadProfileCatalog(), moss.loadProfileCatalog()])
  qwen.selectConfigMethod('lokr')
  const beforeMismatch = calls.length
  await assert.rejects(qwen.selectConfigProfile('default'), /does not belong/)
  assert.equal(calls.length, beforeMismatch)
  assert.equal(qwen.configProfile, '')
  await qwen.selectConfigProfile('kr')
  assert.equal(qwen.configProfile, 'kr')
  assert.equal(qwen.profile, 'kr')
  assert.equal(qwen.draft.train.algo, 'lokr')
  assert.equal(qwen.dirty, false)
  assert.equal(moss.configMethod, '')
  assert.equal(moss.configProfile, '')
  assert.equal(moss.draft.train.algo, 'lora')
  assert.throws(() => qwen.update('train', 'algo', 'lora'), /Choose a method/)
})

test('a profile changed on disk cannot silently replace the selected method or draft', async t => {
  const { qwen, documents, storage, failures } = configWorkspace(t)
  await qwen.loadProfileCatalog()
  qwen.selectConfigMethod('lokr')
  const previous = qwen.draft
  documents.qwen21.kr.train.algo = 'lora'
  await assert.rejects(qwen.selectConfigProfile('kr'), /does not match/)
  assert.equal(qwen.draft, previous)
  assert.equal(qwen.profile, 'default')
  assert.equal(qwen.configMethod, 'lokr')
  assert.equal(qwen.configProfile, '')
  assert.equal(storage.get('monadforge-qwen-profile'), 'default')
  documents.qwen21.kr.train.algo = 'lokr'
  failures.add('GET /api/qwen21/profiles/kr')
  await assert.rejects(qwen.selectConfigProfile('kr'), /profile unavailable/)
  assert.equal(qwen.draft, previous)
  assert.equal(qwen.configProfile, '')
})

test('reset and discard preserve the selected method for saved and new configurations', async t => {
  const { qwen } = configWorkspace(t)
  await qwen.loadProfileCatalog()
  qwen.selectConfigMethod('lokr')
  await qwen.selectConfigProfile('kr')
  qwen.resetDefaults()
  assert.equal(qwen.draft.train.algo, 'lokr')
  assert.equal(qwen.draft.train.rank, 32)
  assert.equal(qwen.dirty, true)
  qwen.discard()
  assert.equal(qwen.draft.train.algo, 'lokr')
  assert.equal(qwen.draft.train.rank, 48)
  assert.equal(qwen.dirty, false)
  qwen.selectConfigMethod('loha')
  qwen.beginConfigProfile('new-ha')
  qwen.update('train', 'rank', 72)
  qwen.discard()
  assert.equal(qwen.draft.train.algo, 'loha')
  assert.equal(qwen.draft.train.rank, 32)
  assert.equal(qwen.configProfile, 'new-ha')
})

test('creating a profile uses chosen method defaults or copies an explicitly selected configuration', async t => {
  const { qwen, documents, calls } = configWorkspace(t)
  await qwen.loadProfileCatalog()
  qwen.update('train', 'rank', 99)
  qwen.selectConfigMethod('lokr')
  await qwen.createConfigProfile('new-kr')
  assert.equal(qwen.draft.train.rank, 32)
  assert.equal(qwen.draft.train.algo, 'lokr')
  assert.equal(qwen.configProfile, 'new-kr')
  assert.equal(qwen.profileAlgorithms['new-kr'], 'lokr')
  assert.equal(qwen.dirty, false)
  assert.equal(documents.qwen21.default.train.algo, 'lora')
  assert.equal(documents.qwen21.default.train.rank, 32)
  qwen.update('train', 'rank', 61)
  await qwen.createConfigProfile('copy-kr')
  assert.equal(documents.qwen21['copy-kr'].train.rank, 61)
  assert.equal(documents.qwen21['copy-kr'].train.algo, 'lokr')
  assert.equal(calls.filter(call => call.init.method === 'PUT').length, 2)
  await assert.rejects(qwen.createConfigProfile('default'), /already exists/)
  assert.equal(qwen.profile, 'copy-kr')
  await qwen.createConfigProfile('fresh-kr', false)
  assert.equal(documents.qwen21['fresh-kr'].train.rank, 32)
  assert.equal(documents.qwen21['fresh-kr'].train.algo, 'lokr')
})

test('failed new-profile saves restore unsaved edits and all prior selection state', async t => {
  const { qwen, failures, storage } = configWorkspace(t)
  await qwen.loadProfileCatalog()
  qwen.selectConfigMethod('lokr')
  await qwen.selectConfigProfile('kr')
  qwen.update('train', 'rank', 71)
  const previousDraft = qwen.draft
  const previousSaved = qwen.saved
  failures.add('PUT /api/qwen21/profiles/copy-kr')
  await assert.rejects(qwen.createConfigProfile('copy-kr'), /profile unavailable/)
  assert.equal(qwen.draft, previousDraft)
  assert.equal(qwen.draft.train.rank, 71)
  assert.equal(qwen.saved, previousSaved)
  assert.equal(qwen.profile, 'kr')
  assert.equal(qwen.configMethod, 'lokr')
  assert.equal(qwen.configProfile, 'kr')
  assert.equal(qwen.dirty, true)
  assert.equal(qwen.profiles.includes('copy-kr'), false)
  assert.equal(qwen.profileAlgorithms['copy-kr'], undefined)
  assert.equal(storage.get('monadforge-qwen-profile'), 'kr')
})

test('catalog failures remain visible and preserve the previous usable catalog and draft', async t => {
  const { qwen, failures } = configWorkspace(t)
  await qwen.loadProfileCatalog()
  const previousCatalog = qwen.profileAlgorithms
  const previousDraft = qwen.draft
  failures.add('GET /api/qwen21/profiles/kr')
  await assert.rejects(qwen.loadProfileCatalog(), /profile unavailable/)
  assert.equal(qwen.profileAlgorithms, previousCatalog)
  assert.equal(qwen.draft, previousDraft)
})

test('switching methods while a profile loads prevents stale selection and draft replacement', async t => {
  const { qwen, documents } = configWorkspace(t)
  await qwen.loadProfileCatalog()
  qwen.selectConfigMethod('lokr')
  const previousDraft = qwen.draft
  const originalFetch = globalThis.fetch
  let finishLoad
  t.mock.method(globalThis, 'fetch', (url, init) => {
    if (url === '/api/qwen21/profiles/kr') return new Promise(resolve => { finishLoad = resolve })
    return originalFetch(url, init)
  })
  const loading = qwen.selectConfigProfile('kr')
  qwen.selectConfigMethod('loha')
  finishLoad(response(documents.qwen21.kr))
  await loading
  assert.equal(qwen.configMethod, 'loha')
  assert.equal(qwen.configProfile, '')
  assert.equal(qwen.draft, previousDraft)
  assert.equal(qwen.profile, 'default')
})
