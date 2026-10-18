import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import * as vue from 'vue'
import * as pinia from 'pinia'
import ts from 'typescript'
function setup(t, runtimeError = false) {
  pinia.setActivePinia(pinia.createPinia())
  const exports = {}, requests = []
  const text = readFileSync(new URL('../src/stores/stagedResolution.ts', import.meta.url), 'utf8')
  const deps = { vue, pinia, '../utils/taskState.ts': { isTaskActive: () => false }, './task': { useTaskStore: () => ({ tasks: [], fetchTasks: async () => {}, fetchQueueStatus: async () => {} }) }, './modelWorkspace': { useModelWorkspace: () => ({ runtimeForSubmit: async () => { if (runtimeError) throw Error('V100 missing'); return 'v100' } }) } }
  new Function('require', 'exports', ts.transpileModule(text, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText)(name => deps[name], exports)
  t.mock.method(globalThis, 'fetch', async (url, init) => { requests.push({ url, init }); return new Response(JSON.stringify({ task_id: 'isolated' })) })
  const store = exports.useStagedResolutionStore(); store.loadedProfileName = 'default'; store.persisted = true
  return { store, requests }
}
for (const phase of ['train', 'preprocess']) test(`staged ${phase} submits the selected runtime`, async t => {
  const { store, requests } = setup(t)
  assert.equal(await store.start(phase), 'isolated')
  assert.deepEqual(JSON.parse(requests[0].init.body), { version: 1, runtime_profile: 'v100' })
})
test('unavailable runtime blocks staged submission without falling back', async t => {
  const { store, requests } = setup(t, true)
  assert.equal(await store.start('train'), null); assert.deepEqual(requests, []); assert.match(store.error, /V100 missing/)
})
