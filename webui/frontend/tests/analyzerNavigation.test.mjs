import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'
function setup(t) {
  const source = readFileSync(new URL('../../../scripts/run_analyzer/static/js/views/detail.js', import.meta.url), 'utf8')
  const nodes = new Map(), pending = new Map(), timers = new Set()
  const $ = key => { if (!nodes.has(key)) nodes.set(key, { innerHTML: '', textContent: '', classList: { add() {}, remove() {} } }); return nodes.get(key) }
  const deps = { '../core/dom.js': { $, $$: () => [], api: url => new Promise(resolve => pending.set(url, resolve)), esc: String, fmtTs: String, fmtNum: String, fmtDur: String, fmtLr: String }, '../theme.js': { ST: {} }, '../core/state.js': { saveState() {} }, '../components/states.js': { loadingHtml: () => '', emptyHtml: String, startTyping() {} }, '../components/kpi.js': { arcHtml: () => '' }, './curves.js': { renderCurves() {} }, './params.js': { renderParams() {} }, './samples.js': { renderSamples() {} }, './events.js': { renderEvents() {} } }
  const previous = globalThis.IntersectionObserver
  globalThis.IntersectionObserver = class { observe() {} disconnect() {} }
  t.after(() => { globalThis.IntersectionObserver = previous })
  t.mock.method(globalThis, 'setInterval', fn => { timers.add(fn); return fn })
  t.mock.method(globalThis, 'clearInterval', fn => timers.delete(fn))
  const exports = {}
  new Function('require', 'exports', ts.transpileModule(source, { compilerOptions: { allowJs: true, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText)(name => deps[name], exports)
  return { ...exports, pending, timers }
}
const payload = (id, state = 'done') => ({ id, run_name: id, state, kpis: {}, sources: {} })
test('analyzer detail keeps the most recently selected run when responses arrive out of order', async t => {
  const api = setup(t), first = api.loadDetail('A'), second = api.loadDetail('B')
  api.pending.get('/api/runs/B')(payload('B')); await second
  api.pending.get('/api/runs/A')(payload('A', 'running')); await first
  assert.equal(api.getDetail().id, 'B'); assert.equal(api.timers.size, 0)
})
test('leaving analyzer detail invalidates an in-flight initial load', async t => {
  const api = setup(t), pending = api.loadDetail('A')
  api.stopDetailPolling(); api.pending.get('/api/runs/A')(payload('A', 'running')); await pending
  assert.equal(api.getDetail(), null); assert.equal(api.timers.size, 0)
})
test('a previous run poll cannot replace the new detail or stop its polling timer', async t => {
  const api = setup(t), initial = api.loadDetail('A')
  api.pending.get('/api/runs/A')(payload('A', 'running')); await initial
  const oldPoll = [...api.timers][0]()
  const next = api.loadDetail('B')
  assert.equal(api.getDetail(), null, 'visibility refresh must not see the previous selected run while loading')
  api.pending.get('/api/runs/B')(payload('B', 'running')); await next
  api.pending.get('/api/runs/A')(payload('A')); await oldPoll
  assert.equal(api.getDetail().id, 'B'); assert.equal(api.timers.size, 1)
  api.stopDetailPolling()
})
