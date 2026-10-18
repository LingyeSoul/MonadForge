import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import * as vue from 'vue'
import ts from 'typescript'
const response = value => new Response(JSON.stringify(value))
function load(path, deps) {
  const exports = {}
  const code = ts.transpileModule(readFileSync(new URL(`../src/${path}`, import.meta.url), 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
  new Function('require', 'exports', code)(name => { assert.ok(name in deps, name); return deps[name] }, exports)
  return exports
}
const states = load('utils/taskState.ts', {}), logs = load('utils/logLines.ts', {})
const { useTaskStream } = load('composables/useTask.ts', { vue: { ...vue, onUnmounted() {} }, '../utils/taskState.ts': states, '../utils/logLines': logs })
const flush = () => new Promise(resolve => setImmediate(resolve))
function setup(t) {
  const task = vue.reactive({ id: 'logical-run', attempt: 'attempt-1' }), sockets = []
  for (const [name, value] of Object.entries({ location: { protocol: 'http:', host: 'fixture' }, window: globalThis, WebSocket: class { constructor(url) { this.url = url; sockets.push(this) } close() {} } })) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, name)
    Object.defineProperty(globalThis, name, { configurable: true, value })
    t.after(() => { if (previous) Object.defineProperty(globalThis, name, previous); else delete globalThis[name] })
  }
  const scope = vue.effectScope()
  let stream
  t.after(() => { stream?.disconnect(); scope.stop() })
  return { task, sockets, connect: () => stream = scope.run(() => useTaskStream(() => task.id, () => task.attempt)) }
}
test('a resumed attempt with the same logical task ID reconnects the terminal log stream', async t => {
  const { task, sockets, connect } = setup(t)
  let requests = 0
  t.mock.method(globalThis, 'fetch', async () => { requests++; return response({ lines: [task.attempt], state: task.attempt === 'attempt-1' ? 'failed' : 'running', exit_code: 1 }) })
  const stream = connect(); await flush()
  assert.equal(stream.done.value, true); assert.equal(sockets.length, 0)
  task.attempt = 'attempt-2'; await vue.nextTick(); await flush()
  assert.equal(stream.done.value, false); assert.equal(requests, 2); assert.equal(sockets.length, 1)
  assert.deepEqual(stream.lines.value.map(line => line.text), ['attempt-2'])
})
test('a late terminal response from a previous attempt cannot stop its replacement stream', async t => {
  const { task, sockets, connect } = setup(t)
  let release
  t.mock.method(globalThis, 'fetch', async () => task.attempt === 'attempt-1' ? new Promise(resolve => { release = resolve }) : response({ lines: ['new attempt'], state: 'running' }))
  const stream = connect(); task.attempt = 'attempt-2'; await vue.nextTick(); await flush()
  release(response({ lines: ['stale failed attempt'], state: 'failed', exit_code: 1 })); await flush()
  assert.equal(stream.done.value, false); assert.equal(sockets.length, 1)
  assert.deepEqual(stream.lines.value.map(line => line.text), ['new attempt'])
})
