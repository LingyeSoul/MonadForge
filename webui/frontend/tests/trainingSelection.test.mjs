import assert from 'node:assert/strict'
import test from 'node:test'
import { createPinia, setActivePinia } from 'pinia'
import { useTrainingStore } from '../src/stores/training.ts'

const response = value => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } })
function deferred() {
  let resolve
  const promise = new Promise(done => { resolve = done })
  return { promise, resolve }
}

test('late metrics from the previous model cannot replace the selected task', async t => {
  setActivePinia(createPinia())
  const store = useTrainingStore()
  const previous = deferred()
  const requests = []
  t.mock.method(globalThis, 'fetch', async url => {
    requests.push(url)
    if (url === '/api/tasks/anima/metrics') return previous.promise
    if (url === '/api/tasks/moss/metrics') return response({ step: 20, avr_loss: 0.25 })
    return response({ items: [] })
  })
  const oldLoad = store.loadFromRest('anima')
  store.reset()
  await store.loadFromRest('moss')
  previous.resolve(response({ step: 999, avr_loss: 0.99 }))
  await oldLoad
  assert.equal(store.metrics.step, 20)
  assert.equal(store.metrics.avr_loss, 0.25)
  assert.ok(!requests.some(url => url.includes('/runs/anima/samples')))
})

test('late samples and disconnected streams do not leak into the next task gallery', async t => {
  setActivePinia(createPinia())
  const store = useTrainingStore()
  const samples = deferred()
  const requested = deferred()
  t.mock.method(globalThis, 'fetch', async url => {
    if (url.includes('/runs/anima/samples')) {
      requested.resolve()
      return samples.promise
    }
    return response({ step: 1 })
  })
  let current = true
  const oldLoad = store.loadFromRest('anima', () => current)
  await requested.promise
  current = false
  samples.resolve(response({ items: [{ path: '/old/anima.png', attempt_id: 'old' }] }))
  await oldLoad
  assert.deepEqual(store.metrics.sample_history, [])
  assert.equal(store.metrics.latest_sample, null)
})
