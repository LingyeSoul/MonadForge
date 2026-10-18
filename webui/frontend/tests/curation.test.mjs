import assert from 'node:assert/strict'
import test from 'node:test'
import { effectScope } from 'vue'
import { captionDiff, curationFileUrl, curationImageKey, curationFieldValue, normalizeCurationRoots, parseMaskSource, spellMaskSource, useCuration, visibleCurationFields } from '../src/composables/useCuration.ts'
import { curationCn, curationEn } from '../src/i18n/curation.ts'

const json = value => new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } })

test('mask sources round-trip Windows paths and literal prompt punctuation', () => {
  for (const spec of ['keep:soft:C:\\models\\subject.pt', 'ignore:text:caption: speech bubble']) assert.equal(spellMaskSource(parseMaskSource(spec)), spec)
})

test('schema bound paths cannot become editable overrides unless the server permits it', () => {
  const fields = [
    { dest: 'src', root: 'src' }, { dest: 'device', auto: true },
    { dest: 'from_report' }, { dest: 'apply' }, { dest: 'masks', kind: 'masks' },
    { dest: 'index', report: 'index.json', overridable: true }, { dest: 'checkpoint', setting: 'checkpoint' },
  ]
  assert.deepEqual(visibleCurationFields(fields).map(field => field.dest), ['masks', 'index'])
  assert.equal(curationFieldValue({ kind: 'int' }, ''), null)
  assert.equal(curationFieldValue({ kind: 'float' }, '0'), 0)
  assert.deepEqual(curationFieldValue({ kind: 'list' }, 'C:\\data a\n /tmp/b \n'), ['C:\\data a', '/tmp/b'])
})

test('literal caption diff preserves quoted commas and clause punctuation', () => {
  const before = 'safe, "Really, yes.". On the left, red hair'
  const after = 'safe, "Really, yes.". On the left, blue hair'
  const diff = captionDiff(before, after)
  assert.equal(diff.prefix + diff.removed + diff.suffix, before)
  assert.equal(diff.prefix + diff.added + diff.suffix, after)
  assert.equal(captionDiff('a', 'ab').added, 'b')
  assert.equal(captionDiff('same', 'same').removed, '')
})

test('file URLs preserve filesystem punctuation and Chinese paths', () => {
  const path = '/tmp/图片 & captions/a#1.png'
  assert.equal(new URL(curationFileUrl(path), 'http://localhost').searchParams.get('path'), path)
  assert.equal(new URL(curationFileUrl(path, true), 'http://localhost').searchParams.get('size'), '256')
  assert.equal(curationFileUrl(null), '')
  assert.deepEqual(normalizeCurationRoots({ src: '/data', dst: { path: '/work', exists: false } }).dst, { path: '/work', exists: false })
})

test('resized group members match source formats without merging directories or dotted stems', () => {
  const members = new Set(['artist.v1/photo.pose.png', 'other/photo.png'].map(curationImageKey))
  for (const rel of ['artist.v1/photo.pose.jpg', 'artist.v1/photo.pose.WEBP', 'artist.v1/photo.pose.png']) {
    assert.equal(members.has(curationImageKey(rel)), true, rel)
  }
  for (const rel of ['artist.v1/photo.jpg', 'different/photo.pose.jpg', 'artist.v1/nested/photo.pose.jpg']) {
    assert.equal(members.has(curationImageKey(rel)), false, rel)
  }
})

for (const action of ['edit', 'save', 'none']) {
  test(`background refresh rechecks editor state after a pending ${action}`, async t => {
    let complete
    t.mock.method(globalThis, 'fetch', () => new Promise(resolve => { complete = resolve }))
    const scope = effectScope()
    t.after(() => scope.stop())
    const api = scope.run(useCuration)
    api.item.value = { rel: 'photo.jpg', versions: [{ kind: 'master', text: 'original caption' }] }
    const original = api.item.value
    let dirty = false
    const refresh = api.selectItem('photo.jpg', () => !dirty && !api.busy.value)
    if (action === 'edit') dirty = true
    if (action === 'save') api.busy.value = true
    complete(json({ rel: 'photo.jpg', versions: [{ kind: 'master', text: 'server caption' }] }))
    await refresh
    if (action === 'none') assert.equal(api.item.value.versions[0].text, 'server caption')
    else assert.equal(api.item.value, original)
  })
}

test('older image requests cannot replace the newer selection', async t => {
  const pending = new Map()
  t.mock.method(globalThis, 'fetch', url => new Promise(resolve => pending.set(new URL(url, 'http://localhost').searchParams.get('rel'), resolve)))
  const scope = effectScope()
  const api = scope.run(useCuration)
  const first = api.selectItem('first.png')
  const second = api.selectItem('second.png')
  pending.get('second.png')(json({ rel: 'second.png' }))
  await second
  pending.get('first.png')(json({ rel: 'first.png' }))
  await first
  assert.equal(api.item.value.rel, 'second.png')
  scope.stop()
})

test('official composer receives clauses without client comma splitting', async t => {
  const calls = []
  t.mock.method(globalThis, 'fetch', async (url, options) => { calls.push({ url, options }); return json({ text: 'composed', parsed: { flat_tags: [], clauses: [], spans: [] } }) })
  const scope = effectScope()
  const api = scope.run(useCuration)
  const clauses = [{ header: 'Japanese text reads as', prefix: 'Japanese text reads as ', position: '', tags: ['"Really, yes."', '「大丈夫、本当に」'] }]
  const result = await api.compose(['safe', 'blue hair'], clauses)
  assert.equal(result.text, 'composed')
  assert.equal(calls[0].url, '/api/curation/compose')
  assert.equal(calls[0].options.method, 'POST')
  assert.deepEqual(JSON.parse(calls[0].options.body), { flat_tags: ['safe', 'blue hair'], clauses })
  scope.stop()
})

test('review replay uses saved job identity without resending current roots or stage values', async t => {
  const calls = []
  t.mock.method(globalThis, 'fetch', async (url, options) => { calls.push({ url, options }); return json(url.endsWith('/jobs') ? { jobs: [] } : { id: 'replay-id' }) })
  const scope = effectScope()
  const api = scope.run(useCuration)
  api.roots.value.src.path = '/different-current-source'
  assert.equal(await api.replay('review job/one', 'apply'), 'replay-id')
  assert.equal(calls[0].url, '/api/curation/jobs/review%20job%2Fone/apply')
  assert.deepEqual(JSON.parse(calls[0].options.body), {})
  scope.stop()
})

test('Chinese curation copy covers every English key', () => {
  assert.deepEqual(Object.keys(curationCn).sort(), Object.keys(curationEn).sort())
})
