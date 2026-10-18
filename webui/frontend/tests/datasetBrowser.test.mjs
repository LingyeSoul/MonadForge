import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import test from 'node:test'
import { parse, compileScript } from '@vue/compiler-sfc'
import * as vue from 'vue'
import ts from 'typescript'

const filename = fileURLToPath(new URL('../src/views/DatasetBrowser.vue', import.meta.url))
const { descriptor } = parse(readFileSync(filename, 'utf8'), { filename })
const compiled = compileScript(descriptor, { id: 'dataset-browser-regression' })
const { outputText } = ts.transpileModule(compiled.content, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
})

function setup(t) {
  const messages = [], exports = {}, scope = vue.effectScope()
  t.after(() => scope.stop())
  const storageDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: { getItem: () => null } })
  t.after(() => {
    if (storageDescriptor) Object.defineProperty(globalThis, 'localStorage', storageDescriptor)
    else delete globalThis.localStorage
  })
  const require = name => {
    if (name === 'vue') return { ...vue, onMounted() {}, onUnmounted() {} }
    if (name === 'vue-router') return { onBeforeRouteLeave() {} }
    if (name === '../stores/app') return { useAppStore: () => ({ language: 'en' }) }
    if (name === '../stores/notify') return { useNotifyStore: () => ({ show: (...args) => messages.push(args) }) }
    if (name === '../composables/useI18n') return { useI18n: () => ({ t: (key, args) => `${key}${args ? JSON.stringify(args) : ''}` }) }
    if (name === '../i18n/curation') return { curationText: () => 'curation' }
    if (name.endsWith('.vue')) return { default: {} }
    throw new Error(`Unexpected import: ${name}`)
  }
  new Function('require', 'exports', outputText)(require, exports)
  const state = scope.run(() => exports.default.setup({}, { expose() {} }))
  state.images.value = [
    { path: 'one.png', filename: 'one.png', caption: 'original' },
    { path: 'two.png', filename: 'two.png', caption: 'second' },
  ]
  return { state, messages }
}

for (const failure of ['http', 'network']) {
  for (const action of ['saveCaptionAndMaybeAdvance', 'saveAndClose', 'arrow']) {
    test(`${failure} failure preserves caption draft through ${action}`, async t => {
      const { state, messages } = setup(t)
      t.mock.method(globalThis, 'fetch', async () => {
        if (failure === 'network') throw new Error('offline')
        return new Response(JSON.stringify({ detail: 'storage unavailable' }), { status: 503 })
      })
      state.selectImage(state.images.value[0])
      state.editCaption.value = 'important unsaved draft'
      if (action === 'saveAndClose') state.tryCloseEditor()
      if (action === 'arrow') {
        await state._onKeyDown({ key: 'ArrowRight', target: { tagName: 'DIV' }, preventDefault() {} })
      } else await state[action]()
      assert.equal(state.selectedImage.value.path, 'one.png')
      assert.equal(state.editCaption.value, 'important unsaved draft')
      assert.equal(state.diskCaption.value, 'original')
      assert.equal(state.editorDialog.value, true)
      assert.equal(state.isDirty.value, true)
      assert.equal(state.saving.value, false)
      assert.equal(messages.at(-1)[1], 'error')
      if (action === 'saveAndClose') assert.equal(state.discardDialog.value, true)
    })
  }
}

test('successful caption save advances and records only the saved caption', async t => {
  const { state } = setup(t)
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ content: 'saved caption' })))
  state.selectImage(state.images.value[0])
  state.editCaption.value = 'saved caption'
  await state.saveCaptionAndMaybeAdvance()
  assert.equal(state.images.value[0].caption, 'saved caption')
  assert.equal(state.selectedImage.value.path, 'two.png')
  assert.equal(state.editCaption.value, 'second')
  assert.equal(state.isDirty.value, false)
})

test('stale image responses cannot overwrite a newer search result or loading state', async t => {
  const { state } = setup(t), resolve = []
  t.mock.method(globalThis, 'fetch', () => new Promise(done => resolve.push(done)))
  const first = state.loadImages(), second = state.loadImages()
  resolve[1](new Response(JSON.stringify({ items: [{ path: 'new.png' }], total: 1, pages: 1 })))
  await second
  resolve[0](new Response(JSON.stringify({ items: [{ path: 'stale.png' }], total: 1, pages: 1 })))
  await first
  assert.equal(state.images.value[0].path, 'new.png')
  assert.equal(state.loadingImages.value, false)
})

test('image selection mode toggles selection instead of opening the editor', t => {
  const { state } = setup(t)
  state.selectionMode.value = true
  state.activateImage(state.images.value[0])
  assert.deepEqual([...state.selectedPaths.value], ['one.png'])
  assert.equal(state.editorDialog.value, false)
  state.activateImage(state.images.value[0])
  assert.equal(state.selectedCount.value, 0)
})

test('route leave cancellation preserves the caption and discard releases navigation', async t => {
  const { state } = setup(t)
  state.selectImage(state.images.value[0])
  state.editCaption.value = 'unsaved'
  const canceled = state.requestEditorLeave()
  state.cancelDiscard()
  assert.equal(await canceled, false)
  assert.equal(state.editCaption.value, 'unsaved')
  const discarded = state.requestEditorLeave()
  state.discardAndClose()
  assert.equal(await discarded, true)
  assert.equal(state.editorDialog.value, false)
})

test('single-file archive selection, clearing, and reopening match Vuetify file values', t => {
  const { state } = setup(t)
  const file = new File(['fixture'], 'training-set.tar.gz', { type: 'application/gzip' })
  state.onUploadFileChange(file)
  assert.equal(state.uploadFile.value, file)
  assert.equal(state.uploadTarget.value, 'datasets/training-set')
  state.onUploadFileChange(undefined)
  assert.equal(state.uploadFile.value, null)
  state.openUploadDialog()
  assert.equal(state.uploadTarget.value, '')
  assert.equal(state.uploadFile.value, null)
})
