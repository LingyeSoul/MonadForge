import assert from 'node:assert/strict'
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { execFileSync } from 'node:child_process'
import test from 'node:test'

const source = readFileSync(new URL('../src/views/ConfigEditor.vue', import.meta.url), 'utf8')
const toolbar = source.slice(source.indexOf('<div class="config-toolbar">'), source.indexOf('    <QwenConfigFields'))
const css = source.match(/<style scoped>([\s\S]*?)<\/style>/)[1].replaceAll(/:deep\(([^)]+)\)/g, '$1')
const asset = path => new URL(`../node_modules/${path}`, import.meta.url).href

// Opt-in real layout check, using the project's Vue/Vuetify and a local Chromium.
// CHROMIUM_BIN=/usr/bin/chromium node --test tests/configToolbar.test.mjs
// Keep the actual toolbar template so added controls participate in this check.
test('toolbar controls do not overlap when sidebars reduce available width', {
  skip: !process.env.CHROMIUM_BIN,
}, () => {
  const directory = mkdtempSync(join(tmpdir(), 'monadforge-toolbar-'))
  try {
    const html = `<!doctype html><html><head><meta charset="utf-8">
      <link rel="stylesheet" href="${asset('vuetify/dist/vuetify.css')}">
      <style>${css}</style></head><body><div id="app"></div><pre id="result">pending</pre>
      <script>const errors = []; window.onerror = message => { errors.push(message); document.querySelector('#result').textContent = JSON.stringify({errors}); };</script>
      <script src="${asset('vue/dist/vue.global.prod.js')}"></script>
      <script src="${asset('vuetify/dist/vuetify.js')}"></script>
      <script>
      const noop = () => {};
      const vm = Vue.createApp({
        template: ${JSON.stringify(toolbar)},
        data: () => ({
          workspace: {family: 'anima'}, modelItems: ['anima', 'qwen21'],
          trainingLaunching: false, testLaunching: false,
          qwen: {busy: false, profiles: ['default'], profile: 'default', draft: {}, perform: noop},
          continuation: {active: false, presetItems: ['default', 'custom'], candidates: [], select: noop, fetchCandidates: noop},
          configStore: {methods: ['lora'], dirty: false, loading: false},
          selectedMethod: 'lora', selectedVariant: 'tlora', variantItems: ['tlora'],
          selectedPreset: 'custom', isCustomPreset: true, extraArgs: '',
          trainingTaskItems: [{title: 'Very long training task '.repeat(16), value: 'task'}],
          profileDialog: false,
        }),
        methods: Object.fromEntries(['requestModelChange', 'onMethodChange', 'onVariantChange',
          'openCreateVariant', 'openCreatePreset', 'deleteCurrentPreset', 'onSave', 'loadConfig',
          'startTraining', 'startTest', 'requestProfileChange', 'startQwenTraining', 'startQwenTest']
          .map(name => [name, noop]).concat([['t', key => key]])),
      }).use(Vuetify.createVuetify()).mount('#app');
      (async () => {
        const failures = [];
        let cases = 0;
        for (const family of ['anima', 'qwen21']) {
          vm.workspace.family = family;
          for (const selected of [undefined, 'task']) {
            vm.continuation.selected = selected;
            for (const width of [320, 420, 600, 780, 960, 1100, 1280, 1440]) {
              document.querySelector('#app').style.width = width + 'px';
              await Vue.nextTick();
              const bar = document.querySelector('.config-toolbar');
              const bounds = bar.getBoundingClientRect();
              const controls = [...bar.querySelectorAll('.v-input, .v-btn')]
                .map(el => el.getBoundingClientRect()).filter(r => r.width && r.height);
              if (controls.length < 8) failures.push({family, width, issue: 'missing controls', count: controls.length});
              for (const [i, a] of controls.entries()) {
                if (a.left < bounds.left - 1 || a.right > bounds.right + 1)
                  failures.push({family, selected, width, issue: 'overflow', i});
                for (const b of controls.slice(i + 1)) {
                  if (Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 &&
                      Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1)
                    failures.push({family, selected, width, issue: 'overlap', i});
                }
              }
              cases++;
            }
          }
        }
        document.querySelector('#result').textContent = JSON.stringify({cases, failures});
      })().catch(e => {document.querySelector('#result').textContent = String(e)});
      </script></body></html>`
    const file = join(directory, 'toolbar.html')
    writeFileSync(file, html)
    const dom = execFileSync(process.env.CHROMIUM_BIN, [
      '--headless', '--no-sandbox', '--disable-gpu', '--allow-file-access-from-files',
      `--user-data-dir=${join(directory, 'profile')}`, '--window-size=1600,1000',
      '--dump-dom', '--virtual-time-budget=3000', pathToFileURL(file).href,
    ], {encoding: 'utf8', timeout: 20000, stdio: ['ignore', 'pipe', 'pipe']})
    const result = JSON.parse(dom.match(/<pre id="result">([^<]*)<\/pre>/)[1])
    assert.equal(result.cases, 32, JSON.stringify(result))
    assert.deepEqual(result.failures, [])
  } finally {
    rmSync(directory, {recursive: true, force: true})
  }
})
