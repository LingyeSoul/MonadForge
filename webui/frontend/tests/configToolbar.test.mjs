import assert from 'node:assert/strict'
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { execFileSync } from 'node:child_process'
import test from 'node:test'
import { parse, compileScript, compileStyle } from '@vue/compiler-sfc'
import { compile as compileSass } from 'sass'
import ts from 'typescript'

const asset = path => new URL(`../node_modules/${path}`, import.meta.url).href
const readSource = path => readFileSync(new URL(`../src/${path}`, import.meta.url), 'utf8')
const transpile = source => ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText
const compileComponent = (path, id) => {
  const { descriptor } = parse(readSource(path), { filename: path })
  const scopeId = `data-v-${id}`
  const script = transpile(compileScript(descriptor, { id: scopeId, inlineTemplate: true }).content)
  const css = descriptor.styles.map(style => {
    const result = compileStyle({ source: style.content, filename: path, id: scopeId, scoped: style.scoped })
    assert.deepEqual(result.errors, [])
    return result.code
  }).join('\n')
  return { script, css, scopeId }
}
const toolbar = compileComponent('components/ConfigToolbar.vue', 'toolbar-layout-test')
const runtimeSelector = compileComponent('components/RuntimeSelector.vue', 'runtime-selector-test')
const modelWorkspace = transpile(readSource('stores/modelWorkspace.ts'))
const sharedCss = compileSass(fileURLToPath(new URL('../src/styles/main.scss', import.meta.url))).css

// Layout contract for the real shared SFC, with real Vuetify controls and RuntimeSelector.
// Slot data intentionally varies by model and language; the editor's business interactions
// are validated on the running page, rather than copied into a second editor here.
// CHROMIUM_BIN=/usr/bin/chromium node --test tests/configToolbar.test.mjs
// The viewport stays wide while the containing panel changes, exercising container queries.
test('shared toolbar keeps model slots aligned and controls separate at each panel width', {
  skip: !process.env.CHROMIUM_BIN,
}, () => {
  const directory = mkdtempSync(join(tmpdir(), 'monadforge-toolbar-'))
  try {
    const html = `<!doctype html><html><head><meta charset="utf-8">
      <link rel="stylesheet" href="${asset('vuetify/dist/vuetify.css')}">
      <link rel="stylesheet" href="${asset('@mdi/font/css/materialdesignicons.css')}">
      <link rel="stylesheet" href="${asset('@fontsource/geist/latin-400.css')}">
      <link rel="stylesheet" href="${asset('@fontsource/geist/latin-500.css')}">
      <style>${sharedCss}\n${toolbar.css}\n${runtimeSelector.css}</style>
      </head><body><div id="app"></div><pre id="result">pending</pre>
      <script>const errors = []; window.onerror = message => { errors.push(message); document.querySelector('#result').textContent = JSON.stringify({errors}); };</script>
      <script src="${asset('vue/dist/vue.global.prod.js')}"></script>
      <script src="${asset('vuetify/dist/vuetify.js')}"></script>
      <script src="${asset('pinia/dist/pinia.iife.prod.js')}"></script>
      <script>
      function loadModule(source, dependencies) {
        const module = {exports: {}};
        new Function('require', 'module', 'exports', source)(name => {
          if (!(name in dependencies)) throw new Error('Unexpected fixture dependency: ' + name);
          return dependencies[name];
        }, module, module.exports);
        return module.exports;
      }
      const runtimeRequests = [];
      window.fetch = async url => {
        runtimeRequests.push(url);
        if (url !== '/api/system/runtimes') throw new Error('Unexpected fixture request: ' + url);
        return new Response(JSON.stringify({runtimes: ['default', 'curation', 'v100'].map(name => ({
          name, available: true, python: '/test/' + name + '/bin/python', error: null,
        }))}), {status: 200, headers: {'Content-Type': 'application/json'}});
      };
      const pinia = Pinia.createPinia();
      const workspaceModule = loadModule(${JSON.stringify(modelWorkspace)}, {vue: Vue, pinia: Pinia});
      const workspace = workspaceModule.useModelWorkspace(pinia);
      const appStore = Vue.reactive({language: 'en'});
      const ConfigToolbar = loadModule(${JSON.stringify(toolbar.script)}, {vue: Vue}).default;
      ConfigToolbar.__scopeId = ${JSON.stringify(toolbar.scopeId)};
      const RuntimeSelector = loadModule(${JSON.stringify(runtimeSelector.script)}, {
        vue: Vue, '../stores/app': {useAppStore: () => appStore}, '../stores/modelWorkspace': workspaceModule,
      }).default;
      const labels = {
        en: {model: 'Base model', method: 'Method', variant: 'Configuration profile', preset: 'Preset', task: 'Training task', empty: 'Empty: new training', train: 'Train', test: 'Test', automatic: 'Automatic sizing', defaults: 'MOSS parameter defaults', long: 'Very long custom configuration and training task name '},
        cn: {model: '基础模型', method: '训练方法', variant: '配置方案', preset: '硬件预设', task: '训练任务', empty: '留空：新建训练', train: '训练', test: '测试', automatic: '自动匹配硬件配置', defaults: 'MOSS 默认训练参数', long: '包含中文和英文的超长自定义配置方案与训练任务名称 '},
      };
      const app = Vue.createApp({
        components: {ConfigToolbar, RuntimeSelector},
        template: \`<v-app><ConfigToolbar>
          <template #model><v-select v-model="family" :items="modelItems" :label="copy.model" variant="outlined" density="compact" hide-details /></template>
          <template #runtime><RuntimeSelector :disabled="busy" /></template>
          <template #method><v-select :model-value="family === 'anima' ? 'lora' : 'LoRA'" :items="family === 'anima' ? ['lora', 'tlora'] : ['LoRA', 'LoKr', 'LoHa']" :label="copy.method" variant="outlined" density="compact" hide-details /></template>
          <template #variant>
            <v-select :model-value="copy.long.repeat(5)" :items="[copy.long.repeat(5)]" :label="family === 'anima' ? 'Variant' : copy.variant" variant="outlined" density="compact" hide-details />
            <v-btn icon="mdi-plus" aria-label="Create configuration" variant="text" density="compact" size="small" />
          </template>
          <template #preset>
            <template v-if="family === 'anima'">
              <v-select :model-value="copy.long.repeat(5)" :items="[copy.long.repeat(5)]" :label="copy.preset" variant="outlined" density="compact" hide-details />
              <v-btn icon="mdi-plus" aria-label="Create preset" variant="text" density="compact" size="small" />
              <v-btn v-if="customPreset" icon="mdi-delete-outline" aria-label="Delete preset" variant="text" density="compact" size="small" />
            </template>
            <v-text-field v-else :model-value="family === 'moss' ? copy.defaults : copy.automatic" :label="copy.preset" readonly variant="outlined" density="compact" hide-details />
          </template>
          <template #task><v-select :model-value="selectedTask ? 'task' : null" :items="[{title: copy.long.repeat(12), value: 'task'}]" :label="copy.task" :placeholder="copy.empty" :disabled="family !== 'anima'" clearable persistent-placeholder variant="outlined" density="compact" hide-details /></template>
          <template #actions>
            <v-btn icon="mdi-content-save-outline" aria-label="Save" variant="text" size="small" />
            <v-btn icon="mdi-refresh" aria-label="Reload" variant="text" size="small" />
            <v-btn prepend-icon="mdi-play" color="primary">{{ copy.train }}</v-btn>
            <v-btn prepend-icon="mdi-test-tube" variant="outlined">{{ copy.test }}</v-btn>
          </template>
        </ConfigToolbar></v-app>\`,
        data: () => ({
          appStore, family: 'anima', busy: false, customPreset: false, selectedTask: false,
          modelItems: [{title: 'Anima', value: 'anima'}, {title: 'Qwen-Image 2.1', value: 'qwen21'}, {title: 'MOSS', value: 'moss'}],
        }),
        computed: {copy() { return labels[this.appStore.language]; }},
      });
      app.config.errorHandler = error => errors.push(String(error));
      app.config.warnHandler = message => errors.push(message);
      const vm = app.use(pinia).use(Vuetify.createVuetify()).mount('#app');
      (async () => {
        const failures = [];
        const baseline = new Map();
        const slotNames = ['model', 'runtime', 'method', 'variant', 'preset', 'task', 'actions'];
        const intersects = (a, b) => Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1;
        let cases = 0;
        await workspace.loadRuntimes();
        await document.fonts.ready;
        for (const language of ['en', 'cn']) {
          vm.appStore.language = language;
          for (const populated of [false, true]) {
            vm.customPreset = populated;
            vm.selectedTask = populated;
            for (const family of ['anima', 'qwen21', 'moss']) {
              vm.family = family;
              workspace.selectRuntime(family === 'anima' ? 'default' : family === 'qwen21' ? 'curation' : 'v100');
              for (const width of [320, 390, 559, 560, 780, 899, 900, 1440]) {
                document.querySelector('#app').style.width = width + 'px';
                await Vue.nextTick();
                const context = {language, populated, family, width};
                const check = (condition, issue, detail = {}) => { if (!condition) failures.push({...context, issue, ...detail}); };
                const bar = document.querySelector('.config-toolbar');
                const bounds = bar.getBoundingClientRect();
                const slots = [...bar.children];
                check(JSON.stringify(slots.map(el => el.dataset.toolbarSlot)) === JSON.stringify(slotNames), 'slot structure changed');
                const rectangles = slots.map(el => el.getBoundingClientRect());
                const coordinates = rectangles.map(({left, top, width, height}) => [left, top, width, height]);
                const key = language + '-' + populated + '-' + width;
                if (family === 'anima') baseline.set(key, coordinates);
                else check(coordinates.every((values, index) => values.every((value, axis) => Math.abs(value - baseline.get(key)[index][axis]) <= 1)), 'model switch moved a slot', {coordinates, baseline: baseline.get(key)});
                for (const [index, rectangle] of rectangles.entries()) {
                  check(rectangle.left >= bounds.left - 1 && rectangle.right <= bounds.right + 1, 'slot overflows panel', {slot: slotNames[index]});
                  check(!rectangles.slice(index + 1).some(other => intersects(rectangle, other)), 'slots overlap', {slot: slotNames[index]});
                }
                const controls = [...bar.querySelectorAll('.v-input, .v-btn')].map(el => ({name: el.getAttribute('aria-label') || el.textContent.trim().slice(0, 30), rect: el.getBoundingClientRect()}));
                check(controls.length >= 11, 'controls missing', {count: controls.length});
                for (const [index, control] of controls.entries()) {
                  check(control.rect.left >= bounds.left - 1 && control.rect.right <= bounds.right + 1, 'control overflows panel', {control: control.name});
                  check(!controls.slice(index + 1).some(other => intersects(control.rect, other.rect)), 'controls overlap', {control: control.name});
                }
                const actions = slots[6];
                const actionButtons = [...actions.querySelectorAll('.v-btn')];
                check(actionButtons.length === 4, 'actions missing');
                for (const button of actionButtons) {
                  const rectangle = button.getBoundingClientRect();
                  check(rectangle.left >= rectangles[6].left - 1 && rectangle.right <= rectangles[6].right + 1 && rectangle.top >= rectangles[6].top - 1 && rectangle.bottom <= rectangles[6].bottom + 1, 'action escapes slot');
                }
                const expectedColumns = width >= 900 ? 4 : width >= 560 ? 2 : 1;
                check(getComputedStyle(bar).gridTemplateColumns.split(' ').length === expectedColumns, 'incorrect container breakpoint');
                check(Math.abs(rectangles[0].width - (bounds.width - 16 * (expectedColumns - 1)) / expectedColumns) <= 1, 'columns are not equal');
                check(getComputedStyle(actions).justifyContent === (width >= 560 ? 'flex-end' : 'flex-start'), 'actions alignment changed');
                check(bar.scrollWidth <= Math.ceil(bounds.width) + 1, 'toolbar scrolls horizontally');
                const runtimeControl = slots[1].querySelector('[data-testid="runtime-profile-select"]');
                check(runtimeControl?.classList.contains('v-input') && runtimeControl.textContent.includes(workspace.runtimeProfile), 'real runtime selector not rendered');
                cases++;
              }
            }
          }
        }
        vm.busy = true;
        await Vue.nextTick();
        if (!document.querySelector('[data-testid="runtime-profile-select"] input')?.disabled) failures.push({issue: 'runtime selector ignores disabled state'});
        document.querySelector('#result').textContent = JSON.stringify({cases, failures, errors, runtimeRequests});
      })().catch(e => {document.querySelector('#result').textContent = JSON.stringify({errors: [...errors, String(e)]})});
      </script></body></html>`
    const file = join(directory, 'toolbar.html')
    writeFileSync(file, html)
    const dom = execFileSync(process.env.CHROMIUM_BIN, [
      '--headless', '--no-sandbox', '--disable-gpu', '--allow-file-access-from-files',
      `--user-data-dir=${join(directory, 'profile')}`, '--window-size=1600,1000',
      '--dump-dom', '--virtual-time-budget=5000', pathToFileURL(file).href,
    ], {encoding: 'utf8', timeout: 20000, stdio: ['ignore', 'pipe', 'pipe'], maxBuffer: 8 * 1024 * 1024})
    const resultElement = dom.match(/<pre id="result">([^<]*)<\/pre>/)
    assert.ok(resultElement, 'Chromium did not return layout results')
    assert.notEqual(resultElement[1], 'pending', 'Layout checks did not finish')
    const result = JSON.parse(resultElement[1].replaceAll('&amp;', '&').replaceAll('&lt;', '<').replaceAll('&gt;', '>'))
    assert.deepEqual(result.errors, [], JSON.stringify(result))
    assert.equal(result.cases, 96, JSON.stringify(result))
    assert.deepEqual(result.failures, [])
    assert.deepEqual(result.runtimeRequests, ['/api/system/runtimes'])
  } finally {
    rmSync(directory, {recursive: true, force: true})
  }
})
