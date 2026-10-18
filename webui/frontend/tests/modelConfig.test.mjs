import assert from 'node:assert/strict'
import test from 'node:test'
import { BASIC_GROUP_ID, groupModelConfigFields, modelAlgorithmItems, modelActionValues, modelFieldActive } from '../src/utils/modelConfig.ts'

// Mirrors one field of the backend schema: `group` comes from
// `library.lines.requests.field_group`, `advanced` from the request metadata.
function field(name, value = '', choices = [], options = {}) {
  return { name, value, choices, kind: 'str', nullable: false, help: name, advanced: false, multiline: false, group: null, ...options }
}
const names = (values, options = {}) => values.split(' ').map(name => field(name, '', [], options))
const entries = groups => groups.flatMap(group => group.sections.flatMap(({ section, fields }) =>
  fields.map(field => ({ group: group.id, section, field }))))

test('basic fields open the page while advanced fields use the backend section', () => {
  const schema = {
    train: [
      ...names('model_dir dit text_encoder vae processor scheduler', { group: 'Paths', advanced: true }),
      ...names('cache output', { group: 'Paths' }),
      ...names('epochs max_steps lr warmup_ratio', { group: 'Training' }),
      ...names('rank alpha', { group: 'Architecture' }),
      field('logit_mean', 0, [], { group: 'Training', advanced: true }),
      ...names('targets factor down_init', { group: 'Architecture', advanced: true }),
      ...names('blocks_to_swap grad_checkpointing compile', { group: 'Performance', advanced: true }),
      field('algo', 'lokr', ['lora', 'lokr', 'loha'], { group: 'Architecture' }),
    ],
    cache: names('src overwrite'),
    generate: [
      ...names('prompt steps seed out_dir', { group: 'Other' }),
      field('example', '', [], { group: 'Other', advanced: true }),
      ...names('spectrum spectrum_w', { group: 'Performance', advanced: true }),
    ],
  }
  const groups = groupModelConfigFields(schema, { train: { algo: 'lokr', compile: true, grad_checkpointing: false, blocks_to_swap: null } })
  // Panels follow the Anima order; `basic` always leads.
  assert.deepEqual(groups.map(group => group.id), [BASIC_GROUP_ID, 'training', 'performance', 'paths', 'architecture', 'other'])
  const grouped = Object.fromEntries(groups.map(group => [group.id, group.sections.map(({ section, fields }) =>
    [section, fields.map(field => field.name)])]))
  // Everything not marked advanced — whatever section it belongs to — opens the page.
  assert.deepEqual(grouped.basic, [
    ['train', ['cache', 'output', 'epochs', 'max_steps', 'lr', 'warmup_ratio', 'rank', 'alpha']],
    ['generate', ['prompt', 'steps', 'seed', 'out_dir']],
  ])
  assert.deepEqual(grouped.training, [['train', ['logit_mean']]])
  assert.deepEqual(grouped.performance, [['train', ['blocks_to_swap', 'grad_checkpointing', 'compile']], ['generate', ['spectrum', 'spectrum_w']]])
  // `down_init` is lora-only and the draft selects lokr, so it stays stored but unrendered.
  assert.deepEqual(grouped.architecture, [['train', ['targets', 'factor']]])
  assert.deepEqual(grouped.other, [['generate', ['example']]])
  // Model paths stay in their own panel even though the section is advanced.
  assert.deepEqual(grouped.paths, [
    ['models', ['model_dir', 'dit', 'text_encoder', 'vae', 'processor', 'scheduler']],
  ])
  assert.equal(entries(groups).length, 12 + 1 + 5 + 6 + 2 + 1)
})

test('a section the backend adds later renders under its own name', () => {
  const schema = {
    train: [field('weird', 1, [], { group: 'Preview Sampling', advanced: true })],
    cache: [],
    generate: [],
  }
  const groups = groupModelConfigFields(schema)
  assert.deepEqual(groups.map(group => group.id), ['preview-sampling'])
  assert.equal(groups[0].titleKey, 'Preview Sampling')
})

test('advanced fields without a backend section fall back to the trailing panel', () => {
  const schema = { train: [field('orphan', 1, [], { advanced: true })], cache: [], generate: [] }
  const groups = groupModelConfigFields(schema)
  assert.deepEqual(groups.map(group => group.id), ['other'])
  assert.equal(groups[0].titleKey, 'cfgGroupOther')
})

test('future fields remain visible while irrelevant algorithm fields stay stored but unrendered', () => {
  const future = field('future_training_parameter', 'original')
  const factor = field('factor', -1, [], { kind: 'int', advanced: true, group: 'Architecture' })
  const downInit = field('down_init', 'weight_svd', ['kaiming', 'weight_svd'], { group: 'Architecture' })
  const generate = field('future_generator_parameter', 'generation value', [], { group: 'Other' })
  const schema = { train: [future, factor, downInit, field('algo', 'loha', [], { group: 'Architecture' })], cache: [], generate: [generate] }
  const before = structuredClone(schema)
  const groups = groupModelConfigFields(schema)
  const rows = entries(groups)
  // Only the future fields render: `factor` is gated on the lokr algorithm and
  // `down_init` on lora, so neither earns a panel in the loha fixture.
  assert.deepEqual(groups.map(group => group.id), [BASIC_GROUP_ID])
  assert.equal(rows.find(row => row.group === BASIC_GROUP_ID).field, future)
  assert.equal(rows.find(row => row.section === 'generate').field, generate)
  assert.equal(rows.some(row => row.field.name === 'factor'), false)
  assert.equal(rows.some(row => row.field.name === 'down_init'), false)
  assert.deepEqual(schema, before)
  assert.equal(new Set(rows.map(row => `${row.section}.${row.field.name}`)).size, rows.length)
})

for (const family of ['qwen21', 'moss']) {
  for (const algo of ['lora', 'lokr', 'loha']) {
    test(`${family}/${algo} exposes only consumed adapter parameters and retains hidden values`, () => {
      const schema = { train: [field('algo', 'lora', [], { group: 'Architecture' }), field('factor', -1, [], { group: 'Architecture' }), field('down_init', 'kaiming', [], { group: 'Architecture' }), ...names('rank alpha targets lora_dtype', { group: 'Architecture' })], cache: [], generate: [] }
      const draft = { train: { algo, factor: 7, down_init: 'weight_svd' } }
      const before = structuredClone(draft)
      const visible = entries(groupModelConfigFields(schema, draft, family)).map(row => row.field.name)
      assert.equal(visible.includes('factor'), algo === 'lokr')
      assert.equal(visible.includes('down_init'), algo === 'lora')
      for (const shared of ['rank', 'alpha', 'targets', 'lora_dtype']) assert.ok(visible.includes(shared))
      assert.deepEqual(draft, before)
      const payload = modelActionValues(family, 'train', schema.train, draft.train)
      assert.equal(payload.factor, algo === 'lokr' ? 7 : -1)
      assert.equal(payload.down_init, algo === 'lora' ? 'weight_svd' : 'kaiming')
      assert.deepEqual(draft, before)
    })
  }
}

test('performance controls track compile, activation-checkpointing, auto-swap and sigma mode', () => {
  for (const family of ['qwen21', 'moss']) {
    for (const compile of [true, false]) {
      assert.equal(modelFieldActive(family, 'train', 'compile_mode', { compile }), compile)
      assert.equal(modelFieldActive(family, 'train', 'compile_seq', { compile }), compile)
      for (const grad_checkpointing of [true, false]) {
        assert.equal(modelFieldActive(family, 'train', 'activation_memory_budget', { compile, grad_checkpointing }), compile && !grad_checkpointing)
      }
    }
  }
  for (const blocks_to_swap of [null, 0, 5]) assert.equal(modelFieldActive('qwen21', 'train', 'activation_reserve_gb', { blocks_to_swap }), blocks_to_swap === null)
  for (const sigma_uniform of [0, 0.5, 1]) for (const name of ['sigma_mu', 'sigma_s', 'sigma_cutoff']) {
    assert.equal(modelFieldActive('moss', 'train', name, { sigma_uniform }), sigma_uniform < 1)
  }
})

test('generation ignores no-adapter multipliers without changing the saved comparison strengths', () => {
  const values = { lora: '', multipliers: '0.75,0.0', true_cfg_scale: 1, negative_prompt: 'saved negative' }
  const fields = [field('multipliers', '1.0,0.0'), field('negative_prompt', '')]
  assert.equal(modelFieldActive('qwen21', 'generate', 'multipliers', values), false)
  assert.equal(modelFieldActive('qwen21', 'generate', 'negative_prompt', values), false)
  assert.deepEqual(modelActionValues('qwen21', 'generate', fields, values), { ...values, multipliers: '0.0', negative_prompt: '' })
  assert.equal(values.multipliers, '0.75,0.0')
  const enabled = { ...values, lora: 'adapter.safetensors', true_cfg_scale: 2 }
  assert.equal(modelFieldActive('qwen21', 'generate', 'multipliers', enabled), true)
  assert.equal(modelFieldActive('qwen21', 'generate', 'negative_prompt', enabled), true)
  assert.deepEqual(modelActionValues('qwen21', 'generate', fields, enabled), enabled)
})

test('MOSS generation keeps fallback CFG controls for examples but hides overwritten parameters', () => {
  for (const cfg of [0, 1, 3]) {
    const plain = { cfg, spectrum: false, lora: '' }
    for (const name of ['negative_prompt', 'uncond_skip', 'batched']) assert.equal(modelFieldActive('moss', 'generate', name, plain), cfg !== 1)
    const example = { ...plain, example: 'parameters.json' }
    for (const name of ['prompt', 'prompts_file', 'prompt_format', 'width', 'height', 'seed', 'negative_prompt']) assert.equal(modelFieldActive('moss', 'generate', name, example), false)
    for (const name of ['cfg', 'uncond_skip', 'steps', 'shift', 'batched']) assert.equal(modelFieldActive('moss', 'generate', name, example), true)
  }
  for (const sampler of ['euler', 'er_sde']) for (const spectrum of [true, false]) {
    assert.equal(modelFieldActive('moss', 'generate', 'spectrum_m', { sampler, spectrum }), spectrum)
  }
  for (const name of ['out_dir', 'model_dir', 'decode_only']) assert.equal(modelFieldActive('moss', 'generate', name, { decode_only: true }), true)
  for (const name of ['prompt', 'lora', 'model', 'steps', 'spectrum', 'example']) assert.equal(modelFieldActive('moss', 'generate', name, { decode_only: true }), false)
})

test('prompt sources, explicit dimensions and cache passes gate only their dependent fields', () => {
  for (const family of ['qwen21', 'moss']) {
    assert.equal(modelFieldActive(family, 'generate', 'prompt', { prompts_file: 'prompts.txt' }), false)
    assert.equal(modelFieldActive(family, 'generate', 'prompt', { prompts_file: '' }), true)
    assert.equal(modelFieldActive(family, 'cache', 'max_text_tokens', { skip_text: true }), false)
    assert.equal(modelFieldActive(family, 'cache', 'caption_format', { skip_text: true }), true)
    for (const name of ['resolution', 'vae_dtype', 'save_crops']) {
      assert.equal(modelFieldActive(family, 'cache', name, { skip_latents: true }), false)
      assert.equal(modelFieldActive(family, 'cache', name, { skip_latents: false }), true)
    }
  }
  assert.equal(modelFieldActive('qwen21', 'generate', 'resolution', { width: 512, height: 512 }), false)
  assert.equal(modelFieldActive('qwen21', 'generate', 'resolution', { width: null, height: null }), true)
})

test('same-name training and generation settings retain independent values and identities', () => {
  const trainSeed = field('seed', 1, [], { group: 'Training' }), generationSeed = field('seed', 42, [], { group: 'Other' })
  const trainModel = field('model', 'base', [], { group: 'Architecture' }), generationModel = field('model', 'sft', [], { group: 'Architecture' })
  const trainCompile = field('compile', false, [], { group: 'Performance', advanced: true }), generationCompile = field('compile', true, [], { group: 'Performance', advanced: true })
  const schema = { train: [trainSeed, trainModel, trainCompile], cache: [], generate: [generationSeed, generationModel, generationCompile] }
  const rows = entries(groupModelConfigFields(schema))
  const keyed = new Map(rows.map(row => [`${row.section}.${row.field.name}`, row.field]))
  assert.equal(keyed.get('train.seed'), trainSeed)
  assert.equal(keyed.get('generate.seed'), generationSeed)
  assert.equal(keyed.get('train.model'), trainModel)
  assert.equal(keyed.get('generate.model'), generationModel)
  assert.equal(keyed.get('train.compile'), trainCompile)
  assert.equal(keyed.get('generate.compile'), generationCompile)
  assert.equal(keyed.size, 6)
})

test('method choices come from the schema without replacing or normalizing API values', () => {
  const choices = ['loha', 'lora', 'lokr', 'custom_adapter', 'LoRA']
  const schema = { train: [field('algo', 'lora', choices, { group: 'Architecture' })], cache: [], generate: [] }
  assert.deepEqual(modelAlgorithmItems(schema), [
    { title: 'LoHa', value: 'loha' }, { title: 'LoRA', value: 'lora' }, { title: 'LoKr', value: 'lokr' },
    { title: 'custom_adapter', value: 'custom_adapter' }, { title: 'LoRA', value: 'LoRA' },
  ])
  assert.deepEqual(choices, ['loha', 'lora', 'lokr', 'custom_adapter', 'LoRA'])
  assert.deepEqual(modelAlgorithmItems({ train: [], cache: [], generate: [] }), [])
  assert.deepEqual(modelAlgorithmItems(null), [])
  assert.deepEqual(groupModelConfigFields(null), [])
})
