import assert from 'node:assert/strict'
import test from 'node:test'
import { groupModelConfigFields, modelAlgorithmItems, modelActionValues, modelFieldActive } from '../src/utils/modelConfig.ts'

function field(name, value = '', choices = []) {
  return { name, value, choices, kind: 'str', nullable: false, help: name, advanced: false, multiline: false }
}
const names = values => values.split(' ').map(name => field(name))
const entries = groups => groups.flatMap(group => group.sections.flatMap(({ section, fields }) =>
  fields.map(field => ({ group: group.id, section, field }))))

test('model fields use the Anima group order while preserving each form section', () => {
  const schema = {
    train: [
      ...names('model_dir dit text_encoder vae processor scheduler cache output'),
      ...names('epochs max_steps rank alpha lr warmup_ratio save_every_epochs max_grad_norm lora_dtype seed logit_mean logit_std caption_dropout sigma_mu sigma_s sigma_cutoff sigma_uniform log_every'),
      ...names('blocks_to_swap activation_reserve_gb activation_memory_budget grad_checkpointing compile compile_seq compile_mode'),
      ...names('model targets factor down_init'),
      field('algo', 'lokr', ['lora', 'lokr', 'loha']),
    ],
    cache: names('src overwrite'),
    generate: names('model_dir dit text_encoder vae processor scheduler prompt model seed compile blocks_to_swap'),
  }
  const groups = groupModelConfigFields(schema, { train: { algo: 'lokr', compile: true, grad_checkpointing: false, blocks_to_swap: null } })
  assert.deepEqual(groups.map(group => group.id), ['training', 'performance', 'paths', 'architecture', 'generate'])
  const grouped = Object.fromEntries(groups.map(group => [group.id, group.sections.map(({ section, fields }) =>
    [section, fields.map(field => field.name)])]))
  assert.deepEqual(grouped.training, [['train', 'epochs max_steps rank alpha lr warmup_ratio save_every_epochs max_grad_norm lora_dtype seed logit_mean logit_std caption_dropout sigma_mu sigma_s sigma_cutoff sigma_uniform log_every'.split(' ')]])
  assert.deepEqual(grouped.performance, [['train', 'blocks_to_swap activation_reserve_gb activation_memory_budget grad_checkpointing compile compile_seq compile_mode'.split(' ')]])
  assert.deepEqual(grouped.paths, [
    ['models', ['model_dir', 'dit', 'text_encoder', 'vae', 'processor', 'scheduler']],
    ['train', ['cache', 'output']],
  ])
  assert.deepEqual(grouped.architecture, [['train', ['model', 'targets', 'factor']]])
  assert.deepEqual(grouped.generate, [['generate', ['prompt', 'model', 'seed', 'compile', 'blocks_to_swap']]])
  assert.equal(entries(groups).length, schema.train.length - 2 + schema.generate.length - 6)
})

test('future fields remain visible while irrelevant algorithm fields stay stored but unrendered', () => {
  const future = field('future_training_parameter', 'original')
  const factor = { ...field('factor', -1), kind: 'int', advanced: true }
  const downInit = field('down_init', 'weight_svd', ['kaiming', 'weight_svd'])
  const generate = field('future_generator_parameter', 'generation value')
  const schema = { train: [future, factor, downInit, field('algo', 'loha')], cache: [], generate: [generate] }
  const before = structuredClone(schema)
  const groups = groupModelConfigFields(schema)
  const rows = entries(groups)
  assert.deepEqual(groups.map(group => group.id), ['generate', 'other'])
  assert.equal(rows.find(row => row.group === 'other').field, future)
  assert.equal(rows.find(row => row.section === 'generate').field, generate)
  assert.equal(rows.some(row => row.field.name === 'factor'), false)
  assert.equal(rows.some(row => row.field.name === 'down_init'), false)
  assert.deepEqual(schema, before)
  assert.equal(new Set(rows.map(row => `${row.section}.${row.field.name}`)).size, rows.length)
})

for (const family of ['qwen21', 'moss']) {
  for (const algo of ['lora', 'lokr', 'loha']) {
    test(`${family}/${algo} exposes only consumed adapter parameters and retains hidden values`, () => {
      const schema = { train: [field('algo', 'lora'), field('factor', -1), field('down_init', 'kaiming'), ...names('rank alpha targets lora_dtype')], cache: [], generate: [] }
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
  const trainSeed = field('seed', 1), generationSeed = field('seed', 42)
  const trainModel = field('model', 'base'), generationModel = field('model', 'sft')
  const trainCompile = field('compile', false), generationCompile = field('compile', true)
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
  const schema = { train: [field('algo', 'lora', choices)], cache: [], generate: [] }
  assert.deepEqual(modelAlgorithmItems(schema), [
    { title: 'LoHa', value: 'loha' }, { title: 'LoRA', value: 'lora' }, { title: 'LoKr', value: 'lokr' },
    { title: 'custom_adapter', value: 'custom_adapter' }, { title: 'LoRA', value: 'LoRA' },
  ])
  assert.deepEqual(choices, ['loha', 'lora', 'lokr', 'custom_adapter', 'LoRA'])
  assert.deepEqual(modelAlgorithmItems({ train: [], cache: [], generate: [] }), [])
  assert.deepEqual(modelAlgorithmItems(null), [])
  assert.deepEqual(groupModelConfigFields(null), [])
})
