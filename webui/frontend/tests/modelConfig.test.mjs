import assert from 'node:assert/strict'
import test from 'node:test'
import { groupModelConfigFields, modelAlgorithmItems } from '../src/utils/modelConfig.ts'

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
  const groups = groupModelConfigFields(schema)
  assert.deepEqual(groups.map(group => group.id), ['training', 'performance', 'paths', 'architecture', 'generate'])
  const grouped = Object.fromEntries(groups.map(group => [group.id, group.sections.map(({ section, fields }) =>
    [section, fields.map(field => field.name)])]))
  assert.deepEqual(grouped.training, [['train', 'epochs max_steps rank alpha lr warmup_ratio save_every_epochs max_grad_norm lora_dtype seed logit_mean logit_std caption_dropout sigma_mu sigma_s sigma_cutoff sigma_uniform log_every'.split(' ')]])
  assert.deepEqual(grouped.performance, [['train', 'blocks_to_swap activation_reserve_gb activation_memory_budget grad_checkpointing compile compile_seq compile_mode'.split(' ')]])
  assert.deepEqual(grouped.paths, [
    ['models', ['model_dir', 'dit', 'text_encoder', 'vae', 'processor', 'scheduler']],
    ['train', ['cache', 'output']],
  ])
  assert.deepEqual(grouped.architecture, [['train', ['model', 'targets', 'factor', 'down_init']]])
  assert.deepEqual(grouped.generate, [['generate', ['prompt', 'model', 'seed', 'compile', 'blocks_to_swap']]])
  assert.equal(entries(groups).length, schema.train.length - 1 + schema.generate.length - 6)
})

test('future fields are retained, metadata is untouched, and algorithm-specific fields are not dropped', () => {
  const future = field('future_training_parameter', 'original')
  const factor = { ...field('factor', -1), kind: 'int', advanced: true }
  const downInit = field('down_init', 'weight_svd', ['kaiming', 'weight_svd'])
  const generate = field('future_generator_parameter', 'generation value')
  const schema = { train: [future, factor, downInit, field('algo', 'loha')], cache: [], generate: [generate] }
  const before = structuredClone(schema)
  const groups = groupModelConfigFields(schema)
  const rows = entries(groups)
  assert.deepEqual(groups.map(group => group.id), ['architecture', 'generate', 'other'])
  assert.equal(rows.find(row => row.group === 'other').field, future)
  assert.equal(rows.find(row => row.section === 'generate').field, generate)
  assert.equal(rows.find(row => row.field.name === 'factor').field, factor)
  assert.equal(rows.find(row => row.field.name === 'down_init').field, downInit)
  assert.deepEqual(schema, before)
  assert.equal(new Set(rows.map(row => `${row.section}.${row.field.name}`)).size, rows.length)
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
