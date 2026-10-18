import { modelFields, type Phase, type QwenDraft, type QwenField, type QwenSchema, type Values } from './qwen21.ts'

export type ModelConfigSection = 'models' | 'train' | 'generate'
export interface ModelConfigGroup {
  id: string
  titleKey: string
  sections: { section: ModelConfigSection; fields: QwenField[] }[]
}

const trainingFields = new Set([
  'epochs', 'max_steps', 'rank', 'alpha', 'lr', 'warmup_ratio',
  'save_every_epochs', 'max_grad_norm', 'lora_dtype', 'seed',
  'logit_mean', 'logit_std', 'caption_dropout', 'sigma_mu', 'sigma_s',
  'sigma_cutoff', 'sigma_uniform', 'log_every',
])
const performanceFields = new Set([
  'blocks_to_swap', 'activation_reserve_gb', 'activation_memory_budget',
  'grad_checkpointing', 'compile', 'compile_seq', 'compile_mode',
])
const architectureFields = new Set(['model', 'targets', 'factor', 'down_init'])
const trainingPathFields = new Set(['cache', 'output'])

export type TypedModelFamily = 'qwen21' | 'moss'

/** Match the actual request consumers, not whether a field happens to be in the schema. */
export function modelFieldActive(family: TypedModelFamily, section: string, name: string, values: Values): boolean {
  const supplied = (key: string) => typeof values[key] === 'string' && String(values[key]).trim().length > 0
  if (section === 'train') {
    const algorithm = String(values.algo ?? 'lora')
    if (name === 'factor') return algorithm === 'lokr'
    if (name === 'down_init') return algorithm === 'lora'
    if (name === 'compile_seq' || name === 'compile_mode') return values.compile === true
    if (name === 'activation_reserve_gb') return values.blocks_to_swap == null
    if (name === 'activation_memory_budget') return values.compile === true && values.grad_checkpointing !== true
    if (family === 'moss' && ['sigma_mu', 'sigma_s', 'sigma_cutoff'].includes(name)) return Number(values.sigma_uniform ?? 0) < 1
  }
  if (section === 'cache') {
    if (['max_text_tokens', 'te_blocks_to_swap', 'text_encoder'].includes(name)) return values.skip_text !== true
    if (['resolution', 'vae_dtype', 'save_crops', 'vae'].includes(name)) return values.skip_latents !== true
  }
  if (section === 'generate') {
    if (family === 'moss' && values.decode_only === true) return ['out_dir', 'model_dir', 'decode_only'].includes(name)
    if (name === 'lora_multiplier' || name === 'multipliers') return supplied('lora')
    if (family === 'qwen21') {
      if (name === 'prompt') return !supplied('prompts_file')
      if (name === 'negative_prompt') return Number(values.true_cfg_scale ?? 1) > 1
      if (name === 'resolution') return values.width == null || values.height == null
    } else {
      const example = supplied('example')
      if (name === 'prompt') return !example && !supplied('prompts_file')
      if (['prompts_file', 'prompt_format', 'width', 'height', 'seed'].includes(name)) return !example
      if (name === 'negative_prompt') return !example && Number(values.cfg ?? 3) !== 1
      if (name === 'uncond_skip' || name === 'batched') return example || Number(values.cfg ?? 3) !== 1
      if (name.startsWith('spectrum_')) return values.spectrum === true
    }
  }
  return true
}

export function modelSectionValues(schema: QwenSchema | null, draft: Partial<QwenDraft> | null, section: ModelConfigSection | 'cache'): Values {
  if (section === 'models') return draft?.models ?? {}
  return { ...Object.fromEntries((schema?.[section] ?? []).map(field => [field.name, field.value])), ...draft?.[section] }
}

/** Ignore inactive values only in action payloads; saved profiles retain them for later reuse. */
export function modelActionValues(family: TypedModelFamily, phase: Phase, fields: QwenField[], values: Values): Values {
  const active = { ...values }
  for (const field of fields) {
    if (!modelFieldActive(family, phase, field.name, values)) active[field.name] = field.value
  }
  if (family === 'qwen21' && phase === 'generate' && !String(values.lora ?? '').trim()) {
    active.multipliers = '0.0'
  }
  return active
}

/** Organize the typed workspaces without merging their storage sections. */
export function groupModelConfigFields(schema: QwenSchema | null, draft: Partial<QwenDraft> | null = null, family: TypedModelFamily = 'qwen21'): ModelConfigGroup[] {
  if (!schema) return []
  const groups: ModelConfigGroup[] = [
    { id: 'training', titleKey: 'cfgGroupTraining', sections: [] },
    { id: 'performance', titleKey: 'cfgGroupPerformance', sections: [] },
    { id: 'paths', titleKey: 'cfgGroupPaths', sections: [] },
    { id: 'architecture', titleKey: 'cfgGroupArchitecture', sections: [] },
    { id: 'generate', titleKey: 'qwTestParameters', sections: [] },
    { id: 'other', titleKey: 'cfgGroupOther', sections: [] },
  ]
  const byId = new Map(groups.map(group => [group.id, group]))
  function append(id: string, section: ModelConfigSection, field: QwenField): void {
    const group = byId.get(id)!
    let entry = group.sections.find(entry => entry.section === section)
    if (!entry) {
      entry = { section, fields: [] }
      group.sections.push(entry)
    }
    entry.fields.push(field)
  }
  for (const field of schema.train) {
    if (modelFields.has(field.name)) {
      append('paths', 'models', field)
      continue
    }
    // The selector owns algo; irrelevant knobs must not reappear through search/help.
    if (field.name === 'algo' || !modelFieldActive(family, 'train', field.name, modelSectionValues(schema, draft, 'train'))) continue
    const id = trainingFields.has(field.name) ? 'training'
      : performanceFields.has(field.name) ? 'performance'
        : trainingPathFields.has(field.name) ? 'paths'
          : architectureFields.has(field.name) ? 'architecture' : 'other'
    append(id, 'train', field)
  }
  for (const field of schema.generate) {
    if (!modelFields.has(field.name) && modelFieldActive(family, 'generate', field.name, modelSectionValues(schema, draft, 'generate'))) append('generate', 'generate', field)
  }
  return groups.filter(group => group.sections.length > 0)
}

/** Preserve the API values and ordering; only familiar display names change. */
export function modelAlgorithmItems(schema: QwenSchema | null): { title: string; value: string }[] {
  const titles: Record<string, string> = { lora: 'LoRA', lokr: 'LoKr', loha: 'LoHa' }
  return (schema?.train.find(field => field.name === 'algo')?.choices ?? [])
    .map(value => ({ title: titles[value] ?? value, value }))
}
