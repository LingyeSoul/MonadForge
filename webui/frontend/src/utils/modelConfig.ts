import { modelFields, type QwenField, type QwenSchema } from './qwen21.ts'

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

/** Organize the typed workspaces without merging their storage sections. */
export function groupModelConfigFields(schema: QwenSchema | null): ModelConfigGroup[] {
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
    // The method selector owns this field; retain every other schema entry.
    if (field.name === 'algo') continue
    const id = trainingFields.has(field.name) ? 'training'
      : performanceFields.has(field.name) ? 'performance'
        : trainingPathFields.has(field.name) ? 'paths'
          : architectureFields.has(field.name) ? 'architecture' : 'other'
    append(id, 'train', field)
  }
  for (const field of schema.generate) {
    if (!modelFields.has(field.name)) append('generate', 'generate', field)
  }
  return groups.filter(group => group.sections.length > 0)
}

/** Preserve the API values and ordering; only familiar display names change. */
export function modelAlgorithmItems(schema: QwenSchema | null): { title: string; value: string }[] {
  const titles: Record<string, string> = { lora: 'LoRA', lokr: 'LoKr', loha: 'LoHa' }
  return (schema?.train.find(field => field.name === 'algo')?.choices ?? [])
    .map(value => ({ title: titles[value] ?? value, value }))
}
