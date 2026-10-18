import { modelFields, type Phase, type QwenDraft, type QwenField, type QwenSchema, type Values } from './qwen21.ts'

export type ModelConfigSection = 'models' | 'train' | 'generate'
export interface ModelConfigGroup {
  id: string
  titleKey: string
  sections: { section: ModelConfigSection; fields: QwenField[] }[]
}

export type TypedModelFamily = 'qwen21' | 'moss'

/**
 * The everyday-knob card every model page opens with, mirroring Anima's
 * 「基本设置」. A field lands here when it is NOT advanced (Anima's `is_basic`,
 * the model lines' `advanced=False`), so the browser card, the desktop GUI's
 * Advanced fold and the request defaults all describe the same split.
 */
export const BASIC_GROUP_ID = 'basic'

// Panel order for the advanced knobs. The section of a field comes from the
// backend (`library.lines.requests.field_group`, shared with the desktop GUI);
// only the heading and the order are the browser's business. A section added
// later still renders — after these, under its own name.
const PANEL_ORDER = ['training', 'performance', 'paths', 'architecture', 'generate', 'other'] as const
const PANEL_TITLE_KEYS: Record<string, string> = {
  training: 'cfgGroupTraining',
  performance: 'cfgGroupPerformance',
  paths: 'cfgGroupPaths',
  architecture: 'cfgGroupArchitecture',
  generate: 'qwTestParameters',
  other: 'cfgGroupOther',
}

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
    { id: BASIC_GROUP_ID, titleKey: 'cfgBasicSettings', sections: [] },
    ...PANEL_ORDER.map(id => ({ id, titleKey: PANEL_TITLE_KEYS[id], sections: [] })),
  ]
  const byId = new Map(groups.map(group => [group.id, group]))
  function append(id: string, section: ModelConfigSection, field: QwenField): void {
    let group = byId.get(id)
    if (!group) {
      // A section the backend introduced after this list was written: keep it,
      // under its own name until a translation key exists for it.
      group = { id, titleKey: field.group ?? id, sections: [] }
      byId.set(id, group)
      groups.push(group)
    }
    let entry = group.sections.find(entry => entry.section === section)
    if (!entry) {
      entry = { section, fields: [] }
      group.sections.push(entry)
    }
    entry.fields.push(field)
  }
  /** Basic fields open the page; advanced ones go to the section the backend named. */
  function target(field: QwenField): string {
    if (!field.advanced) return BASIC_GROUP_ID
    const name = (field.group ?? '').trim()
    return name ? name.toLowerCase().replace(/[^a-z0-9]+/g, '-') : 'other'
  }
  for (const field of schema.train) {
    if (modelFields.has(field.name)) {
      append(target(field), 'models', field)
      continue
    }
    // The selector owns algo; irrelevant knobs must not reappear through search/help.
    if (field.name === 'algo' || !modelFieldActive(family, 'train', field.name, modelSectionValues(schema, draft, 'train'))) continue
    append(target(field), 'train', field)
  }
  for (const field of schema.generate) {
    if (!modelFields.has(field.name) && modelFieldActive(family, 'generate', field.name, modelSectionValues(schema, draft, 'generate'))) append(target(field), 'generate', field)
  }
  return groups.filter(group => group.sections.length > 0)
}

/** Preserve the API values and ordering; only familiar display names change. */
export function modelAlgorithmItems(schema: QwenSchema | null): { title: string; value: string }[] {
  const titles: Record<string, string> = { lora: 'LoRA', lokr: 'LoKr', loha: 'LoHa' }
  return (schema?.train.find(field => field.name === 'algo')?.choices ?? [])
    .map(value => ({ title: titles[value] ?? value, value }))
}
