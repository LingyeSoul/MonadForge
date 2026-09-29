/** Runtime-checked Qwen form transport; defaults are owned by the Python requests. */
export type Scalar = string | number | boolean | null
export type Values = Record<string, Scalar>
export type Phase = 'cache' | 'train' | 'generate'
type Json = Scalar | Json[] | { [key: string]: Json }
type JsonObject = { [key: string]: Json }
export interface QwenField {
  name: string
  kind: 'str' | 'int' | 'float' | 'bool'
  value: Scalar
  nullable: boolean
  choices: string[]
  help: string
  advanced: boolean
  multiline: boolean
}
export interface QwenSchema {
  cache: QwenField[]
  train: QwenField[]
  generate: QwenField[]
}
export interface QwenDraft {
  models: Values
  cache: Values
  train: Values
  generate: Values
}
export interface QwenResult {
  file: string
  multiplier: number
  seed: number
  prompt: string
}
export const modelFields = new Set(['model_dir', 'dit', 'text_encoder', 'vae', 'processor', 'scheduler'])
export const draftKey = 'monadforge-qwen21-form-v1'

function object(value: Json, label: string): JsonObject {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label}: expected an object`)
  return value
}
function string(value: Json, label: string): string {
  if (typeof value !== 'string') throw new Error(`${label}: expected a string`)
  return value
}
function bool(value: Json, label: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${label}: expected a boolean`)
  return value
}
function number(value: Json, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${label}: expected a finite number`)
  return value
}
function scalar(value: Json, label: string): Scalar {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value
  return number(value, label)
}
function field(value: Json): QwenField {
  const row = object(value, 'Qwen field')
  const kind = row.kind
  if (kind !== 'str' && kind !== 'int' && kind !== 'float' && kind !== 'bool') throw new Error('Invalid Qwen field kind')
  if (!Array.isArray(row.choices)) throw new Error('Missing Qwen choices')
  return {
    name: string(row.name, 'field.name'), kind,
    value: scalar(row.value, 'field.value'), nullable: bool(row.nullable, 'field.nullable'),
    choices: row.choices.map(item => string(item, 'field.choice')),
    help: string(row.help, 'field.help'), advanced: bool(row.advanced, 'field.advanced'),
    multiline: bool(row.multiline, 'field.multiline'),
  }
}
export function parseSchema(value: Json): QwenSchema {
  const row = object(value, 'Qwen schema')
  function fields(phase: Phase): QwenField[] {
    const entries = row[phase]
    if (!Array.isArray(entries)) throw new Error(`Missing ${phase} fields`)
    const result = entries.map(field)
    if (new Set(result.map(item => item.name)).size !== result.length) throw new Error(`Duplicate ${phase} fields`)
    return result
  }
  return { cache: fields('cache'), train: fields('train'), generate: fields('generate') }
}
export function initialDraft(schema: QwenSchema): QwenDraft {
  const defaults = (fields: QwenField[]): Values => Object.fromEntries(fields.map(item => [item.name, item.value]))
  return {
    models: defaults(schema.train.filter(item => modelFields.has(item.name))),
    cache: defaults(schema.cache.filter(item => !modelFields.has(item.name))),
    train: defaults(schema.train.filter(item => !modelFields.has(item.name))),
    generate: defaults(schema.generate.filter(item => !modelFields.has(item.name))),
  }
}
export function checkedValues(fields: QwenField[], value: Json): Values {
  const row = object(value, 'Qwen values')
  const entries = fields.map(item => {
    const val = scalar(row[item.name], item.name)
    const valid = val === null ? item.nullable
      : item.kind === 'bool' ? typeof val === 'boolean'
        : item.kind === 'str' ? typeof val === 'string'
          : typeof val === 'number' && Number.isFinite(val) && (item.kind !== 'int' || Number.isInteger(val))
    if (!valid || (item.choices.length > 0 && !item.choices.includes(String(val)))) throw new Error(`Invalid ${item.name}: ${String(val)}`)
    return [item.name, val] as const
  })
  return Object.fromEntries(entries)
}
export function inputValue(field: Pick<QwenField, 'kind' | 'nullable'>, value: Scalar): Scalar {
  if (value === '' || value === null) return field.nullable ? null : value
  return field.kind === 'int' || field.kind === 'float' ? Number(value) : value
}
export function readDraft(schema: QwenSchema, text: string | null): QwenDraft {
  if (text === null) return initialDraft(schema)
  const value: Json = JSON.parse(text)
  const row = object(value, 'Saved Qwen draft')
  return {
    models: checkedValues(schema.train.filter(item => modelFields.has(item.name)), row.models),
    cache: checkedValues(schema.cache.filter(item => !modelFields.has(item.name)), row.cache),
    train: checkedValues(schema.train.filter(item => !modelFields.has(item.name)), row.train),
    generate: checkedValues(schema.generate.filter(item => !modelFields.has(item.name)), row.generate),
  }
}
export async function requestJson(path: string, init: RequestInit): Promise<Json> {
  const response = await fetch(path, init)
  const text = await response.text()
  let body: Json
  try { body = JSON.parse(text) } catch {
    throw new Error(`${path}: HTTP ${response.status}; invalid JSON: ${text.slice(0, 300)}`)
  }
  if (!response.ok) {
    const row = object(body, `HTTP ${response.status}`)
    throw new Error(`${path}: HTTP ${response.status}; ${JSON.stringify(row.detail ?? body)}`)
  }
  return body
}
export function jobId(value: Json): string {
  const id = string(object(value, 'Qwen job').task_id, 'task_id')
  if (!id) throw new Error('Empty task_id in Qwen response')
  return id
}
export function resolvedPaths(value: Json): Record<string, string> {
  const row = object(value, 'Qwen model paths')
  return Object.fromEntries(['dit', 'text_encoder', 'vae', 'processor', 'scheduler'].map(name => [name, string(row[name], name)]))
}
export function resultImages(value: Json): QwenResult[] {
  const row = object(value, 'Qwen results')
  if (!Array.isArray(row.images)) throw new Error('Missing Qwen result images')
  return row.images.map(value => {
    const image = object(value, 'Qwen image')
    return { file: string(image.file, 'file'), prompt: string(image.prompt, 'prompt'), multiplier: number(image.multiplier, 'multiplier'), seed: number(image.seed, 'seed') }
  })
}

export function profileNames(value: Json): string[] {
  if (!Array.isArray(value)) throw new Error('Expected a Qwen profile list')
  return value.map(item => string(item, 'profile name'))
}
export interface CacheStatus {
  directory: string
  pairs: number
  missing_text: number
  missing_latents: number
  errors: string[]
  ready: boolean
}
export function cacheStatus(value: Json): CacheStatus {
  const row = object(value, 'Qwen cache status')
  if (!Array.isArray(row.errors)) throw new Error('Missing cache errors')
  return {
    directory: string(row.directory, 'directory'), pairs: number(row.pairs, 'pairs'),
    missing_text: number(row.missing_text, 'missing_text'), missing_latents: number(row.missing_latents, 'missing_latents'),
    errors: row.errors.map(item => string(item, 'cache error')), ready: bool(row.ready, 'ready'),
  }
}
