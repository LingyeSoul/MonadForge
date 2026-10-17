export type CurationValues = Record<string, unknown>
export type RootName = 'src' | 'master' | 'dst' | 'masks' | 'out'
export type CurationRoots = Record<RootName, { path: string; exists?: boolean }>
export interface CurationField {
  dest: string; kind: string; label: string; help?: string; default?: unknown
  required?: boolean; choices?: unknown[]; gate?: string | null; group?: string
  root?: string | null; setting?: string | null; report?: string | null
  mask?: unknown; auto?: boolean; overridable?: boolean; advanced?: boolean
  path?: boolean; path_kind?: string
}
export interface CurationStage {
  id: string; title: string; short?: string; panel?: string; available: boolean
  error?: string; notes?: string; doc?: string; hidden?: boolean; scoped?: boolean
  preprocess?: string | null; report?: boolean; apply?: boolean; replay?: boolean; fields: CurationField[]
}
export interface CaptionClause { header: string; prefix: string; position: string; tags: string[] }
export interface ParsedCaption {
  flat_tags: string[]; clauses: CaptionClause[]
  spans: { start: number; end: number; kind: string; clause: number | null }[]
}
export interface CaptionVersion {
  kind: string; rung: string; note?: string; path: string; exists: boolean
  editable: boolean; text: string; parsed?: ParsedCaption | null
}
export interface CurationImage { path: string; width?: number | null; height?: number | null; bytes?: number }
export interface CurationRow {
  rel: string; dir: string; name: string; stem: string; captions: Record<string, boolean>
  resized: boolean; mask: boolean; excluded: boolean
}
export interface OcrLine { seq: number; box: number[]; text: string; det: number; score: number; usable: boolean }
export interface CurationItem {
  rel: string; name: string; dir: string; stem: string; excluded: CurationValues | null
  image: CurationImage | null; resized: CurationImage | null; mask: CurationImage | null
  mask_base: CurationImage | null; ocr_base: CurationImage | null
  versions: CaptionVersion[]; ocr: OcrLine[]
}
export interface CurationGroup { id: number; artist: string; members: string[]; mean_cosine?: number }
export interface CurationGroups { groups: CurationGroup[]; missing?: boolean; stale?: boolean; source_dir?: string }
export interface CurationJob {
  id: string; task_id?: string; stage: string; state: string; apply: boolean
  report_path?: string | null; roots?: CurationRoots; values?: CurationValues
  created_at?: string; error?: string
  operation?: 'run' | 'apply' | 'undo'
}
export interface CurationProposal {
  rel: string; image: string; kind: string; path: string; before: string; after: string
  status: string; before_parsed?: ParsedCaption; after_parsed?: ParsedCaption
  artifact?: string; action?: string
}
export interface CurationSchema {
  stages: CurationStage[]; settings?: CurationValues; roots?: CurationRoots
  ladder?: { kind: string; editable: boolean }[]
}
