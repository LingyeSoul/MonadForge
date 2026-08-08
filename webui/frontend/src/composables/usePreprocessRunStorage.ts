const STORAGE_PREFIX = 'monadforge.preprocessRun'
const GLOBAL_STORAGE_KEY = STORAGE_PREFIX

function contextPart(value: string | null | undefined): string {
  return encodeURIComponent(value || 'default')
}

/**
 * Return the single selected preprocessing manifest shared by all training
 * methods and presets. The optional arguments remain for source compatibility
 * with older callers, but are intentionally ignored.
 */
export function preprocessRunStorageKey(
  _variant?: string | null,
  _preset?: string | null,
): string {
  return GLOBAL_STORAGE_KEY
}

function legacyStorageKey(variant?: string | null, preset?: string | null): string {
  return `${STORAGE_PREFIX}:${contextPart(variant)}:${contextPart(preset)}`
}

export function readPreprocessRun(
  variant?: string | null,
  preset?: string | null,
): string | null {
  try {
    const current = localStorage.getItem(GLOBAL_STORAGE_KEY)
    if (current) return current

    // One-time best-effort migration from the pre-fix method/preset keys. This
    // keeps a run selected before the upgrade without preserving the coupling.
    let legacy = localStorage.getItem(legacyStorageKey(variant, preset))
    if (!legacy) {
      // The active method may not be the method that created the old key. Scan
      // only this namespace so an upgrade does not strand the prior selection.
      for (let index = 0; index < localStorage.length; index += 1) {
        const key = localStorage.key(index)
        if (!key?.startsWith(`${STORAGE_PREFIX}:`)) continue
        legacy = localStorage.getItem(key)
        if (legacy) break
      }
    }
    if (legacy) localStorage.setItem(GLOBAL_STORAGE_KEY, legacy)
    return legacy
  } catch {
    return null
  }
}

export function writePreprocessRun(
  manifest: string | null,
  _variant?: string | null,
  _preset?: string | null,
): void {
  try {
    if (manifest) localStorage.setItem(GLOBAL_STORAGE_KEY, manifest)
    else localStorage.removeItem(GLOBAL_STORAGE_KEY)
  } catch {
    // Storage can be disabled by the browser; the backend remains authoritative.
  }
}
