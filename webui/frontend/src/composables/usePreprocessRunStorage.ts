const STORAGE_PREFIX = 'monadforge.preprocessRun'

function contextPart(value: string | null | undefined): string {
  return encodeURIComponent(value || 'default')
}

/** Keep a selected manifest scoped to the config context that can consume it. */
export function preprocessRunStorageKey(
  variant?: string | null,
  preset?: string | null,
): string {
  return `${STORAGE_PREFIX}:${contextPart(variant)}:${contextPart(preset)}`
}

export function readPreprocessRun(
  variant?: string | null,
  preset?: string | null,
): string | null {
  try {
    return localStorage.getItem(preprocessRunStorageKey(variant, preset))
  } catch {
    return null
  }
}

export function writePreprocessRun(
  manifest: string | null,
  variant?: string | null,
  preset?: string | null,
): void {
  try {
    const key = preprocessRunStorageKey(variant, preset)
    if (manifest) localStorage.setItem(key, manifest)
    else localStorage.removeItem(key)
  } catch {
    // Storage can be disabled by the browser; the backend remains authoritative.
  }
}
