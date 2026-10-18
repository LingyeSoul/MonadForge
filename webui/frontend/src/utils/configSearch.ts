export interface SearchableField {
  key: string
  description?: string
  description_en?: string
}

/**
 * Match a field against a whitespace-separated query.
 *
 * `extra` carries surface-specific text a user may type instead of the raw key —
 * a model line's localized label, for instance — so the same matcher serves the
 * Anima editor (keys + backend descriptions) and the typed model forms.
 */
export function matchesConfigSearch(field: SearchableField, query: string | null, extra: string[] = []): boolean {
  const terms = (query ?? '').trim().toLocaleLowerCase().split(/\s+/).filter(Boolean)
  const text = [field.key, field.key.replace(/_/g, ' '), field.description, field.description_en, ...extra]
    .filter(Boolean).join(' ').toLocaleLowerCase()
  return terms.every(term => text.includes(term))
}
