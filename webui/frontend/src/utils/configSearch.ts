export interface SearchableField {
  key: string
  description?: string
  description_en?: string
}

export function matchesConfigSearch(field: SearchableField, query: string | null): boolean {
  const terms = (query ?? '').trim().toLocaleLowerCase().split(/\s+/).filter(Boolean)
  const text = [field.key, field.key.replace(/_/g, ' '), field.description, field.description_en]
    .filter(Boolean).join(' ').toLocaleLowerCase()
  return terms.every(term => text.includes(term))
}
