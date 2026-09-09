// Guidebook markdown rendering shared by the dialog and its tests.
//
// marked removed automatic heading ids (headerIds) in v12, so the
// guidebook's hand-written TOC links like `[§7.4](#74-dataset-tab-...)`
// pointed at nothing. We re-derive ids with the same convention the
// documents use: lowercase, drop punctuation/symbols (including CJK
// fullwidth punctuation and the middle dot) but keep hyphens, turn each
// whitespace character into one hyphen, keep letters/digits/underscores.
import { Marked, type Tokens } from 'marked'

export function slugifyAnchor(text: string): string {
  return text
    .replace(/[^\p{L}\p{N}_\-\s]+/gu, '')
    .toLowerCase()
    .replace(/\s/g, '-')
}

export function renderGuidebook(markdown: string): string {
  const renderer = {
    heading(this: { parser: { parseInline(tokens: Tokens.Generic[]): string } }, token: Tokens.Heading) {
      const raw = token.tokens.map(part => part.raw).join('')
      const html = this.parser.parseInline(token.tokens)
      return `<h${token.depth} id="${slugifyAnchor(raw)}">${html}</h${token.depth}>`
    },
  }
  // A private instance keeps the custom renderer out of the global marked
  // configuration in case other code starts using marked later on.
  const md = new Marked({ breaks: true, renderer, async: false })
  return md.parse(markdown) as string
}
