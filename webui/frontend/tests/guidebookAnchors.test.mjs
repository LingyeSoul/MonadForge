import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { renderGuidebook, slugifyAnchor } from '../src/utils/guidebook.ts'

// The three guidebooks actually served by webui/api/docs.py
const DOCS = ['guidebook.md', 'guidebook_en.md', '가이드북.md']
  .map(name => ({ name, text: readFileSync(new URL(`../../../docs/guidelines/${name}`, import.meta.url), 'utf8') }))

test('slugifyAnchor drops punctuation (incl. fullwidth/CJK) and maps each space to one hyphen', () => {
  assert.equal(slugifyAnchor('6. 预处理：缩放 · 潜变量 · 文本嵌入缓存'), '6-预处理缩放--潜变量--文本嵌入缓存')
  assert.equal(slugifyAnchor('8.6 自动续训（`checkpointing_epochs`）'), '86-自动续训checkpointing_epochs')
  assert.equal(slugifyAnchor('7.4 Dataset Tab: Autotag & Grouping'), '74-dataset-tab-autotag--grouping')
  assert.equal(slugifyAnchor('Hello World'), 'hello-world')
})

test('renderGuidebook injects heading ids', () => {
  const html = renderGuidebook('## 1. 系统要求\n')
  assert.equal(html, '<h2 id="1-系统要求">1. 系统要求</h2>')
})

for (const { name, text } of DOCS) {
  test(`${name}: every in-document anchor link resolves to a rendered heading id`, () => {
    const html = renderGuidebook(text)
    const ids = new Set([...html.matchAll(/<h[1-6] id="([^"]*)"/g)].map(match => match[1]))
    assert.ok(ids.size > 0, 'no heading ids rendered')

    // Match against rendered hrefs (marked percent-encodes them, like the
    // DOM the click handler actually sees), decoded back to raw characters.
    const anchors = [...html.matchAll(/<a href="#([^"]*)"/g)].map(match => decodeURIComponent(match[1]))
    assert.ok(anchors.length > 0, 'no anchor links found in document')

    const dangling = anchors.filter(anchor => !ids.has(anchor))
    assert.deepEqual(dangling, [], `unresolvable anchors: ${dangling.join(', ')}`)
  })

  test(`${name}: heading ids are unique`, () => {
    const html = renderGuidebook(text)
    const ids = [...html.matchAll(/<h[1-6] id="([^"]*)"/g)].map(match => match[1])
    assert.equal(new Set(ids).size, ids.length)
  })
}
