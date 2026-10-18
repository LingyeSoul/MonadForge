import assert from 'node:assert/strict'
import test from 'node:test'
import { matchesConfigSearch } from '../src/utils/configSearch.ts'

const field = {
  key: 'learning_rate',
  description: 'Optimizer step size',
  description_en: 'Learning rate for the adapter',
}

test('empty and cleared searches retain every field', () => {
  for (const query of ['', null, ' \t ']) {
    assert.equal(matchesConfigSearch(field, query), true)
  }
})

test('parameter keys are case insensitive and support spaces', () => {
  for (const query of ['LEARNING_RATE', 'learning rate', 'rate learning']) {
    assert.equal(matchesConfigSearch(field, query), true)
  }
})

test('search includes both descriptions', () => {
  assert.equal(matchesConfigSearch(field, 'optimizer'), true)
  assert.equal(matchesConfigSearch(field, 'adapter'), true)
})

test('all search terms must match', () => {
  assert.equal(matchesConfigSearch(field, 'learning adapter'), true)
  assert.equal(matchesConfigSearch(field, 'learning checkpoint'), false)
})

test('missing descriptions and special characters are safe', () => {
  assert.equal(matchesConfigSearch({ key: 'lr' }, '['), false)
  assert.equal(matchesConfigSearch({ key: 'lr' }, 'lr'), true)
})

test('localized descriptions remain searchable', () => {
  const localized = { key: 'learning_rate', description: '\u5b66\u4e60\u7387' }
  assert.equal(matchesConfigSearch(localized, '\u5b66\u4e60'), true)
})

test('extra text lets a typed model form match its localized label', () => {
  const typed = { key: 'blocks_to_swap', description: 'transformer blocks kept in host memory' }
  assert.equal(matchesConfigSearch(typed, 'blocks_to_swap'), true)
  assert.equal(matchesConfigSearch(typed, '\u4ea4\u6362\u5757\u6570'), false)
  assert.equal(matchesConfigSearch(typed, '\u4ea4\u6362\u5757\u6570', ['\u4ea4\u6362\u5757\u6570', '\u4ea4\u6362\u5230\u5185\u5b58\u7684\u5757\u6570']), true)
  // Every term must match somewhere in the combined text.
  assert.equal(matchesConfigSearch(typed, 'swap \u4ea4\u6362', ['\u4ea4\u6362\u5757\u6570']), true)
  assert.equal(matchesConfigSearch(typed, 'swap \u4e0d\u5b58\u5728', ['\u4ea4\u6362\u5757\u6570']), false)
})
