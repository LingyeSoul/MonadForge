import assert from 'node:assert/strict'
import test from 'node:test'
import { inputValue } from '../src/utils/qwen21.ts'

test('clearing optional fields restores auto mode, including compile_mode', () => {
  for (const kind of ['str', 'int', 'float']) {
    for (const value of ['', null]) {
      const field = { kind, nullable: true }
      const persisted = JSON.parse(JSON.stringify({ value: inputValue(field, value) }))
      assert.equal(persisted.value, null)
    }
  }
})

test('zero remains explicit and required empty strings retain their meaning', () => {
  assert.equal(inputValue({ kind: 'int', nullable: true }, '0'), 0)
  assert.equal(inputValue({ kind: 'float', nullable: true }, '0'), 0)
  assert.equal(inputValue({ kind: 'str', nullable: false }, ''), '')
  assert.equal(inputValue({ kind: 'str', nullable: true }, 'reduce-overhead'), 'reduce-overhead')
  assert.equal(inputValue({ kind: 'bool', nullable: false }, false), false)
})
