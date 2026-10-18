import assert from 'node:assert/strict'
import test from 'node:test'
import { isTaskActive, isTaskStoppable, isTaskTerminal, canResumeTask } from '../src/utils/taskState.ts'

test('a paused task stays active and can resume or stop without a checkpoint', () => {
  const task = { state: 'paused', resumable: false }
  assert.equal(isTaskActive(task.state), true)
  assert.equal(isTaskStoppable(task.state), true)
  assert.equal(isTaskTerminal(task.state), false)
  assert.equal(canResumeTask(task), true)
})

test('terminal training continuation still requires checkpoint evidence', () => {
  for (const state of ['failed', 'cancelled']) {
    assert.equal(canResumeTask({ state, resumable: false }), false)
    assert.equal(canResumeTask({ state, resumable: true }), true)
    assert.equal(isTaskActive(state), false)
    assert.equal(isTaskStoppable(state), false)
    assert.equal(isTaskTerminal(state), true)
  }
  assert.equal(canResumeTask({ state: 'success', resumable: true }), false)
})

test('stopping is active but cannot submit another stop or be deleted as terminal', () => {
  assert.equal(isTaskActive('stopping'), true)
  assert.equal(isTaskStoppable('stopping'), false)
  assert.equal(isTaskTerminal('stopping'), false)
  assert.equal(canResumeTask({ state: 'stopping', resumable: true }), false)
})
