import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import {
  curationFieldChoices, curationFieldCopy, curationFieldCopyCatalog, curationFieldGroup,
  curationStageCopyCatalog, curationStageNotes, curationStageSummary,
} from '../src/i18n/curationSchema.ts'

test('negated CLI flags describe the enabled destination value in the form', () => {
  const cases = [
    ['sheets', '--no_sheets', '生成审阅联系表', 'Create review sheets'],
    ['skip_en', '--keep_en', '过滤纯 ASCII 文本', 'Filter ASCII-only lines'],
    ['strip_symbols', '--keep_symbols', '移除装饰符号', 'Remove decorative symbols'],
    ['strict_count', '--no_strict_count', '严格核对人物数量', 'Require matching subject counts'],
    ['discriminative_only', '--keep_shared_tags', '子句仅保留差异特征', 'Use distinctive clause tags only'],
    ['rewrite', '--no_rewrite', '将专属标签移入子句', 'Move attributed tags into clauses'],
  ]
  for (const [dest, label, cn, en] of cases) {
    const field = { dest, label, kind: 'bool', default: true }
    assert.equal(curationFieldCopy('cn', field).label, cn)
    assert.equal(curationFieldCopy('en', field).label, en)
    assert.equal(field.default, true)
  }
})

test('localized choices preserve upstream wire values, including unknown extensions', () => {
  const field = Object.freeze({ dest: 'mode', kind: 'enum', choices: Object.freeze(['missing', 'merge', 'overwrite', 'future-mode']) })
  const choices = curationFieldChoices('cn', field)
  assert.deepEqual(choices.map(choice => choice.value), field.choices)
  assert.equal(choices[0].title, '仅补齐缺失标注')
  assert.equal(choices[3].title, 'future-mode')
  assert.deepEqual(curationFieldCopy('zh-CN', { dest: 'mode' }), curationFieldCopy('cn', { dest: 'mode' }))
  assert.deepEqual(curationFieldCopy('fr', { dest: 'mode' }), curationFieldCopy('en', { dest: 'mode' }))
})

test('export presentation retains source-write risks and recovery safeguards', () => {
  const stage = { id: 'export' }
  assert.match(curationStageNotes('cn', stage), /回写源标注/)
  assert.match(curationStageNotes('cn', stage), /删除过期附属文件/)
  assert.match(curationStageNotes('cn', stage), /撤销使用任务备份/)
  assert.match(curationStageNotes('cn', stage), /目标或备份已变化时会拒绝恢复/)
  assert.doesNotMatch(curationStageNotes('cn', stage), /图片替换无法撤销/)
  assert.match(curationFieldCopy('cn', { dest: 'sidecars_only' }).help, /不能与尺寸限制、WebP/)
})

// Set CURATION_SCHEMA_PATH to a read-only GET /api/curation/schema capture to
// verify the installed upstream package, including fields outside the defaults.
const schemaPath = process.env.CURATION_SCHEMA_PATH
test('installed curation schema has complete bilingual presentation and unchanged choices', { skip: !schemaPath }, () => {
  const schema = JSON.parse(readFileSync(schemaPath, 'utf8'))
  assert.equal(schema.stages.length, 11)
  const dests = new Set()
  for (const stage of schema.stages) {
    assert.ok(curationStageCopyCatalog[stage.id], `Missing stage copy: ${stage.id}`)
    for (const language of ['cn', 'en']) {
      assert.ok(curationStageSummary(language, stage))
      assert.ok(curationStageNotes(language, stage))
    }
    for (const field of stage.fields) {
      dests.add(field.dest)
      assert.ok(curationFieldCopyCatalog[field.dest], `Missing field copy: ${stage.id}.${field.dest}`)
      for (const language of ['cn', 'en']) {
        const copy = curationFieldCopy(language, field)
        assert.ok(copy.label && copy.help, `${language}: ${field.dest}`)
        assert.doesNotMatch(copy.label, /^--/)
        assert.doesNotMatch(copy.help, /contract\.|docs\//)
        if (language === 'cn') assert.match(copy.label, /[\u3400-\u9fff]/)
        assert.deepEqual(curationFieldChoices(language, field).map(choice => choice.value), field.choices || [])
      }
      if (field.group) assert.notEqual(curationFieldGroup('cn', field.group), field.group)
    }
  }
  assert.ok(dests.size >= 117, `Expected the full schema, got ${dests.size} field destinations`)
})
