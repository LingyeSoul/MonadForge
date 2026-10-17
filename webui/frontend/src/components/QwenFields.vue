<template>
  <v-row>
    <v-col v-for="field in fields" :key="field.name" cols="12" :sm="field.multiline ? 12 : 6">
      <ConfigField :field="metadata(field)" :model-value="qwen.draft?.[section][field.name]"
        @update="qwen.update(section, field.name, $event as Scalar)" @help-click="emit('help', `${section}.${field.name}`)" />
    </v-col>
  </v-row>
</template>

<script setup lang="ts">
import ConfigField from './ConfigField.vue'
import { useI18n } from '../composables/useI18n'
import { useActiveModelWorkspace, type QwenSection } from '../stores/qwenWorkspace'
import { useAppStore } from '../stores/app'
import { modelFieldLabel } from '../i18n/modelWorkspace'
import type { FieldMeta } from '../stores/config'
import type { QwenField, Scalar } from '../utils/qwen21'

const props = defineProps<{ section: QwenSection; fields: QwenField[] }>()
const emit = defineEmits<{ help: [key: string] }>()
const qwen = useActiveModelWorkspace()
const app = useAppStore()
const { t } = useI18n()
const pathKinds: Record<string, 'file' | 'directory' | 'either'> = {
  model_dir: 'directory', dit: 'either', text_encoder: 'either', vae: 'either', processor: 'directory', scheduler: 'either',
  src: 'directory', cache: 'directory', out: 'directory', out_dir: 'directory', output: 'file', lora: 'file', prompts_file: 'file',
}
function metadata(field: QwenField): FieldMeta {
  const path = pathKinds[field.name]
  return {
    key: `${props.section}.${field.name}`, label: modelFieldLabel(qwen.value.family, app.language, field.name, t(`qwenField_${field.name}`)),
    value: qwen.value.draft?.[props.section][field.name], default_value: field.value,
    field_type: path ? 'path' : field.choices.length ? 'select' : field.kind,
    description: field.nullable ? t('qwenAuto') : '', origin: 'method', group: props.section,
    is_virtual: false, read_only: qwen.value.busy, options: field.choices, nullable: field.nullable,
    path_kind: path, path_extensions: field.name === 'scheduler' ? '.json' : field.name === 'prompts_file' ? '.txt' : '.safetensors',
    multiline: field.multiline,
  }
}
</script>
