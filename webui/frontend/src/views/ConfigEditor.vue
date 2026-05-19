<template>
  <v-container fluid class="pa-4">
    <v-row align="center" class="mb-4">
      <v-col cols="12" md="3">
        <v-select
          v-model="selectedMethod"
          :items="configStore.methods"
          :label="t('cfgMethod')"
          variant="outlined"
          density="compact"
          hide-details
          @update:model-value="onMethodChange"
        />
      </v-col>
      <v-col cols="12" md="3">
        <v-select
          v-model="selectedVariant"
          :items="variantItems"
          item-title="label"
          item-value="value"
          :label="t('cfgVariant')"
          variant="outlined"
          density="compact"
          hide-details
          :disabled="!selectedMethod"
          @update:model-value="onVariantChange"
        />
      </v-col>
      <v-col cols="12" md="2">
        <v-select
          v-model="selectedPreset"
          :items="configStore.presets"
          :label="t('cfgPreset')"
          variant="outlined"
          density="compact"
          hide-details
          @update:model-value="onVariantChange"
        />
      </v-col>
      <v-col cols="12" md="4" class="d-flex justify-end ga-2">
        <v-btn
          color="primary"
          :loading="configStore.loading"
          :disabled="!configStore.dirty"
          prepend-icon="mdi-content-save"
          @click="configStore.save()"
        >
          {{ t('cfgSave') }}
        </v-btn>
        <v-btn
          variant="outlined"
          :disabled="!selectedVariant"
          prepend-icon="mdi-refresh"
          @click="loadConfig"
        >
          {{ t('cfgReload') }}
        </v-btn>
      </v-col>
    </v-row>

    <v-alert
      v-if="configStore.error"
      type="error"
      variant="tonal"
      closable
      class="mb-4"
      @click:close="configStore.error = ''"
    >
      {{ configStore.error }}
    </v-alert>

    <v-progress-linear
      v-if="configStore.loading"
      indeterminate
      color="primary"
      class="mb-4"
    />

    <v-card v-if="configStore.basicFields.length > 0" class="mb-4" variant="tonal">
      <v-card-title class="text-subtitle-1">
        <v-icon icon="mdi-tune" class="mr-2" />
        {{ t('cfgBasicSettings') }}
      </v-card-title>
      <v-card-text>
        <v-row>
          <v-col
            v-for="field in configStore.basicFields"
            :key="field.key"
            cols="12"
            md="6"
          >
            <ConfigField
              :field="field"
              @update="(v) => configStore.setFieldValue(field.key, v)"
            />
          </v-col>
        </v-row>
      </v-card-text>
    </v-card>

    <v-expansion-panels v-if="Object.keys(configStore.groupedAdvanced).length > 0" variant="accordion">
      <v-expansion-panel
        v-for="(groupFields, groupName) in configStore.groupedAdvanced"
        :key="groupName"
      >
        <v-expansion-panel-title>
          <v-icon icon="mdi-cog-outline" class="mr-2" size="small" />
          {{ groupName }}
          <v-chip size="x-small" class="ml-2" variant="outlined">{{ groupFields.length }}</v-chip>
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <v-row>
            <v-col
              v-for="field in groupFields"
              :key="field.key"
              cols="12"
              md="6"
            >
              <ConfigField
                :field="field"
                @update="(v) => configStore.setFieldValue(field.key, v)"
              />
            </v-col>
          </v-row>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>

    <div v-if="!configStore.loading && configStore.fields.length === 0" class="text-center pa-12">
      <v-icon icon="mdi-cog-transfer-outline" size="64" color="grey" class="mb-4" />
      <div class="text-h6 text-medium-emphasis">{{ t('cfgSelectHint') }}</div>
    </div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useConfigStore } from '../stores/config'
import { useI18n } from '../composables/useI18n'
import ConfigField from '../components/ConfigField.vue'

const configStore = useConfigStore()
const { t } = useI18n()

const selectedMethod = ref('')
const selectedVariant = ref('')
const selectedPreset = ref('default')

const variantItems = computed(() =>
  configStore.variants.map(v => ({
    value: v,
    label: configStore.variantLabels[v] || v,
  }))
)

async function onMethodChange(method: string) {
  selectedVariant.value = ''
  await configStore.fetchVariants(method)
}

async function onVariantChange() {
  if (selectedVariant.value) {
    await loadConfig()
  }
}

async function loadConfig() {
  if (selectedVariant.value) {
    await configStore.fetchMerged(selectedVariant.value, selectedPreset.value)
  }
}

onMounted(async () => {
  await Promise.all([
    configStore.fetchMethods(),
    configStore.fetchPresets(),
  ])
})
</script>
