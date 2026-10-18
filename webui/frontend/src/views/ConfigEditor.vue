<template>
  <v-container fluid class="pa-4 config-editor">
    <WorkspaceHeading :title="t('cfgWorkspaceTitle')" :subtitle="t('cfgWorkspaceMeta', { model: modelLabel(workspace.family) })">
      <span v-if="workspaceDirty" class="config-dirty" role="status">
        <v-icon icon="mdi-circle-small" size="20" />{{ t('cfgPendingChanges') }}
      </span>
    </WorkspaceHeading>
    <ConfigToolbar>
      <template #model>
        <v-select :model-value="workspace.family" :items="modelItems" :label="t('qwBaseModel')"
          density="compact" variant="outlined" hide-details data-testid="config-model-family"
          :disabled="toolbarBusy || continuation.active" @update:model-value="requestModelChange" />
      </template>
      <template #runtime><RuntimeSelector :disabled="toolbarBusy || continuation.active || taskStore.loading" /></template>
      <template #method>
        <v-select :model-value="workspaceMethod || undefined" :items="workspaceMethodItems" :label="t('cfgMethod')"
          :disabled="toolbarBusy || continuation.active" variant="outlined" density="compact" hide-details
          data-testid="config-method-select" @update:model-value="requestMethodChange" />
      </template>
      <template #variant>
        <v-select :model-value="workspaceVariant || undefined" :items="workspaceVariantItems" item-title="label" item-value="value"
          :label="t('cfgVariant')" :disabled="!workspaceMethod || toolbarBusy || continuation.active"
          variant="outlined" density="compact" hide-details data-testid="config-variant"
          @update:model-value="requestVariantChange" />
        <v-btn icon="mdi-plus" variant="text" density="compact" size="small"
          :disabled="!workspaceMethod || toolbarBusy || continuation.active" :aria-label="t('cfgNewVariant')"
          data-testid="config-new-variant" @click="openWorkspaceVariant">
          <v-icon icon="mdi-plus" /><v-tooltip activator="parent">{{ t('cfgNewVariant') }}</v-tooltip>
        </v-btn>
      </template>
      <template #preset>
        <!-- Presets are an Anima concept; the typed model workspaces carry their
             own saved profiles, so the slot stays empty rather than showing a
             disabled control that looks editable. -->
        <template v-if="isAnima">
        <v-select :model-value="isAnima ? selectedPreset : modelPresetLabel"
          :items="isAnima ? continuation.presetItems : [modelPresetLabel]" :label="t('cfgPreset')"
          :disabled="!isAnima || toolbarBusy || continuation.active" variant="outlined" density="compact" hide-details
          data-testid="config-preset" @update:model-value="requestPresetChange" />
        <v-btn icon="mdi-plus" variant="text" density="compact" size="small"
          :disabled="!isAnima || toolbarBusy || continuation.active" :aria-label="t('cfgPresetCreate')" @click="openCreatePreset">
          <v-icon icon="mdi-plus" /><v-tooltip activator="parent">{{ t('cfgPresetCreate') }}</v-tooltip>
        </v-btn>
        <v-btn v-if="isAnima && isCustomPreset" icon="mdi-delete-outline" variant="text" density="compact" size="small"
          color="error" :disabled="toolbarBusy || continuation.active" :aria-label="t('cfgPresetDelete')" @click="deleteCurrentPreset" />
        </template>
      </template>
      <template #task>
        <!-- Continuation reads Anima checkpoints; a typed line has no candidate list. -->
        <v-select v-if="isAnima" :model-value="isAnima ? continuation.selected : undefined" :items="isAnima ? trainingTaskItems : []"
          :label="t('cfgTrainingTask')" :placeholder="t(isAnima ? 'cfgTrainingTaskEmpty' : 'cfgNewTrainingOnly')"
          :loading="isAnima && (continuation.loading || continuation.listLoading)"
          :disabled="!isAnima || !workspaceReady || toolbarBusy || configStore.loading"
          item-title="title" item-value="value" clearable persistent-placeholder
          variant="outlined" density="compact" hide-details data-testid="training-task-select"
          @update:model-value="continuation.select" @update:menu="(open) => isAnima && open && continuation.fetchCandidates(selectedVariant)" />
      </template>
      <template #actions>
        <v-btn :color="workspaceDirty ? 'warning' : 'primary'" :loading="isAnima ? configStore.loading : qwen.busy"
          :disabled="!workspaceReady || !workspaceDirty || toolbarBusy" icon="mdi-content-save-outline" variant="text" size="small"
          :aria-label="t('cfgSave')" data-testid="config-save" @click="saveWorkspace">
          <v-icon icon="mdi-content-save-outline" /><v-tooltip activator="parent">{{ t('cfgSave') }}</v-tooltip>
        </v-btn>
        <v-btn icon="mdi-refresh" variant="text" size="small" :disabled="!workspaceReady || toolbarBusy || continuation.loading"
          :aria-label="t('cfgReload')" data-testid="config-reload" @click="requestWorkspaceChange({ kind: 'reload' })">
          <v-icon icon="mdi-refresh" /><v-tooltip activator="parent">{{ t('cfgReload') }}</v-tooltip>
        </v-btn>
        <v-btn color="primary" :disabled="!workspaceReady || toolbarBusy || (isAnima && continuation.active && !continuation.ready)"
          :loading="isAnima ? trainingLaunching : lineLaunching === 'train'" prepend-icon="mdi-play"
          data-testid="config-train" @click="isAnima ? startTraining() : startQwenTraining()">{{ t('cfgTrain') }}</v-btn>
        <v-btn variant="outlined" :disabled="!workspaceReady || toolbarBusy" :loading="isAnima ? testLaunching : lineLaunching === 'test'"
          prepend-icon="mdi-test-tube" data-testid="config-test" @click="isAnima ? startTest() : startQwenTest()">{{ t('cfgTest') }}</v-btn>
      </template>
    </ConfigToolbar>
    <v-alert v-if="!isAnima && !workspaceReady && qwen.error" type="error" variant="tonal" class="mb-4">{{ qwen.error }}</v-alert>
    <v-progress-linear v-if="!isAnima && !workspaceReady && (qwen.loading || switching)" indeterminate class="mb-4" />
    <ConfigurationPicker v-if="!workspaceReady" :title="t(workspaceMethod ? 'cfgVariantLibrary' : 'cfgMethodLibrary')"
      :items="workspacePickerItems" :variant="Boolean(workspaceMethod)" :disabled="toolbarBusy"
      :create-label="workspaceMethod ? t('cfgNewVariant') : undefined" :empty-text="t('cfgSelectHint')"
      @select="chooseWorkspaceConfiguration" @create="openWorkspaceVariant" />
    <QwenConfigFields v-if="!isAnima && workspaceReady" :key="workspace.family" />
    <template v-if="isAnima">
    <!-- WandB Tracking Panel -->
    <v-expansion-panels v-if="selectedVariant" v-model="wandbPanel" class="mb-4 config-tracking" variant="accordion">
      <v-expansion-panel elevation="0">
        <v-expansion-panel-title class="text-subtitle-2">
          <v-icon icon="mdi-chart-box" class="mr-2" />
          {{ t('cfgWandbSection') }}
          <v-spacer />
          <v-switch
            v-model="wandb.enabled"
            :label="t('cfgWandbEnabled')"
            density="compact"
            hide-details
            color="primary"
            class="mr-2"
            @click.stop
            @update:model-value="saveWandbSettings"
          />
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <v-row density="comfortable">
            <v-col cols="12" md="4">
              <v-text-field
                v-model="wandb.project"
                :label="t('cfgWandbProject')"
                variant="outlined"
                density="compact"
                hide-details
                @change="saveWandbSettings"
              />
            </v-col>
            <v-col cols="12" md="4">
              <v-text-field
                v-model="wandb.run_name"
                :label="t('cfgWandbRunName')"
                variant="outlined"
                density="compact"
                hide-details
                @change="saveWandbSettings"
              />
            </v-col>
            <v-col cols="12" md="4">
              <v-text-field
                v-model="wandb.api_key"
                :label="t('cfgWandbApiKey')"
                type="password"
                variant="outlined"
                density="compact"
                hide-details
                @change="saveWandbSettings"
              />
            </v-col>
          </v-row>
          <v-row density="comfortable" class="mt-2">
            <v-col cols="12" md="3">
              <v-text-field
                v-model.number="wandb.log_every_n_steps"
                :label="t('cfgWandbLogEvery')"
                type="number"
                variant="outlined"
                density="compact"
                hide-details
                min="1"
                @change="saveWandbSettings"
              />
            </v-col>
            <v-col cols="12" md="3" class="d-flex align-center">
              <v-checkbox
                v-model="wandb.log_gradients"
                :label="t('cfgWandbLogGradients')"
                density="compact"
                hide-details
                color="primary"
                @update:model-value="saveWandbSettings"
              />
            </v-col>
            <v-col cols="12" md="3" class="d-flex align-center">
              <v-checkbox
                v-model="wandb.log_weights"
                :label="t('cfgWandbLogWeights')"
                density="compact"
                hide-details
                color="primary"
                @update:model-value="saveWandbSettings"
              />
            </v-col>
            <v-col cols="12" md="3" class="d-flex align-center">
              <v-checkbox
                v-model="wandb.log_checkpoint_artifact"
                :label="t('cfgWandbLogArtifact')"
                density="compact"
                hide-details
                color="primary"
                @update:model-value="saveWandbSettings"
              />
            </v-col>
          </v-row>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>

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

    <v-alert
      v-if="isExperimental"
      type="warning"
      variant="tonal"
      density="compact"
      class="mb-4"
      icon="mdi-flask-outline"
    >
      {{ t('cfgExperimentalWarning') }}
    </v-alert>

    <v-alert
      v-if="prelaunchResult?.model_layout"
      :type="prelaunchResult.compatibility?.supported ? 'success' : 'error'"
      variant="tonal"
      density="compact"
      class="mb-4"
      :icon="prelaunchResult.compatibility?.supported ? 'mdi-shield-check-outline' : 'mdi-shield-alert-outline'"
    >
      <div class="text-subtitle-2">{{ t('cfgModelCompatibility') }}</div>
      <div>
        {{ t('cfgModelDetected', {
          variant: prelaunchResult.model_layout.variant,
          blocks: prelaunchResult.model_layout.num_blocks,
          profile: prelaunchResult.compatibility?.profile,
        }) }}
      </div>
      <ul v-if="prelaunchResult.compatibility?.blockers?.length" class="mt-2 pl-5">
        <li v-for="blocker in prelaunchResult.compatibility.blockers" :key="blocker">
          {{ localizeAnimaBlocker(blocker) }}
        </li>
      </ul>
    </v-alert>

    <ConfigWorkspaceLayout v-if="selectedVariant" v-model:search="fieldSearch" v-model:show-guide="showGuide"
      :field-count="visibleFieldCount" :filterable="Boolean(selectedVariant && configStore.fields.length)"
      search-test-id="anima-field-search" help-test-id="anima-help-toggle">
        <v-card v-if="filteredBasic.length > 0" class="mb-4 config-section" variant="flat">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-tune" class="mr-2" />
            {{ t('cfgBasicSettings') }}
          </v-card-title>
          <v-card-text>
            <v-row>
              <v-col
                v-for="field in filteredBasic"
                :key="field.key"
                cols="12"
                sm="6"
              >
                <ConfigField
                  :field="{ ...field, read_only: field.read_only || configStore.loading }"
                  @update="(v) => onConfigFieldUpdate(field.key, v)"
                  @help-click="(k) => onFieldHelp(k, field.origin)"
                />
              </v-col>
            </v-row>
          </v-card-text>
        </v-card>

        <v-expansion-panels v-if="Object.keys(filteredAdvanced).length > 0"
          :model-value="fieldSearch ? Object.keys(filteredAdvanced) : expandedGroups"
          multiple variant="accordion" @update:model-value="onExpandedGroups">
          <v-expansion-panel
            v-for="(groupFields, groupName) in filteredAdvanced"
            :key="groupName"
            :value="groupName"
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
                  :sm="field.key === 'sample_prompts' || field.field_type === 'regex_set' ? 12 : 6"
                >
                  <PreviewPromptEditor
                    v-if="field.key === 'sample_prompts' && !continuation.active"
                    :prompt-path="String(field.value ?? 'sample_prompts.txt')"
                  />
                  <ConfigField
                    v-else
                    :field="{ ...field, read_only: field.read_only || configStore.loading }"
                    @update="(v) => onConfigFieldUpdate(field.key, v)"
                    @help-click="(k) => onFieldHelp(k, field.origin)"
                  />
                </v-col>
              </v-row>
            </v-expansion-panel-text>
          </v-expansion-panel>
        </v-expansion-panels>
        <div v-if="fieldSearch && !filteredBasic.length && !Object.keys(filteredAdvanced).length" class="workspace-empty" role="status">
          <v-icon icon="mdi-magnify" size="32" />
          <h2>{{ t('cfgNoMatchingFields') }}</h2>
          <v-btn variant="outlined" @click="fieldSearch = ''">{{ t('cfgClearSearch') }}</v-btn>
        </div>

        <!-- Extra Args section -->
        <v-card v-if="selectedVariant" class="mt-4 config-section" variant="flat">
          <v-card-title
            class="text-subtitle-1 d-flex align-center cursor-pointer"
            @click="showExtraArgs = !showExtraArgs"
          >
            <v-icon icon="mdi-code-braces" class="mr-2" size="small" />
            {{ t('cfgExtraArgs') }}
            <v-spacer />
            <v-icon :icon="showExtraArgs ? 'mdi-chevron-up' : 'mdi-chevron-down'" />
          </v-card-title>
          <v-expand-transition>
            <v-card-text v-show="showExtraArgs">
              <v-textarea
                v-model="extraArgs"
                :disabled="continuation.active || configStore.loading"
                :placeholder="t('cfgExtraArgsHint')"
                variant="outlined"
                density="compact"
                rows="4"
                auto-grow
                max-rows="10"
                hide-details
                style="font-family: monospace; font-size: 13px;"
              />
            </v-card-text>
          </v-expand-transition>
        </v-card>

      <template v-if="selectedVariant" #guide>
        <HelpPanel ref="helpPanelRef" :variant="selectedVariant" :field-help="fieldHelpData"
          :guide-html="guideHtml" height="min(680px, calc(100dvh - 120px))" />
      </template>
    </ConfigWorkspaceLayout>

    <!-- No-cache warning dialog -->
    <v-dialog v-model="showNoCacheDlg" max-width="500">
      <v-card>
        <v-card-title>{{ t('cfgNoCache') }}</v-card-title>
        <v-card-text>{{ t('cfgNoCacheBody') }}</v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showNoCacheDlg = false">{{ t('dsCancel') }}</v-btn>
          <v-btn color="primary" :loading="preprocessRunning" @click="runPreprocessThenTrain">
            {{ t('cfgRunPreprocess') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Legacy LoKR full-factor sentinel migration -->
    <v-dialog v-model="showLegacyLokrDlg" max-width="560">
      <v-card>
        <v-card-title>{{ t('cfgLokrLegacyTitle') }}</v-card-title>
        <v-card-text>
          {{ t('cfgLokrLegacyBody') }}
          <pre class="lokr-migration-preview">network_dim = 32
network_alpha = 32
lokr_full_factor = true
decompose_both = false</pre>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showLegacyLokrDlg = false">
            {{ t('cfgLokrLegacyBack') }}
          </v-btn>
          <v-btn color="primary" prepend-icon="mdi-auto-fix" @click="applyLokrFullFactorMigration">
            {{ t('cfgLokrLegacyApply') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Conditioning interception dialog -->
    <v-dialog v-model="showConditioningDlg" max-width="520">
      <v-card>
        <v-card-title>{{ t('cfgConditioningBlocked') }}</v-card-title>
        <v-card-text>
          {{ t('cfgConditioningBlockedBody', { keys: conditioningCtx.keys.join(', ') }) }}
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showConditioningDlg = false">{{ t('dsCancel') }}</v-btn>
          <v-btn color="primary" @click="confirmRemoveConditioning">
            {{ t('cfgConditioningRemove') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Create preset dialog -->
    <v-dialog v-model="showCreatePresetDlg" max-width="400">
      <v-card>
        <v-card-title>{{ t('cfgPresetCreate') }}</v-card-title>
        <v-card-text>
          <v-text-field
            v-model="newPresetName"
            :label="t('cfgPresetName')"
            :placeholder="t('cfgPresetNameHint')"
            variant="outlined"
            density="compact"
            autofocus
            @keyup.enter="confirmCreatePreset"
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showCreatePresetDlg = false">{{ t('cfgCancel') }}</v-btn>
          <v-btn
            color="primary"
            :disabled="!newPresetName.trim()"
            @click="confirmCreatePreset"
          >
            {{ t('cfgPresetCreate') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    </template>
    <v-dialog v-model="showCreateVariantDlg" max-width="400">
      <v-card>
        <v-card-title>{{ t('cfgNewVariant') }}</v-card-title>
        <v-card-text>
          <v-text-field v-model="newVariantName" data-testid="config-variant-name"
            :label="t('cfgNewVariantName')" :placeholder="t('cfgNewVariantNameHint')" :rules="[variantNameRule]"
            :disabled="creatingVariant" variant="outlined" density="compact" autofocus @keyup.enter="confirmCreateWorkspaceVariant" />
          <v-checkbox v-model="seedFromCurrent" :label="t('cfgNewVariantSeed')" :disabled="!workspaceReady || creatingVariant"
            density="compact" hide-details />
          <v-alert v-if="isAnima ? configStore.error : qwen.error" type="error" variant="tonal" class="mt-3">
            {{ isAnima ? configStore.error : qwen.error }}
          </v-alert>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" :disabled="creatingVariant" @click="showCreateVariantDlg = false">{{ t('cfgCancel') }}</v-btn>
          <v-btn data-testid="config-variant-create" color="primary" :loading="creatingVariant"
            :disabled="!newVariantName.trim() || variantNameRule(newVariantName.trim()) !== true" @click="confirmCreateWorkspaceVariant">
            {{ t('cfgNewVariant') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
    <v-alert v-if="workspaceError" type="error" variant="tonal" data-testid="workspace-switch-error">{{ workspaceError }}</v-alert>
    <v-dialog :model-value="pendingChange !== null" persistent max-width="500">
      <v-card :title="t('qwUnsavedTitle')">
        <v-card-text>{{ t('qwUnsavedBody') }}</v-card-text>
        <v-card-actions class="flex-wrap">
          <v-btn data-testid="model-switch-cancel" :disabled="switching" @click="cancelWorkspaceChange">{{ t('dsCancel') }}</v-btn>
          <v-spacer />
          <v-btn data-testid="model-switch-discard" :disabled="switching" @click="discardWorkspaceChange">{{ t('qwDiscardSwitch') }}</v-btn>
          <v-btn data-testid="model-switch-save" color="primary" :loading="switching" @click="saveWorkspaceChange">{{ t('qwSaveSwitch') }}</v-btn>
        </v-card-actions>
        <v-alert v-if="workspaceError" type="error" variant="tonal">{{ workspaceError }}</v-alert>
      </v-card>
    </v-dialog>
    <v-dialog v-model="qwenCacheDialog" max-width="540">
      <v-card :title="qwen.family === 'moss' ? mossText(appStore.language, 'cacheNotReady') : t('qwCacheNotReady')">
        <v-card-text>{{ t('qwenChainHint', { path: String(qwen.draft?.cache.out ?? '') }) }}<pre class="text-caption" style="white-space: pre-wrap">{{ cacheProblem }}</pre></v-card-text>
        <v-alert v-if="qwen.error" type="error" variant="tonal">{{ qwen.error }}</v-alert>
        <v-card-actions class="flex-wrap"><v-btn @click="qwenCacheDialog = false">{{ t('dsCancel') }}</v-btn><v-btn to="/preprocess">{{ t('navPreprocess') }}</v-btn><v-spacer /><v-btn data-testid="qwen-cache-then-train" color="primary" :loading="qwen.busy" @click="chainQwenTraining">{{ t('qwenChain') }}</v-btn></v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import WorkspaceHeading from '../components/WorkspaceHeading.vue'
import { ref, reactive, computed, nextTick, onMounted, onBeforeUnmount, watch } from 'vue'
import { useRoute, useRouter, onBeforeRouteLeave } from 'vue-router'
import { useModelWorkspace, modelLabel, type ModelFamily } from '../stores/modelWorkspace'
import { useActiveModelWorkspace, useQwenWorkspace, useMossWorkspace } from '../stores/qwenWorkspace'
import { mossText } from '../i18n/modelWorkspace'
import QwenConfigFields from '../components/QwenConfigFields.vue'
import ConfigWorkspaceLayout from '../components/ConfigWorkspaceLayout.vue'
import ConfigToolbar from '../components/ConfigToolbar.vue'
import ConfigurationPicker from '../components/ConfigurationPicker.vue'
import { modelAlgorithmItems } from '../utils/modelConfig'
import RuntimeSelector from '../components/RuntimeSelector.vue'
import { useConfigStore } from '../stores/config'
import { useTaskStore } from '../stores/task'
import { useNotifyStore } from '../stores/notify'
import { useAppStore } from '../stores/app'
import { useI18n } from '../composables/useI18n'
import ConfigField from '../components/ConfigField.vue'
import PreviewPromptEditor from '../components/PreviewPromptEditor.vue'
import HelpPanel from '../components/HelpPanel.vue'
import { readPreprocessRun, writePreprocessRun } from '../composables/usePreprocessRunStorage'
import { matchesConfigSearch } from '../utils/configSearch'
import { useTrainingContinuation, type ContinuationCandidate } from '../composables/useTrainingContinuation'

const workspace = useModelWorkspace()
const qwen = useActiveModelWorkspace()
const lineWorkspaces = { qwen21: useQwenWorkspace(), moss: useMossWorkspace() }
const router = useRouter()
const route = useRoute()
const modelItems = [{ title: 'Anima', value: 'anima' }, { title: 'Qwen-Image 2.1', value: 'qwen21' }, { title: 'MOSS', value: 'moss' }]
const workspaceError = ref('')
const switching = ref(false)
type WorkspaceChange = { kind: 'model'; family: ModelFamily } | { kind: 'profile'; name: string }
  | { kind: 'method'; method: string } | { kind: 'preset'; name: string } | { kind: 'reload' }
  | { kind: 'leave'; resolve: (allow: boolean) => void }
const pendingChange = ref<WorkspaceChange | null>(null)
const workspaceDirty = computed(() => workspace.family !== 'anima' ? qwen.value.dirty : configStore.dirty || !!extraArgs.value)
const creatingVariant = ref(false)
const qwenCacheDialog = ref(false)
const cacheProblem = ref('')

const isAnima = computed(() => workspace.family === 'anima')
const lineLaunching = ref<'train' | 'test' | null>(null)
const toolbarBusy = computed(() => switching.value || creatingVariant.value || trainingLaunching.value || testLaunching.value || qwen.value.busy || qwen.value.loading || configStore.loading)
const workspaceMethod = computed(() => isAnima.value ? selectedMethod.value : qwen.value.configMethod)
const workspaceVariant = computed(() => isAnima.value ? selectedVariant.value : qwen.value.configProfile)
const workspaceReady = computed(() => Boolean(workspaceVariant.value) && (isAnima.value
  ? selectedVariant.value === configStore.variant && selectedPreset.value === configStore.preset
  : qwen.value.draft?.train.algo === workspaceMethod.value))
const workspaceMethodItems = computed(() => isAnima.value
  ? configStore.methods.map(value => ({ title: value, value })) : modelAlgorithmItems(qwen.value.schema))
const workspaceVariantItems = computed(() => isAnima.value ? variantItems.value
  : qwen.value.profiles.filter(name => qwen.value.profileAlgorithms[name] === workspaceMethod.value).map(value => ({ label: value, value })))
const workspacePickerItems = computed(() => workspaceMethod.value ? workspaceVariantItems.value
  : workspaceMethodItems.value.map(item => ({ value: item.value, label: item.title })))
const modelPresetLabel = computed(() => qwen.value.family === 'moss' ? mossText(appStore.language, 'hardware') : t('qwAutomaticHardware'))

async function applyWorkspaceChange(change: WorkspaceChange): Promise<void> {
  if (change.kind === 'leave') {
    change.resolve(true)
  } else if (change.kind === 'profile') {
    if (isAnima.value) {
      const previous = selectedVariant.value
      selectedVariant.value = change.name
      try { await onVariantChange() }
      catch (cause) { selectedVariant.value = previous; throw cause }
    }
    else await qwen.value.selectConfigProfile(change.name)
  } else if (change.kind === 'method') {
    if (isAnima.value) { selectedMethod.value = change.method; await onMethodChange(change.method) }
    else qwen.value.selectConfigMethod(change.method)
  } else if (change.kind === 'preset') {
    const previous = selectedPreset.value
    selectedPreset.value = change.name
    try { await onVariantChange() }
    catch (cause) { selectedPreset.value = previous; throw cause }
  } else if (change.kind === 'reload') {
    if (isAnima.value) await loadConfig()
    else await qwen.value.selectConfigProfile(qwen.value.configProfile)
  } else {
    if (change.family !== 'anima') {
      const target = lineWorkspaces[change.family]
      await target.ensureLoaded()
      await target.loadProfileCatalog()
    }
    workspace.select(change.family)
  }
  pendingChange.value = null
}
function cancelWorkspaceChange(): void {
  if (pendingChange.value?.kind === 'leave') pendingChange.value.resolve(false)
  pendingChange.value = null
}
onBeforeRouteLeave(() => {
  if (toolbarBusy.value) return false
  if (!workspaceDirty.value) return true
  cancelWorkspaceChange()
  workspaceError.value = ''
  return new Promise<boolean>(resolve => { pendingChange.value = { kind: 'leave', resolve } })
})
function beforeUnload(event: BeforeUnloadEvent): void {
  if (workspaceDirty.value) { event.preventDefault(); event.returnValue = '' }
}
onMounted(() => window.addEventListener('beforeunload', beforeUnload))
onBeforeUnmount(() => {
  window.removeEventListener('beforeunload', beforeUnload)
  cancelWorkspaceChange()
})
async function requestWorkspaceChange(change: WorkspaceChange): Promise<void> {
  if (toolbarBusy.value) return
  workspaceError.value = ''
  if (workspaceDirty.value) { pendingChange.value = change; return }
  switching.value = true
  try { await applyWorkspaceChange(change) }
  catch (err) { workspaceError.value = err instanceof Error ? err.message : String(err) }
  finally { switching.value = false }
}
async function requestModelChange(family: ModelFamily): Promise<void> {
  if (family !== workspace.family) await requestWorkspaceChange({ kind: 'model', family })
}
async function requestMethodChange(method: string): Promise<void> {
  if (method !== workspaceMethod.value) await requestWorkspaceChange({ kind: 'method', method })
}
async function requestVariantChange(name: string): Promise<void> {
  await requestWorkspaceChange({ kind: 'profile', name })
}
async function requestPresetChange(name: string): Promise<void> {
  if (isAnima.value) await requestWorkspaceChange({ kind: 'preset', name })
}
async function chooseWorkspaceConfiguration(value: string): Promise<void> {
  if (workspaceMethod.value) await requestVariantChange(value)
  else await requestMethodChange(value)
}
function openWorkspaceVariant(): void {
  if (!workspaceMethod.value || toolbarBusy.value) return
  openCreateVariant()
  if (isAnima.value) configStore.error = ''
  else qwen.value.error = ''
}
async function saveWorkspace(): Promise<void> {
  if (!workspaceReady.value || toolbarBusy.value) return
  if (isAnima.value) await onSave()
  else await qwen.value.perform(qwen.value.save)
}
async function discardWorkspaceChange(): Promise<void> {
  if (!pendingChange.value) return
  switching.value = true
  try {
    if (workspace.family !== 'anima') qwen.value.discard()
    else { await loadConfig(); extraArgs.value = '' }
    await applyWorkspaceChange(pendingChange.value)
  } catch (err) { workspaceError.value = err instanceof Error ? err.message : String(err) }
  finally { switching.value = false }
}
async function saveWorkspaceChange(): Promise<void> {
  if (!pendingChange.value) return
  switching.value = true
  try {
    if (workspace.family !== 'anima') await qwen.value.save()
    else {
      await onSave()
      if (configStore.dirty || extraArgs.value) throw new Error(configStore.error || t('qwSaveRequired'))
    }
    await applyWorkspaceChange(pendingChange.value)
  } catch (err) { workspaceError.value = err instanceof Error ? err.message : String(err) }
  finally { switching.value = false }
}
async function confirmCreateWorkspaceVariant(): Promise<void> {
  const name = newVariantName.value.trim()
  if (creatingVariant.value || !name || variantNameRule(name) !== true) return
  creatingVariant.value = true
  try {
    if (isAnima.value) await confirmCreateVariant()
    else await qwen.value.perform(async () => {
      await qwen.value.createConfigProfile(name, seedFromCurrent.value)
      showCreateVariantDlg.value = false
      newVariantName.value = ''
      notify.show(t('cfgNewVariantCreated', { name }), 'success')
    })
  } finally { creatingVariant.value = false }
}
async function openQwenTask(taskId: string): Promise<void> {
  await taskStore.fetchTasks()
  await router.push({ path: '/tasks', query: { task: taskId } })
}
async function startQwenTraining(): Promise<void> {
  if (!workspaceReady.value || toolbarBusy.value) return
  lineLaunching.value = 'train'
  try { await qwen.value.perform(async () => {
    if (qwen.value.dirty) await qwen.value.save()
    const status = await qwen.value.preflight()
    if (!status.ready) {
      cacheProblem.value = [status.directory, ...status.errors, t('qwCacheSummary', { pairs: status.pairs, text: status.missing_text, latents: status.missing_latents })].join('\n')
      qwenCacheDialog.value = true
      return
    }
    await openQwenTask(await qwen.value.submit('train', {}))
  }) } finally { lineLaunching.value = null }
}
async function startQwenTest(): Promise<void> {
  if (!workspaceReady.value || toolbarBusy.value) return
  lineLaunching.value = 'test'
  try { await qwen.value.perform(async () => {
    if (qwen.value.dirty) await qwen.value.save()
    await openQwenTask(await qwen.value.submit('generate', {}))
  }) } finally { lineLaunching.value = null }
}
async function chainQwenTraining(): Promise<void> {
  await qwen.value.perform(async () => {
    const id = await qwen.value.chain()
    qwenCacheDialog.value = false
    await openQwenTask(id)
  })
}

const configStore = useConfigStore()
const taskStore = useTaskStore()
const notify = useNotifyStore()
const appStore = useAppStore()
const { t } = useI18n()
const fieldSearch = ref<string | null>('')
const expandedGroups = ref<string[]>([])
const showGuide = ref(true)
const filteredBasic = computed(() => selectedVariant.value
  ? configStore.basicFields.filter(field => matchesConfigSearch(field, fieldSearch.value)) : [])
const filteredAdvanced = computed(() => {
  if (!selectedVariant.value) return {}
  return Object.fromEntries(Object.entries(configStore.groupedAdvanced)
    .map(([group, fields]) => [group, fields.filter(field => matchesConfigSearch(field, fieldSearch.value))] as const)
    .filter(([, fields]) => fields.length))
})
const visibleFieldCount = computed(() => filteredBasic.value.length
  + Object.values(filteredAdvanced.value).reduce((total, fields) => total + fields.length, 0))
function onExpandedGroups(value: unknown) {
  if (!fieldSearch.value && Array.isArray(value)) expandedGroups.value = value
}

const animaBlockerKeys: Record<string, string> = {
  REPA: 'cfgModelBlockerRepa',
  'VR loss': 'cfgModelBlockerVr',
  LoKr: 'cfgModelBlockerLokr',
  GLoKr: 'cfgModelBlockerGlokr',
  LoHa: 'cfgModelBlockerLoha',
  ControlNet: 'cfgModelBlockerControl',
  EasyControl: 'cfgModelBlockerEasycontrol',
  BYG: 'cfgModelBlockerByg',
  SoftTokens: 'cfgModelBlockerSoftTokens',
  'Soft Tokens': 'cfgModelBlockerSoftTokens',
  'Hydra/route_per_layer': 'cfgModelBlockerHydra',
  'Hydra/router_source': 'cfgModelBlockerHydra',
  'Turbo step experts': 'cfgModelBlockerTurbo',
  train_llm_adapter: 'cfgModelBlockerTextEncoder',
  'text encoder training': 'cfgModelBlockerTextEncoder',
  'text-encoder-only training': 'cfgModelBlockerTextEncoder',
}

function localizeAnimaBlocker(blocker: unknown): string {
  const detail = String(blocker)
  const directKey = animaBlockerKeys[detail]
  if (directKey) return t(directKey)
  if (detail.startsWith('uncertified 40-block checkpoint')) {
    return t('cfgModelBlockerUncertified', { detail })
  }
  if (detail.startsWith('profile must')) {
    return t('cfgModelBlockerUnsupportedProfile')
  }
  if (detail.includes('channel stats')) {
    return t('cfgModelBlockerChannelStats', { detail })
  }
  return t('cfgModelBlockerDetail', { detail })
}

const selectedMethod = ref('')
const selectedVariant = ref('')
const selectedPreset = ref('default')

function selectedPreprocessRun(): string | null {
  return readPreprocessRun()
}

function storePreprocessRun(manifest: string | null) {
  writePreprocessRun(manifest)
}

const trainingLaunching = ref(false)
const testLaunching = ref(false)
const preprocessRunning = ref(false)

// Extra args state
const extraArgs = ref('')
const showExtraArgs = ref(false)
const continuation = reactive(useTrainingContinuation(selectedPreset, extraArgs))
// current_epoch is the one-based epoch in progress, not epochs completed —
// an interrupted task must read as "stopped inside epoch N", never "N done".
function continuationProgress(candidate: ContinuationCandidate): string {
  if (candidate.state === 'done') {
    return candidate.budget_key === 'max_train_epochs'
      ? t('cfgContinuationDoneEpochs', { epochs: candidate.target })
      : t('cfgContinuationDoneSteps', { step: candidate.step })
  }
  return candidate.epoch > 0
    ? t('cfgContinuationInterrupted', { epoch: candidate.epoch, step: candidate.step })
    : t('cfgContinuationSteps', { step: candidate.step })
}
const trainingTaskItems = computed(() => continuation.candidates.map(candidate => ({
  value: candidate.task_id,
  title: `${candidate.name} · ${new Date(candidate.submitted_at * 1000).toLocaleDateString()} · ${t(`cfgTrainingState_${candidate.state}`)} · ${continuationProgress(candidate)} · ${candidate.job_id.slice(-6)}`,
  props: { disabled: !candidate.available, subtitle: continuation.reasonLabel(candidate) || undefined },
})))

// Experimental feature tracking
const isExperimental = ref(false)

// Help panel state
const helpPanelRef = ref<InstanceType<typeof HelpPanel> | null>(null)
const fieldHelpData = ref<Record<string, string>>({})
const guideHtml = ref('')

// Dialog state
const showNoCacheDlg = ref(false)
const showLegacyLokrDlg = ref(false)
const prelaunchResult = ref<any>(null)

// Conditioning interception — block conditioning_data_dir / cond_cache_dir
// on variants whose family isn't conditioning-capable (controlnet/easycontrol).
// These keys only reach the editor via the Extra Args textarea or a manual
// flat-key edit; the default merge never surfaces them.
// MUST stay in sync with webui/services/config_service.py::CONDITIONING_CAPABLE_FAMILIES
// (the backend is the authoritative gate; this frontend check is UX-layer only).
const CONDITIONING_FAMILIES = new Set(['controlnet', 'easycontrol'])
const CONDITIONING_KEYS = ['conditioning_data_dir', 'cond_cache_dir'] as const
const showConditioningDlg = ref(false)
const conditioningCtx = ref<{ keys: string[]; action: 'save' | 'train' }>({
  keys: [],
  action: 'save',
})

// Preset management
const showCreatePresetDlg = ref(false)
const newPresetName = ref('')

// Variant management
const showCreateVariantDlg = ref(false)
const newVariantName = ref('')
const seedFromCurrent = ref(true)

// WandB settings
const wandb = ref({
  enabled: false,
  project: 'anima-lora',
  run_name: '',
  api_key: '',
  log_every_n_steps: 50,
  log_gradients: true,
  log_weights: true,
  log_checkpoint_artifact: true,
})
const wandbPanel = ref(false)

async function loadWandbSettings() {
  try {
    const res = await fetch('/api/config/wandb-settings')
    if (res.ok) wandb.value = await res.json()
  } catch { /* ignore */ }
}

async function saveWandbSettings() {
  try {
    await fetch('/api/config/wandb-settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(wandb.value),
    })
    notify.show(t('cfgWandbSaved'), 'success')
  } catch { /* ignore */ }
}

const variantNameRule = (v: string) => {
  if (!v) return true
  return /^[A-Za-z0-9_-]+$/.test(v) ? true : t('cfgNewVariantNameInvalid')
}

const isCustomPreset = computed(() => {
  const builtin = ['default', 'low_vram', 'graft', 'half', 'quarter', 'tenth', 'debug']
  return selectedPreset.value && !builtin.includes(selectedPreset.value)
})

const variantItems = computed(() =>
  configStore.variants.map(v => ({
    value: v,
    label: configStore.variantLabels[v] || v,
  }))
)

const LEGACY_LOKR_DIM_RE = /^\s*network_dim\s*=\s*114514\s*(?:#.*)?$/m
const LEGACY_LOKR_ALLOW_RE = /^\s*lokr_allow_legacy_dim\s*=\s*true\s*(?:#.*)?$/mi
const LOKR_MIGRATION_OVERRIDE_RE = /^\s*(?:network_dim|network_alpha|lokr_full_factor|decompose_both)\s*=/i

function isLegacyLokrSentinel(): boolean {
  if (configStore.getFieldValue('use_lokr') !== true) return false
  const explicitlyAllowed =
    configStore.getFieldValue('lokr_allow_legacy_dim') === true
    || LEGACY_LOKR_ALLOW_RE.test(extraArgs.value)
  if (explicitlyAllowed) return false
  return Number(configStore.getFieldValue('network_dim')) === 114514
    || LEGACY_LOKR_DIM_RE.test(extraArgs.value)
}

function promptLegacyLokrSentinel(): boolean {
  if (!isLegacyLokrSentinel()) return false
  showLegacyLokrDlg.value = true
  return true
}

function onConfigFieldUpdate(key: string, value: unknown) {
  if (continuation.active && (!continuation.ready || key !== continuation.budgetKey)) return
  configStore.setFieldValue(key, value)
  if (key === 'network_dim' && Number(value) === 114514) {
    promptLegacyLokrSentinel()
  }
}

function applyLokrFullFactorMigration() {
  configStore.setFieldValue('network_dim', 32)
  configStore.setFieldValue('network_alpha', 32)
  configStore.setFieldValue('lokr_full_factor', true)
  configStore.setFieldValue('decompose_both', false)
  extraArgs.value = extraArgs.value
    .split('\n')
    .filter(line => !LOKR_MIGRATION_OVERRIDE_RE.test(line))
    .join('\n')
    .trim()
  showLegacyLokrDlg.value = false
}

// ── Navigation ────────────────────────────────────────────────

async function onMethodChange(method: string) {
  continuation.exit()
  selectedVariant.value = ''
  configStore.variant = ''
  await configStore.fetchVariants(method)
}

async function onVariantChange() {
  const requestedPreset = selectedPreset.value
  continuation.exit()
  selectedPreset.value = requestedPreset
  if (selectedVariant.value) {
    await loadConfig()
  }
}

async function loadConfig() {
  if (continuation.selected) {
    // Entering task mode: drop stale new-training checks so an old
    // compatibility alert cannot hang over the pinned snapshot form.
    prelaunchResult.value = null
    isExperimental.value = false
    await continuation.select(continuation.selected)
    await continuation.fetchCandidates(selectedVariant.value)
    return
  }
  if (selectedVariant.value) {
    prelaunchResult.value = null
    await configStore.fetchMerged(selectedVariant.value, selectedPreset.value)
    await fetchFieldHelp()
    await checkExperimental()
    await continuation.fetchCandidates(selectedVariant.value)
  }
}

async function fetchFieldHelp() {
  if (!selectedVariant.value) return
  try {
    const lang = appStore.language
    const res = await fetch(`/api/config/field-help?variant=${encodeURIComponent(selectedVariant.value)}&lang=${encodeURIComponent(lang)}`)
    if (res.ok) {
      const data = await res.json()
      fieldHelpData.value = data.field_help || {}
      guideHtml.value = data.guide_html || ''
    }
  } catch {
    // Silently ignore — help is non-critical
  }
}

async function checkExperimental() {
  if (!selectedVariant.value) {
    isExperimental.value = false
    return
  }
  try {
    const res = await fetch(`/api/config/variant-meta?variant=${encodeURIComponent(selectedVariant.value)}`)
    if (res.ok) {
      const data = await res.json()
      isExperimental.value = !!data.experimental
    }
  } catch {
    isExperimental.value = false
  }
}

/**
 * Detect conditioning keys (conditioning_data_dir / cond_cache_dir) that would
 * land on a subset but the current variant's family isn't conditioning-capable.
 *
 * Returns the list of offending key names (empty = pass). On a fetch error or
 * unknown family, defaults to pass-through (mirrors checkExperimental's
 * tolerance) — the backend's dreambooth.py FileNotFoundError is the backstop.
 *
 * Scans two sources:
 *   - extraArgsText: free-form TOML text from the Extra Args textarea
 *     (line-regex match — the backend does the real TOML parse on save)
 *   - configStore.editedValues: flat form overrides (defensive; the standard
 *     form doesn't expose these keys, but a custom field could)
 */
async function checkConditioning(extraArgsText?: string): Promise<string[]> {
  if (!selectedVariant.value) return []
  let family: string | undefined
  try {
    const res = await fetch(
      `/api/config/variant-meta?variant=${encodeURIComponent(selectedVariant.value)}`,
    )
    if (res.ok) family = (await res.json()).family
  } catch {
    // Network/parse failure → don't block; backend remains the backstop.
    return []
  }
  if (family && CONDITIONING_FAMILIES.has(family)) return []

  const hits = new Set<string>()
  if (extraArgsText) {
    for (const k of CONDITIONING_KEYS) {
      const kre = new RegExp(`^\\s*${k}\\s*=`, 'm')
      if (kre.test(extraArgsText)) hits.add(k)
    }
  }
  for (const k of CONDITIONING_KEYS) {
    const v = configStore.editedValues[k]
    if (v !== undefined && v !== null && v !== '') hits.add(k)
  }
  return [...hits]
}

/** Strip conditioning key assignments from free-form TOML text. */
function stripConditioningFromExtraArgs(text: string): string {
  return text
    .split('\n')
    .filter((line) => {
      const m = line.match(/^\s*(conditioning_data_dir|cond_cache_dir)\s*=/)
      return !m
    })
    .join('\n')
    .trim()
}

async function onFieldHelp(key: string, origin: string) {
  showGuide.value = true
  await nextTick()
  helpPanelRef.value?.showFieldHelp(key, origin)
  if (window.matchMedia('(max-width: 1279px)').matches) {
    document.querySelector('.config-reference')?.scrollIntoView({
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
      block: 'start',
    })
  }
}

onMounted(async () => {
  if (route.query.model === 'qwen21' || route.query.model === 'moss') await requestModelChange(route.query.model)
  if (workspace.family !== 'anima' && !Object.keys(qwen.value.profileAlgorithms).length) {
    await qwen.value.perform(qwen.value.loadProfileCatalog)
  }
  await Promise.all([
    configStore.fetchMethods(),
    configStore.fetchPresets(),
    loadWandbSettings(),
  ])

  // Restore previous selections from store (survives page navigation)
  if (configStore.method) {
    selectedMethod.value = configStore.method
    await configStore.fetchVariants(configStore.method)
    if (configStore.variant) {
      selectedVariant.value = configStore.variant
      selectedPreset.value = configStore.preset
      try { await loadConfig() } catch { /* The store keeps the draft and exposes the error. */ }
    }
  }
})

// Refetch help data when language changes
watch(() => appStore.language, () => {
  if (selectedVariant.value) fetchFieldHelp()
})

// ── Save ───────────────────────────────────────────────────────

async function onSave() {
  if (continuation.active) {
    try { continuation.saveDraft(); notify.show(t('cfgContinuationDraftSaved'), 'success') }
    catch (e: any) { configStore.error = e.message }
    return
  }
  if (promptLegacyLokrSentinel()) return
  try {
    const args = extraArgs.value.trim() || undefined

    // Conditioning interception: block save when conditioning keys are present
    // on a non-conditioning variant. Prompt the user to remove them.
    const badKeys = await checkConditioning(args)
    if (badKeys.length) {
      conditioningCtx.value = { keys: badKeys, action: 'save' }
      showConditioningDlg.value = true
      return
    }

    await configStore.save(args)
    if (args) extraArgs.value = ''
    notify.show(t('notifyConfigSaved'), 'success')
  } catch (e: any) {
    notify.show(t('notifyConfigSaveFailed'), 'error')
  }
}

// ── Preset management ────────────────────────────────────────

function openCreatePreset() {
  newPresetName.value = ''
  showCreatePresetDlg.value = true
}

async function confirmCreatePreset() {
  const name = newPresetName.value.trim()
  if (!name) return
  const data: Record<string, unknown> = {}
  for (const f of configStore.fields) {
    if (f.is_virtual) continue
    if (f.key in configStore.editedValues) {
      data[f.key] = configStore.editedValues[f.key]
    } else if (f.origin !== 'base') {
      data[f.key] = f.value
    }
  }
  const ok = await configStore.createPreset(name, data)
  if (ok) {
    showCreatePresetDlg.value = false
    selectedPreset.value = name
    try { await loadConfig() } catch { /* The store exposes the load error. */ }
    notify.show(t('cfgPresetCreated', { name }), 'success')
  }
}

async function deleteCurrentPreset() {
  const name = selectedPreset.value
  if (!name || !isCustomPreset.value) return
  const ok = await configStore.deletePreset(name)
  if (ok) {
    selectedPreset.value = configStore.presets.includes('default') ? 'default' : (configStore.presets[0] || '')
    try { await loadConfig() } catch { /* The store exposes the load error. */ }
    notify.show(t('cfgPresetDeleted', { name }), 'success')
  }
}

// ── Variant management ──────────────────────────────────────

function openCreateVariant() {
  newVariantName.value = ''
  seedFromCurrent.value = workspaceReady.value
  showCreateVariantDlg.value = true
}

async function confirmCreateVariant() {
  const name = newVariantName.value.trim()
  if (!name) return
  try {
    const body: Record<string, unknown> = { name }
    if (seedFromCurrent.value && selectedVariant.value) {
      body.seed_from = selectedVariant.value
    }
    const res = await fetch('/api/config/variants', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      const data = await res.json()
      configStore.error = data.detail || 'Failed to create variant'
      return
    }
    const data = await res.json()
    configStore.variants = data.variants || []
    showCreateVariantDlg.value = false
    selectedVariant.value = `custom/${name}`
    await loadConfig()
    notify.show(t('cfgNewVariantCreated', { name }), 'success')
  } catch (e: any) {
    configStore.error = e.message
  }
}

// ── Prelaunch check ──────────────────────────────────────────

async function fetchPrelaunch(): Promise<any> {
  const params = new URLSearchParams({
    variant: selectedVariant.value,
    preset: selectedPreset.value,
  })
  const manifest = selectedPreprocessRun()
  if (manifest) params.set('preprocess_run', manifest)
  const res = await fetch(`/api/config/prelaunch-check?${params}`)
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail || `Prelaunch check failed: ${res.status}`)
  }
  return res.json()
}

function acceptsPrelaunch(result: any): boolean {
  prelaunchResult.value = result
  return result.compatibility?.supported !== false
}

async function autoSaveIfDirty() {
  if (continuation.active) return
  if (configStore.dirty || extraArgs.value.trim()) {
    const args = extraArgs.value.trim() || undefined
    await configStore.save(args)
    if (args) extraArgs.value = ''
  }
}

// ── Training launch ──────────────────────────────────────────

async function startTraining() {
  if (!selectedVariant.value) return
  if (continuation.active) {
    configStore.error = ''
    trainingLaunching.value = true
    try {
      await continuation.submit()
      await taskStore.fetchTasks()
      await taskStore.fetchQueueStatus()
      notify.show(t('notifyTrainingLaunched'), 'success')
    } catch (e: any) { configStore.error = e.message }
    finally { trainingLaunching.value = false }
    return
  }
  if (promptLegacyLokrSentinel()) return
  trainingLaunching.value = true
  try {
    await autoSaveIfDirty()

    // Conditioning interception (post-save): re-check the persisted config.
    // autoSaveIfDirty already cleared extraArgs on save, so a hit here means
    // the keys live in the variant TOML on disk (e.g. a prior unguarded save
    // or a hand-edited file). editedValues may still carry them if the user
    // typed but didn't trigger a save path that stripped them.
    const badKeys = await checkConditioning()
    if (badKeys.length) {
      conditioningCtx.value = { keys: badKeys, action: 'train' }
      showConditioningDlg.value = true
      return
    }

    const result = await fetchPrelaunch()
    if (!acceptsPrelaunch(result)) return

    if (!result.has_cache) {
      showNoCacheDlg.value = true
      return
    }

    await launchTrainingTask()
  } catch (e: any) {
    configStore.error = e.message
  } finally {
    trainingLaunching.value = false
  }
}

async function launchTrainingTask() {
  const env: Record<string, string> = { PRESET: selectedPreset.value, ANIMA_TRAIN_FRESH: '1' }
  const args = [selectedVariant.value]
  const manifest = selectedPreprocessRun()
  if (manifest) args.push('--preprocess_run', manifest)

  // Inject wandb env vars when enabled
  if (wandb.value.enabled) {
    env['WANDB_ENABLED'] = '1'
    if (wandb.value.project) env['WANDB_PROJECT'] = wandb.value.project
    if (wandb.value.run_name) env['WANDB_RUN_NAME'] = wandb.value.run_name
    if (wandb.value.api_key) env['WANDB_API_KEY'] = wandb.value.api_key
    env['WANDB_LOG_EVERY_N'] = String(wandb.value.log_every_n_steps || 50)
    if (wandb.value.log_gradients) env['WANDB_LOG_GRADIENTS'] = '1'
    if (wandb.value.log_weights) env['WANDB_LOG_WEIGHTS'] = '1'
    if (wandb.value.log_checkpoint_artifact) env['WANDB_LOG_ARTIFACT'] = '1'
  }

  const taskId = await taskStore.startTask('lora-gui', args, env, true)
  if (taskId) {
    notify.show(t('notifyTrainingLaunched'), 'success')
  } else {
    notify.show(t('notifyTaskStartFailed', { command: t('cfgTrain') }), 'error')
  }
}

/**
 * Confirmation callback for the conditioning interception dialog.
 *
 * 'save'  : strip the keys from extraArgs, re-run the original save.
 * 'train' : strip + re-save (autoSaveIfDirty), then resume the launch flow
 *           (prelaunch → launch). Keys that already live in the on-disk
 *           variant TOML but aren't in extraArgs can't be auto-stripped here
 *           — the user is told to remove them manually (see dialog body).
 */
async function confirmRemoveConditioning() {
  showConditioningDlg.value = false
  const action = conditioningCtx.value.action

  // Strip from the Extra Args textarea (the only writable source we control).
  if (extraArgs.value) {
    extraArgs.value = stripConditioningFromExtraArgs(extraArgs.value)
  }
  // Drop any flat editedValues entries for the conditioning keys.
  for (const k of CONDITIONING_KEYS) {
    if (k in configStore.editedValues) delete configStore.editedValues[k]
  }

  try {
    if (action === 'save') {
      const args = extraArgs.value.trim() || undefined
      await configStore.save(args)
      if (args) extraArgs.value = ''
      notify.show(t('notifyConfigSaved'), 'success')
    } else {
      // 'train' — persist the stripped state, then resume the launch flow.
      await autoSaveIfDirty()
      const result = await fetchPrelaunch()
      if (!acceptsPrelaunch(result)) return
      if (!result.has_cache) {
        showNoCacheDlg.value = true
      } else {
        await launchTrainingTask()
      }
    }
  } catch (e: any) {
    if (action === 'save') notify.show(t('notifyConfigSaveFailed'), 'error')
    else configStore.error = e.message
  } finally {
    if (action === 'train') trainingLaunching.value = false
  }
}

async function runPreprocessThenTrain() {
  showNoCacheDlg.value = false
  preprocessRunning.value = true
  try {
    const selectedBeforeRun = selectedPreprocessRun()
    const env: Record<string, string> = {
      PRESET: selectedPreset.value,
      METHOD: selectedVariant.value,
      METHODS_SUBDIR: 'gui-methods',
    }
    if (selectedBeforeRun) env.PREPROCESS_RUN = selectedBeforeRun
    const taskId = await taskStore.startTask('preprocess', [], env)
    if (!taskId) {
      configStore.error = 'Failed to start preprocessing'
      return
    }
    await waitForTask(taskId)
    if (!selectedBeforeRun) await adoptLatestPreprocessRun()
    const result = await fetchPrelaunch()
    if (!acceptsPrelaunch(result)) return
    if (result.has_cache) {
      await launchTrainingTask()
    }
  } catch (e: any) {
    configStore.error = e.message
  } finally {
    preprocessRunning.value = false
    trainingLaunching.value = false
  }
}

async function adoptLatestPreprocessRun() {
  try {
    const common = new URLSearchParams({
      variant: selectedVariant.value,
      preset: selectedPreset.value,
    })
    const pathRes = await fetch(`/api/preprocess/paths?${common}`)
    if (!pathRes.ok) return
    const paths = await pathRes.json()
    if (paths.source_image_dir) common.set('source', paths.source_image_dir)
    const runsRes = await fetch(`/api/preprocess/runs?${common}`)
    if (!runsRes.ok) return
    const runs = await runsRes.json()
    const latest = Array.isArray(runs)
      ? runs.find(run => run?.complete && run?.status === 'ready')
      : null
    storePreprocessRun(latest?.manifest ?? null)
  } catch { /* prelaunch will surface a missing cache */ }
}

// ── Test inference ────────────────────────────────────────────

async function startTest() {
  if (!selectedVariant.value) return
  testLaunching.value = true
  try {
    await autoSaveIfDirty()
    const meta = await fetch(`/api/config/variant-meta?variant=${selectedVariant.value}`)
    const metaData = await meta.json()
    let command = 'test'
    const family = metaData.family || ''
    if (family === 'hydralora' || family === 'fera') command = 'test-hydra'
    else if (family === 'easycontrol') command = 'test-easycontrol'
    else if (family === 'chimera') command = 'test-hydra'

    // Pass test prompt/negative_prompt from config so the backend uses
    // the user's custom values instead of the hardcoded defaults.
    const args: string[] = []
    const prompt = configStore.getFieldValue('test_prompt')
    if (prompt) args.push('--prompt', String(prompt))
    const negPrompt = configStore.getFieldValue('test_negative_prompt')
    if (negPrompt) args.push('--negative_prompt', String(negPrompt))

    const taskId = await taskStore.startTask(command, args)
    if (taskId) {
      notify.show(t('notifyTestLaunched'), 'success')
    } else {
      notify.show(t('notifyTaskStartFailed', { command: t('cfgTest') }), 'error')
    }
  } catch (e: any) {
    configStore.error = e.message
  } finally {
    testLaunching.value = false
  }
}

// ── Utilities ─────────────────────────────────────────────────

function waitForTask(taskId: string): Promise<void> {
  return new Promise((resolve, reject) => {
    // Poll the single-task endpoint, not the full task list — fetching
    // every task (up to 500) every 2s just to watch one state flag.
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/tasks/${taskId}`)
        if (!res.ok) return
        const task = await res.json()
        if (task.state === 'success') {
          clearInterval(interval)
          resolve()
        } else if (task.state === 'failed' || task.state === 'cancelled') {
          clearInterval(interval)
          reject(new Error(`Task ${task.state}`))
        }
      } catch {
        // transient network error — retry on the next tick
      }
    }, 2000)
  })
}
</script>

<style scoped>
.lokr-migration-preview {
  margin-top: 12px;
  padding: 12px;
  overflow-x: auto;
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 4px;
  background: rgba(var(--v-theme-surface-variant), 0.45);
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
}

.config-editor {
  flex: 1 1 0;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
}

.config-dirty {
  display: flex;
  align-items: center;
  color: rgb(var(--v-theme-warning));
  font-size: 12px;
}
.config-section {
  background: transparent;
  border-radius: 0 !important;
  border-bottom: 1px solid var(--border-subtle);
}
.config-section > :deep(.v-card-title) { padding: 0 0 22px; }
.config-section > :deep(.v-card-text) { padding: 0 0 20px; }
@media (max-width: 599px) {
  .config-tracking :deep(.v-expansion-panel-title) { padding-inline: 12px; flex-wrap: wrap; gap: 4px; }
}
</style>
