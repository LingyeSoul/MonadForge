/** Model-specific wording for the shared typed-request workbench. */
const mossEn = {
  sourcesHint: 'Select a MOSS model root containing MOSS-Image-Base/ and MOSS-Image-SFT/. Text encoder, VAE and tokenizer are resolved from the base variant; the selected model determines the DiT.',
  cacheHint: 'Build MOSS text embeddings and packed latent caches before training. Existing pairs are reused unless overwrite is enabled.',
  generateHint: 'Choose a model variant, prompt and optional adapter. The LoRA strength controls this generation; 0 disables the adapter.',
  autoHint: 'MOSS uses its own swap and activation-budget defaults. A swap count of -1 sizes from free VRAM. Adapter alpha follows the selected algorithm; inspect each field’s help for its default.',
  unsupported: 'MOSS uses its own trainer and supports LoRA, LoKr and LoHa adapters. Anima presets and Qwen component overrides do not apply.',
  cacheIsolation: 'MOSS caches are separate from Anima and Qwen caches. Model sources are edited in Configuration.',
  cacheNotReady: 'MOSS cache is not ready', hardware: 'MOSS parameter defaults',
}
const mossCn: Record<keyof typeof mossEn, string> = {
  sourcesHint: '选择包含 MOSS-Image-Base/ 与 MOSS-Image-SFT/ 的 MOSS 模型根目录。文本编码器、VAE 和 tokenizer 来自 base；所选模型变体决定 DiT。',
  cacheHint: '训练前生成 MOSS 文本嵌入与打包潜变量缓存；未启用覆盖时复用已有缓存对。',
  generateHint: '选择模型变体、提示词及可选适配器。LoRA 强度用于当前生成，填 0 可禁用适配器。',
  autoHint: 'MOSS 使用自己的换出层数与激活内存预算默认值；换出层数填 -1 时按可用显存估算。alpha 默认值随适配器算法变化，详见对应字段说明。',
  unsupported: 'MOSS 使用独立训练后端，支持 LoRA、LoKr 和 LoHa。Anima 硬件预设与 Qwen 单组件覆盖参数不适用于此模型。',
  cacheIsolation: 'MOSS 缓存与 Anima、Qwen 缓存独立；模型来源在配置页修改。',
  cacheNotReady: 'MOSS 缓存尚未就绪', hardware: 'MOSS 参数默认值',
}
export function mossText(language: string, key: keyof typeof mossEn): string {
  return language === 'cn' || language.startsWith('zh') ? mossCn[key] : mossEn[key]
}
const mossFields: Record<string, [string, string]> = {
  model: ['Model variant', '模型变体'], algo: ['Adapter algorithm', '适配器算法'], max_steps: ['Maximum steps', '最大步数'],
  caption_format: ['Caption format', '标注格式'], prompt_format: ['Prompt format', '提示词格式'], vae_dtype: ['VAE precision', 'VAE 精度'],
  max_text_tokens: ['Text token limit', '文本 token 上限'], lora_multiplier: ['Adapter strength', '适配器强度'], cfg: ['CFG scale', 'CFG 强度'],
  sampler: ['Sampler', '采样器'], uncond_skip: ['Unconditional skip rule', '无条件分支跳过规则'], example: ['Example prompt', '示例提示词'],
  batched: ['Batch conditional passes', '合并条件分支'], shift: ['Scheduler shift', '调度偏移'], decode_only: ['Decode existing latents only', '仅解码已有潜变量'],
  spectrum: ['Enable Spectrum', '启用 Spectrum'], spectrum_window_size: ['Spectrum window size', 'Spectrum 窗口大小'],
  spectrum_flex_window: ['Spectrum flexible window', 'Spectrum 弹性窗口'], spectrum_warmup: ['Spectrum warmup steps', 'Spectrum 预热步数'],
  spectrum_stop_caching_step: ['Spectrum cache stop step', 'Spectrum 停止缓存步数'], spectrum_w: ['Spectrum weight', 'Spectrum 权重'],
  spectrum_m: ['Spectrum polynomial degree', 'Spectrum 多项式阶数'], spectrum_lam: ['Spectrum regularization', 'Spectrum 正则强度'],
  factor: ['LoKr factor', 'LoKr 因子'], down_init: ['Down-projection initialization', '降维投影初始化'], caption_dropout: ['Caption dropout', '标注丢弃率'],
  sigma_mu: ['Noise sampling mean', '噪声采样均值'], sigma_s: ['Noise sampling deviation', '噪声采样标准差'], sigma_cutoff: ['Noise cutoff', '噪声截断'],
  sigma_uniform: ['Uniform noise mixture', '均匀噪声比例'], activation_memory_budget: ['Activation memory budget', '激活内存预算'], log_every: ['Log interval', '日志间隔'],
}
export function modelFieldLabel(family: string, language: string, name: string, fallback: string): string {
  const entry = family === 'moss' ? mossFields[name] : undefined
  if (entry) return entry[language === 'cn' || language.startsWith('zh') ? 1 : 0]
  return fallback === `qwenField_${name}` ? name.replace(/_/g, ' ') : fallback
}
