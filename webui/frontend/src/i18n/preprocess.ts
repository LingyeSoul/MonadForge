/** Shared wording for the preprocessing cards; model-specific capabilities stay in the schemas. */
const en = {
  pairs: 'Complete pairs', missingText: 'Missing text', missingLatents: 'Missing latents',
  textHint: 'Cache text encoder outputs from image captions. The VAE pass is skipped.',
  vaeHint: 'Resize images at their original aspect ratio and cache VAE latents. The text encoder pass is skipped.',
  staleStatus: 'Settings changed. Refresh to inspect the current cache paths.',
}
const cn: Record<keyof typeof en, string> = {
  pairs: '完整配对', missingText: '缺少文本', missingLatents: '缺少潜变量',
  textHint: '根据图片标注缓存文本编码器输出，跳过 VAE 阶段。',
  vaeHint: '保持原始宽高比缩放图片并缓存 VAE 潜变量，跳过文本编码阶段。',
  staleStatus: '设置已变更，请刷新以检查当前缓存路径。',
}
const ja: Record<keyof typeof en, string> = {
  pairs: '完全なペア', missingText: 'テキスト不足', missingLatents: '潜在不足',
  textHint: '画像キャプションからテキスト埋め込みをキャッシュします。VAE 処理はスキップします。',
  vaeHint: '元の縦横比で画像をリサイズして VAE 潜在をキャッシュします。テキスト処理はスキップします。',
  staleStatus: '設定が変更されました。更新して現在のキャッシュパスを確認してください。',
}
const ko: Record<keyof typeof en, string> = {
  pairs: '완전한 쌍', missingText: '텍스트 없음', missingLatents: '잠재 없음',
  textHint: '이미지 캡션에서 텍스트 임베딩을 캐시합니다. VAE 단계는 건너뜁니다.',
  vaeHint: '원래 종횡비로 이미지 크기를 조정하고 VAE 잠재를 캐시합니다. 텍스트 단계는 건너뜁니다.',
  staleStatus: '설정이 변경되었습니다. 새로 고침하여 현재 캐시 경로를 확인하세요.',
}
export function preprocessText(language: string, key: keyof typeof en): string {
  return (language === 'cn' || language.startsWith('zh') ? cn : language === 'ja' ? ja : language === 'ko' ? ko : en)[key]
}
