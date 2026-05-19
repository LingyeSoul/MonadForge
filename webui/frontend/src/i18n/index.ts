import en from './en'
import cn from './cn'

export type TranslationKey = keyof typeof en
export type TranslationMessages = Record<string, string>

const frontendLocales: Record<string, TranslationMessages> = { en, cn }

/** Get frontend translations for a given language, falling back to en */
export function getFrontendTranslations(lang: string): TranslationMessages {
  return frontendLocales[lang] ?? frontendLocales.en
}
