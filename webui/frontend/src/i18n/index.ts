import en from './en'
import cn from './cn'
import ko from './ko'
import ja from './ja'

export type TranslationKey = keyof typeof en
export type TranslationMessages = Record<string, string>

const frontendLocales: Record<string, TranslationMessages> = { en, cn, ko, ja }

/** Get frontend translations for a given language, falling back to en */
export function getFrontendTranslations(lang: string): TranslationMessages {
  return frontendLocales[lang] ?? frontendLocales.en
}
