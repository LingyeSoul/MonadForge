// Wallpaper helpers: uploaded images are downscaled to a JPEG data URL before
// being persisted, so a 4K wallpaper cannot blow past the ~5MB localStorage
// quota. Blur is pre-baked into the displayed image with canvas — never a CSS
// filter — because full-viewport filtered layers trip content-culling bugs in
// Edge at fractional device scale factors (cards + text vanish).

const MAX_EDGE = 1920
// Blurred content is soft; a smaller canvas makes the filter pass cheap.
const BLUR_MAX_EDGE = 1280

/** Decode an image file and downscale it to a JPEG data URL. */
export function fileToWallpaperDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      URL.revokeObjectURL(objectUrl)
      try {
        resolve(drawScaled(img, img.naturalWidth, img.naturalHeight, Math.min(1, MAX_EDGE / Math.max(img.naturalWidth, img.naturalHeight)), 0))
      } catch (err) {
        reject(err)
      }
    }
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl)
      reject(new Error('image decode failed'))
    }
    img.src = objectUrl
  })
}

/**
 * Produce the image actually painted as wallpaper: `src` with `blur` px of
 * gaussian blur baked in. Returns `src` unchanged when blur is 0. Throws for
 * cross-origin sources the canvas is not allowed to read back — callers
 * should fall back to the sharp original.
 */
export async function makeWallpaperDisplay(src: string, blur: number): Promise<string> {
  if (blur <= 0) return src
  const img = await decodeImage(src)
  const w = img.naturalWidth
  const h = img.naturalHeight
  const scale = Math.min(1, BLUR_MAX_EDGE / Math.max(w, h))
  return drawScaled(img, w, h, scale, blur * scale)
}

function decodeImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('image load failed'))
    img.src = src
  })
}

function drawScaled(
  img: HTMLImageElement,
  naturalWidth: number,
  naturalHeight: number,
  scale: number,
  blurPx: number,
): string {
  const w = Math.max(1, Math.round(naturalWidth * scale))
  const h = Math.max(1, Math.round(naturalHeight * scale))
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas 2d context unavailable')
  // Overscan so the gaussian blur does not fade the image edges out.
  const pad = Math.min(64, Math.round(blurPx * 2))
  if (blurPx > 0) ctx.filter = `blur(${blurPx.toFixed(2)}px)`
  ctx.drawImage(img, -pad, -pad, w + pad * 2, h + pad * 2)
  return canvas.toDataURL('image/jpeg', 0.85)
}

/** Resolve once `src` is displayable; reject if the browser fails to load it. */
export function verifyImageSrc(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve()
    img.onerror = () => reject(new Error('image load failed'))
    img.src = src
  })
}
