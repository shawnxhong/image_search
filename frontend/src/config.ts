export type ThumbnailSize = 'small' | 'medium' | 'large'

export const THUMBNAIL_PIXELS: Record<ThumbnailSize, number> = {
  small: 120,
  medium: 180,
  large: 260,
}

export const DEFAULT_THUMBNAIL_SIZE: ThumbnailSize = 'medium'
export const DEFAULT_MAX_RESULTS = 20
