import type { MatchExplanation } from '../types'
import { type ThumbnailSize, THUMBNAIL_PIXELS } from '../config'
import styles from './ImageCard.module.css'

interface BaseProps {
  image_id: string
  file_path: string
  thumbSize: ThumbnailSize
  onImageClick?: (filePath: string) => void
}

interface SearchProps extends BaseProps {
  variant: 'search'
  score: number
  showScore: boolean
  caption: string | null
  capture_timestamp: string | null
  country: string | null
  state: string | null
  city: string | null
  explanation: MatchExplanation | null
}

interface LibraryProps extends BaseProps {
  variant: 'library'
  caption: string | null
  capture_timestamp: string | null
  country: string | null
  state: string | null
  city: string | null
}

export type ImageCardProps = SearchProps | LibraryProps

function formatDate(ts: string | null): string {
  if (!ts) return 'No date'
  const d = new Date(ts)
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

function formatLocation(city: string | null, state: string | null, country: string | null): string | null {
  const parts = [city, state, country].filter(Boolean)
  return parts.length > 0 ? parts.join(', ') : null
}

export default function ImageCard(props: ImageCardProps) {
  const px = THUMBNAIL_PIXELS[props.thumbSize]

  return (
    <article className={styles.card}>
      <img
        className={`${styles.thumb} ${props.onImageClick ? styles.clickable : ''}`}
        style={{ height: px }}
        alt={props.file_path}
        src={`/image-preview?path=${encodeURIComponent(props.file_path)}`}
        loading="lazy"
        onClick={props.onImageClick ? () => props.onImageClick!(props.file_path) : undefined}
      />
      <div className={styles.content}>
        {props.variant === 'search' && (
          <>
            <div className={styles.muted}>{props.file_path}</div>
            <div className={styles.caption}>{props.caption || 'No description'}</div>
            <p className={styles.reason}>{props.explanation?.reason || 'No explanation'}</p>
            {props.showScore && (
              <div className={styles.metaRow}>
                <span className={styles.metaLabel}>Score</span>
                <span>{props.score.toFixed(4)}</span>
              </div>
            )}
            <div className={styles.metaRow}>
              <span className={styles.metaLabel}>Time</span>
              <span>{formatDate(props.capture_timestamp)}</span>
            </div>
            <div className={styles.metaRow}>
              <span className={styles.metaLabel}>Place</span>
              <span>{formatLocation(props.city, props.state, props.country) || 'No place'}</span>
            </div>
          </>
        )}
        {props.variant === 'library' && (
          <>
            <div className={styles.date}>{formatDate(props.capture_timestamp)}</div>
            {formatLocation(props.city, props.state, props.country) && (
              <div className={styles.location}>{formatLocation(props.city, props.state, props.country)}</div>
            )}
            <div className={styles.caption}>{props.caption || 'No caption'}</div>
            <div className={styles.muted}>{props.file_path}</div>
          </>
        )}
      </div>
    </article>
  )
}
