import type { SearchResultItem } from '../types'
import { type ThumbnailSize, THUMBNAIL_PIXELS } from '../config'
import styles from './ImageCard.module.css'

interface ImageCardProps {
  item: SearchResultItem
  thumbSize: ThumbnailSize
}

export default function ImageCard({ item, thumbSize }: ImageCardProps) {
  const px = THUMBNAIL_PIXELS[thumbSize]

  return (
    <article className={styles.card}>
      <img
        className={styles.thumb}
        style={{ height: px }}
        alt={item.file_path}
        src={`/image-preview?path=${encodeURIComponent(item.file_path)}`}
        loading="lazy"
      />
      <div className={styles.content}>
        <div><strong>Score:</strong> {item.score.toFixed(4)}</div>
        <div className={styles.muted}>{item.file_path}</div>
        <p className={styles.reason}>{item.explanation?.reason || 'No explanation'}</p>
      </div>
    </article>
  )
}
