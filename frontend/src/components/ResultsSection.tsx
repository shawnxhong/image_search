import type { SearchResultItem } from '../types'
import type { ThumbnailSize } from '../config'
import ImageCard from './ImageCard'
import styles from './ResultsSection.module.css'

interface ResultsSectionProps {
  title: string
  items: SearchResultItem[]
  thumbSize: ThumbnailSize
  emptyText: string
}

export default function ResultsSection({ title, items, thumbSize, emptyText }: ResultsSectionProps) {
  return (
    <section className={styles.panel}>
      <h2>
        {title} <span className={styles.count}>{items.length}</span>
      </h2>
      {items.length === 0 ? (
        <p className={styles.empty}>{emptyText}</p>
      ) : (
        <div className={styles.grid}>
          {items.map((item) => (
            <ImageCard key={item.image_id} item={item} thumbSize={thumbSize} />
          ))}
        </div>
      )}
    </section>
  )
}
