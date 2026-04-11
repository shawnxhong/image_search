import type { SearchResultItem } from '../types'
import type { ThumbnailSize } from '../config'
import ImageCard from './ImageCard'
import styles from './ResultsSection.module.css'

interface ResultsSectionProps {
  title: string
  items: SearchResultItem[]
  thumbSize: ThumbnailSize
  showScore: boolean
  emptyText: string
  onImageClick?: (filePath: string) => void
}

export default function ResultsSection({ title, items, thumbSize, showScore, emptyText, onImageClick }: ResultsSectionProps) {
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
            <ImageCard
              key={item.image_id}
              variant="search"
              image_id={item.image_id}
              file_path={item.file_path}
              score={item.score}
              showScore={showScore}
              caption={item.caption}
              capture_timestamp={item.capture_timestamp}
              country={item.country}
              state={item.state}
              city={item.city}
              explanation={item.explanation}
              thumbSize={thumbSize}
              onImageClick={onImageClick}
            />
          ))}
        </div>
      )}
    </section>
  )
}
