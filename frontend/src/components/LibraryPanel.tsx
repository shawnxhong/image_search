import { useEffect, useState, useCallback } from 'react'
import type { LibraryImageItem } from '../types'
import { type ThumbnailSize, DEFAULT_THUMBNAIL_SIZE } from '../config'
import { fetchLibrary } from '../api'
import ImageCard from './ImageCard'
import styles from './LibraryPanel.module.css'

export default function LibraryPanel() {
  const [images, setImages] = useState<LibraryImageItem[]>([])
  const [total, setTotal] = useState(0)
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [thumbSize, setThumbSize] = useState<ThumbnailSize>(DEFAULT_THUMBNAIL_SIZE)

  const loadPage = useCallback(async (cursor?: string) => {
    setLoading(true)
    setError('')
    try {
      const data = await fetchLibrary(50, cursor)
      if (cursor) {
        setImages((prev) => [...prev, ...data.images])
      } else {
        setImages(data.images)
      }
      setTotal(data.total)
      setNextCursor(data.next_cursor)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load library')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadPage()
  }, [loadPage])

  function handleLoadMore() {
    if (nextCursor) loadPage(nextCursor)
  }

  return (
    <div>
      <section className={styles.panel}>
        <div className={styles.header}>
          <h2>
            Library <span className={styles.count}>{total}</span>
          </h2>
          <label className={styles.sizeControl}>
            Thumbnail size
            <select value={thumbSize} onChange={(e) => setThumbSize(e.target.value as ThumbnailSize)}>
              <option value="small">Small</option>
              <option value="medium">Medium</option>
              <option value="large">Large</option>
            </select>
          </label>
        </div>

        {error && <p className={styles.error}>{error}</p>}

        {!loading && images.length === 0 && !error && (
          <p className={styles.empty}>No images in the library yet. Use the Ingest tab to add images.</p>
        )}

        {images.length > 0 && (
          <div className={styles.grid}>
            {images.map((img) => (
              <ImageCard
                key={img.image_id}
                variant="library"
                image_id={img.image_id}
                file_path={img.file_path}
                caption={img.caption}
                capture_timestamp={img.capture_timestamp}
                country={img.country}
                state={img.state}
                city={img.city}
                thumbSize={thumbSize}
              />
            ))}
          </div>
        )}

        {loading && <p className={styles.loading}>Loading...</p>}

        {nextCursor && !loading && (
          <div className={styles.loadMoreRow}>
            <button className={styles.loadMoreBtn} onClick={handleLoadMore}>
              Load More ({images.length} of {total})
            </button>
          </div>
        )}
      </section>
    </div>
  )
}
