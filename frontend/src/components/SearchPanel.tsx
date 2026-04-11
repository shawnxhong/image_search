import { useState, type KeyboardEvent } from 'react'
import { type ThumbnailSize, DEFAULT_THUMBNAIL_SIZE, DEFAULT_MAX_RESULTS } from '../config'
import styles from './SearchPanel.module.css'

interface SearchPanelProps {
  onSearch: (textQuery: string, topK: number, thumbSize: ThumbnailSize) => void
  loading: boolean
  checkModels?: () => string | null
}

export default function SearchPanel({ onSearch, loading, checkModels }: SearchPanelProps) {
  const [textQuery, setTextQuery] = useState('')
  const [maxResults, setMaxResults] = useState(DEFAULT_MAX_RESULTS)
  const [thumbSize, setThumbSize] = useState<ThumbnailSize>(DEFAULT_THUMBNAIL_SIZE)
  const [modelWarning, setModelWarning] = useState<string | null>(null)

  function handleSearch() {
    const warning = checkModels?.()
    if (warning) {
      setModelWarning(warning)
      return
    }
    setModelWarning(null)
    onSearch(textQuery, maxResults, thumbSize)
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter') handleSearch()
  }

  return (
    <section className={styles.panel}>
      <div className={styles.header}>
        <h2>Search</h2>
        <span className={styles.modeTag}>Text Search</span>
      </div>

      <div className={styles.row}>
        <label className={styles.field}>
          Max results per list
          <input
            type="number"
            min={1}
            max={100}
            value={maxResults}
            onChange={(e) => setMaxResults(Math.max(1, Number(e.target.value)))}
          />
        </label>
      </div>

      <label className={styles.field}>
        Text query
        <input
          type="text"
          placeholder="find photos with tom in it taken on a beach last year"
          value={textQuery}
          onChange={(e) => setTextQuery(e.target.value)}
          onKeyDown={handleKeyDown}
        />
      </label>

      {modelWarning && (
        <p className={styles.modelWarning}>
          {modelWarning}
          <button className={styles.dismissBtn} onClick={() => setModelWarning(null)}>Dismiss</button>
        </p>
      )}

      <div className={styles.row}>
        <label className={styles.field}>
          Thumbnail size
          <select value={thumbSize} onChange={(e) => setThumbSize(e.target.value as ThumbnailSize)}>
            <option value="small">Small</option>
            <option value="medium">Medium</option>
            <option value="large">Large</option>
          </select>
        </label>
        <button className={styles.searchBtn} onClick={handleSearch} disabled={loading}>
          {loading ? 'Searching...' : 'Run Search'}
        </button>
      </div>
    </section>
  )
}
