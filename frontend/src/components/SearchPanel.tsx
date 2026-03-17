import { useState, type KeyboardEvent } from 'react'
import type { QueryMode } from '../types'
import { type ThumbnailSize, DEFAULT_THUMBNAIL_SIZE, DEFAULT_MAX_RESULTS } from '../config'
import styles from './SearchPanel.module.css'

interface SearchPanelProps {
  onSearch: (mode: QueryMode, textQuery: string, imagePath: string, topK: number, thumbSize: ThumbnailSize) => void
  loading: boolean
  checkModels?: () => string | null
}

export default function SearchPanel({ onSearch, loading, checkModels }: SearchPanelProps) {
  const [mode, setMode] = useState<QueryMode>('text')
  const [textQuery, setTextQuery] = useState('')
  const [imagePath, setImagePath] = useState('')
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
    onSearch(mode, textQuery, imagePath, maxResults, thumbSize)
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter') handleSearch()
  }

  return (
    <section className={styles.panel}>
      <h2>Search</h2>

      <div className={styles.row}>
        <label className={styles.field}>
          Query mode
          <select value={mode} onChange={(e) => setMode(e.target.value as QueryMode)}>
            <option value="text">Text</option>
            <option value="image">Image</option>
            <option value="image+text">Image + Text</option>
          </select>
        </label>
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
          disabled={mode === 'image'}
        />
      </label>

      <label className={styles.field}>
        Image path (for image mode)
        <input
          type="text"
          placeholder="/path/to/query.jpg"
          value={imagePath}
          onChange={(e) => setImagePath(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={mode === 'text'}
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
