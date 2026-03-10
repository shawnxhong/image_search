import { useState } from 'react'
import type { QueryMode, SearchResultItem } from './types'
import { type ThumbnailSize, DEFAULT_THUMBNAIL_SIZE } from './config'
import { search } from './api'
import TabNav from './components/TabNav'
import SearchPanel from './components/SearchPanel'
import ResultsSection from './components/ResultsSection'
import IngestPanel from './components/IngestPanel'
import styles from './App.module.css'

export default function App() {
  const [activeTab, setActiveTab] = useState('search')

  // Search state
  const [solidResults, setSolidResults] = useState<SearchResultItem[]>([])
  const [softResults, setSoftResults] = useState<SearchResultItem[]>([])
  const [thumbSize, setThumbSize] = useState<ThumbnailSize>(DEFAULT_THUMBNAIL_SIZE)
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)

  async function handleSearch(
    mode: QueryMode,
    textQuery: string,
    imagePath: string,
    topK: number,
    size: ThumbnailSize,
  ) {
    if ((mode === 'text' || mode === 'image+text') && !textQuery.trim()) {
      setStatus('Please enter a text query.')
      return
    }
    if ((mode === 'image' || mode === 'image+text') && !imagePath.trim()) {
      setStatus('Please enter an image path.')
      return
    }

    setLoading(true)
    setStatus('Searching...')
    setThumbSize(size)

    try {
      const data = await search(mode, textQuery.trim(), imagePath.trim(), topK)

      const solidIds = new Set(data.solid_results.map((r) => r.image_id))
      const dedupedSoft = data.soft_results.filter((r) => !solidIds.has(r.image_id))

      setSolidResults(data.solid_results.slice(0, topK))
      setSoftResults(dedupedSoft.slice(0, topK))
      setStatus('Search complete.')
      setSearched(true)
    } catch (err) {
      setStatus(err instanceof Error ? err.message : 'Search failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className={styles.container}>
      <header className={styles.header}>
        <h1>Agentic Image Search</h1>
      </header>

      <TabNav activeTab={activeTab} onTabChange={setActiveTab} />

      {activeTab === 'search' && (
        <>
          <SearchPanel onSearch={handleSearch} loading={loading} />

          {status && <p className={styles.status}>{status}</p>}

          {searched && (
            <>
              <ResultsSection
                title="Solid Matches"
                items={solidResults}
                thumbSize={thumbSize}
                emptyText="No solid matches found."
              />
              <ResultsSection
                title="Soft Matches"
                items={softResults}
                thumbSize={thumbSize}
                emptyText="No additional soft matches."
              />
            </>
          )}
        </>
      )}

      {activeTab === 'ingest' && <IngestPanel />}
    </main>
  )
}
