import { useState, useCallback, useRef } from 'react'
import type { AgentStep, QueryMode, SearchResultItem } from './types'
import { type ThumbnailSize, DEFAULT_THUMBNAIL_SIZE } from './config'
import { search, searchTextStream } from './api'
import TabNav from './components/TabNav'
import LLMPanel, { type ModelStatusSnapshot } from './components/LLMPanel'
import SearchPanel from './components/SearchPanel'
import AgentLog from './components/AgentLog'
import ResultsSection from './components/ResultsSection'
import LibraryPanel from './components/LibraryPanel'
import IngestPanel from './components/IngestPanel'
import ImageLightbox from './components/ImageLightbox'
import styles from './App.module.css'

/** Check if models are ready for search. Returns warning message or null. */
function checkSearchModels(snap: ModelStatusSnapshot | null): string | null {
  if (!snap) return 'Model status unknown. Please wait for model status to load.'
  const missing: string[] = []
  if (!snap.llm.loaded) missing.push('LLM')
  if (!snap.models.embeddings.loaded) missing.push('Embeddings')
  if (missing.length === 0) return null
  return `Search requires ${missing.join(' and ')} to be loaded. Use "Search Mode" in Model Control to load them.`
}

/** Check if models are ready for ingestion. Returns warning message or null. */
function checkIngestModels(snap: ModelStatusSnapshot | null): string | null {
  if (!snap) return 'Model status unknown. Please wait for model status to load.'
  const missing: string[] = []
  if (!snap.models.vlm.loaded) missing.push('VL Captioner')
  if (!snap.models.embeddings.loaded) missing.push('Embeddings')
  if (!snap.models.face_detection.loaded) missing.push('Face Detection')
  if (missing.length === 0) return null
  return `Ingestion requires ${missing.join(', ')} to be loaded. Use "Ingest Mode" in Model Control to load them.`
}

export default function App() {
  const [activeTab, setActiveTab] = useState('search')
  const modelStatusRef = useRef<ModelStatusSnapshot | null>(null)

  // Search state
  const [solidResults, setSolidResults] = useState<SearchResultItem[]>([])
  const [softResults, setSoftResults] = useState<SearchResultItem[]>([])
  const [thumbSize, setThumbSize] = useState<ThumbnailSize>(DEFAULT_THUMBNAIL_SIZE)
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)

  // Agent activity
  const [agentSteps, setAgentSteps] = useState<AgentStep[]>([])

  // Lightbox
  const [lightboxPath, setLightboxPath] = useState<string | null>(null)
  const handleImageClick = useCallback((filePath: string) => setLightboxPath(filePath), [])

  const handleModelStatusChange = useCallback((snap: ModelStatusSnapshot) => {
    modelStatusRef.current = snap
  }, [])

  const getSearchWarning = useCallback((): string | null => {
    return checkSearchModels(modelStatusRef.current)
  }, [])

  const getIngestWarning = useCallback((): string | null => {
    return checkIngestModels(modelStatusRef.current)
  }, [])

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
    setAgentSteps([])
    setSolidResults([])
    setSoftResults([])

    try {
      if (mode === 'text') {
        // Use streaming endpoint for text search
        await searchTextStream(
          textQuery.trim(),
          topK,
          (step) => {
            setAgentSteps((prev) => [...prev, step])
          },
          (data) => {
            const solidIds = new Set(data.solid_results.map((r) => r.image_id))
            const dedupedSoft = data.soft_results.filter((r) => !solidIds.has(r.image_id))
            setSolidResults(data.solid_results.slice(0, topK))
            setSoftResults(dedupedSoft.slice(0, topK))
          },
        )
        setStatus('Search complete.')
      } else {
        // Image search uses the non-streaming endpoint
        const data = await search(mode, textQuery.trim(), imagePath.trim(), topK)
        const solidIds = new Set(data.solid_results.map((r) => r.image_id))
        const dedupedSoft = data.soft_results.filter((r) => !solidIds.has(r.image_id))
        setSolidResults(data.solid_results.slice(0, topK))
        setSoftResults(dedupedSoft.slice(0, topK))
        setStatus('Search complete.')
      }
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

      <LLMPanel activeTab={activeTab} onStatusChange={handleModelStatusChange} />

      <TabNav activeTab={activeTab} onTabChange={setActiveTab} />

      {activeTab === 'search' && (
        <>
          <SearchPanel onSearch={handleSearch} loading={loading} checkModels={getSearchWarning} />

          <AgentLog steps={agentSteps} />

          {status && <p className={styles.status}>{status}</p>}

          {searched && (
            <>
              <ResultsSection
                title="Solid Matches"
                items={solidResults}
                thumbSize={thumbSize}
                emptyText="No solid matches found."
                onImageClick={handleImageClick}
              />
              <ResultsSection
                title="Soft Matches"
                items={softResults}
                thumbSize={thumbSize}
                emptyText="No additional soft matches."
                onImageClick={handleImageClick}
              />
            </>
          )}
        </>
      )}

      {activeTab === 'library' && <LibraryPanel onImageClick={handleImageClick} />}

      {activeTab === 'ingest' && <IngestPanel checkModels={getIngestWarning} onImageClick={handleImageClick} />}

      {lightboxPath && (
        <ImageLightbox filePath={lightboxPath} onClose={() => setLightboxPath(null)} />
      )}
    </main>
  )
}
