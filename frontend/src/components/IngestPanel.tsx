import { useState } from 'react'
import type { IngestCardState } from '../types'
import { browseImages, ingestImage } from '../api'
import IngestCard from './IngestCard'
import styles from './IngestPanel.module.css'

export default function IngestPanel() {
  const [selectedPaths, setSelectedPaths] = useState<string[]>([])
  const [cards, setCards] = useState<IngestCardState[]>([])
  const [running, setRunning] = useState(false)
  const [browsing, setBrowsing] = useState(false)

  async function handleBrowse() {
    setBrowsing(true)
    try {
      const paths = await browseImages()
      if (paths.length > 0) {
        setSelectedPaths((prev) => {
          const existing = new Set(prev)
          const newPaths = paths.filter((p) => !existing.has(p))
          return [...prev, ...newPaths]
        })
      }
    } catch {
      // dialog cancelled or error
    } finally {
      setBrowsing(false)
    }
  }

  function handleRemovePath(index: number) {
    setSelectedPaths((prev) => prev.filter((_, i) => i !== index))
  }

  function handleClearAll() {
    setSelectedPaths([])
  }

  function updateCard(index: number, update: Partial<IngestCardState>) {
    setCards((prev) => prev.map((c, i) => (i === index ? { ...c, ...update } : c)))
  }

  async function handleStart() {
    if (selectedPaths.length === 0) return

    const initial: IngestCardState[] = selectedPaths.map((p) => ({
      file_path: p,
      status: 'pending',
      image_id: null,
      caption: null,
      capture_timestamp: null,
      lat: null,
      lon: null,
      faces: [],
      error: null,
    }))
    setCards(initial)
    setRunning(true)

    const promises = selectedPaths.map(async (path, index) => {
      setCards((prev) =>
        prev.map((c, i) => (i === index ? { ...c, status: 'processing' as const } : c)),
      )

      try {
        const result = await ingestImage(path)
        setCards((prev) =>
          prev.map((c, i) =>
            i === index
              ? {
                  ...c,
                  status:
                    result.ingestion_status === 'ready'
                      ? ('ready' as const)
                      : result.ingestion_status === 'pending_labels'
                        ? ('pending_labels' as const)
                        : ('failed' as const),
                  image_id: result.image_id,
                  caption: result.caption,
                  capture_timestamp: result.capture_timestamp,
                  lat: result.lat,
                  lon: result.lon,
                  faces: result.faces,
                  error: result.error,
                }
              : c,
          ),
        )
      } catch {
        setCards((prev) =>
          prev.map((c, i) =>
            i === index ? { ...c, status: 'failed' as const, error: 'Network error or server unreachable' } : c,
          ),
        )
      }
    })

    await Promise.all(promises)
    setRunning(false)
  }

  return (
    <div>
      <section className={styles.panel}>
        <h2>Ingest Images</h2>
        <p className={styles.hint}>Browse to select images, then start ingestion.</p>

        <div className={styles.browseRow}>
          <button className={styles.browseBtn} onClick={handleBrowse} disabled={running || browsing}>
            {browsing ? 'Opening...' : 'Browse Images'}
          </button>
          {selectedPaths.length > 0 && (
            <button className={styles.clearBtn} onClick={handleClearAll} disabled={running}>
              Clear All
            </button>
          )}
        </div>

        {selectedPaths.length > 0 && (
          <div className={styles.pathList}>
            {selectedPaths.map((p, i) => (
              <div key={`${p}-${i}`} className={styles.pathItem}>
                <span className={styles.pathText}>{p}</span>
                {!running && (
                  <button className={styles.removeBtn} onClick={() => handleRemovePath(i)} title="Remove">
                    &times;
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        <button
          className={styles.startBtn}
          onClick={handleStart}
          disabled={running || selectedPaths.length === 0}
        >
          {running ? 'Ingesting...' : `Start Ingestion (${selectedPaths.length} image${selectedPaths.length !== 1 ? 's' : ''})`}
        </button>
      </section>

      {cards.length > 0 && (
        <section className={styles.panel}>
          <h2>
            Ingestion Results{' '}
            <span className={styles.count}>
              {cards.filter((c) => c.status === 'ready' || c.status === 'pending_labels').length}/{cards.length}
            </span>
          </h2>
          <div className={styles.cardGrid}>
            {cards.map((card, i) => (
              <IngestCard key={`${card.file_path}-${i}`} card={card} onUpdate={(u) => updateCard(i, u)} />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
