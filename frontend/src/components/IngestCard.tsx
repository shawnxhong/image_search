import { useState } from 'react'
import type { IngestCardState, DetectedFace } from '../types'
import { updateFaces, dismissFace } from '../api'
import styles from './IngestCard.module.css'

interface IngestCardProps {
  card: IngestCardState
  onUpdate: (update: Partial<IngestCardState>) => void
}

const STATUS_DISPLAY: Record<string, { label: string; className: string }> = {
  pending: { label: 'Pending', className: 'pending' },
  processing: { label: 'Processing...', className: 'processing' },
  ready: { label: 'Ready', className: 'ready' },
  failed: { label: 'Failed', className: 'failed' },
}

function formatTimestamp(ts: string | null): string {
  if (!ts) return '—'
  const d = new Date(ts)
  return d.toLocaleString()
}

function formatGps(lat: number | null, lon: number | null): string {
  if (lat == null || lon == null) return '—'
  const latDir = lat >= 0 ? 'N' : 'S'
  const lonDir = lon >= 0 ? 'E' : 'W'
  return `${Math.abs(lat).toFixed(5)}°${latDir}, ${Math.abs(lon).toFixed(5)}°${lonDir}`
}

export default function IngestCard({ card, onUpdate }: IngestCardProps) {
  const [faceNames, setFaceNames] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {}
    for (const f of card.faces) {
      initial[f.face_id] = f.name || ''
    }
    return initial
  })
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [dismissing, setDismissing] = useState<Record<string, boolean>>({})

  const statusInfo = STATUS_DISPLAY[card.status] || STATUS_DISPLAY.pending
  const activeFaces = card.faces.filter((f) => !f.dismissed)
  const dismissedCount = card.faces.length - activeFaces.length

  async function handleSaveNames() {
    if (!card.image_id) return
    setSaving(true)
    setSaved(false)

    const entries = activeFaces
      .filter((f) => faceNames[f.face_id]?.trim())
      .map((f) => ({ face_id: f.face_id, name: faceNames[f.face_id].trim() }))

    try {
      await updateFaces(card.image_id, entries)
      const updatedFaces: DetectedFace[] = card.faces.map((f) => ({
        ...f,
        name: faceNames[f.face_id]?.trim() || f.name,
      }))
      onUpdate({ faces: updatedFaces })
      setSaved(true)
    } catch {
      // silently fail for demo
    } finally {
      setSaving(false)
    }
  }

  async function handleDismiss(faceId: string) {
    if (!card.image_id) return
    setDismissing((prev) => ({ ...prev, [faceId]: true }))

    try {
      await dismissFace(card.image_id, faceId)
      const updatedFaces: DetectedFace[] = card.faces.map((f) =>
        f.face_id === faceId ? { ...f, dismissed: true } : f,
      )
      onUpdate({ faces: updatedFaces })
    } catch {
      // silently fail for demo
    } finally {
      setDismissing((prev) => ({ ...prev, [faceId]: false }))
    }
  }

  return (
    <article className={styles.card}>
      {card.status === 'ready' || card.status === 'failed' ? (
        <img
          className={styles.thumb}
          src={`/image-preview?path=${encodeURIComponent(card.file_path)}`}
          alt={card.file_path}
          loading="lazy"
        />
      ) : (
        <div className={styles.thumbPlaceholder}>
          {card.status === 'processing' && <span className={styles.spinner} />}
        </div>
      )}

      <div className={styles.body}>
        <div className={styles.filePath}>{card.file_path}</div>
        <span className={`${styles.badge} ${styles[statusInfo.className]}`}>{statusInfo.label}</span>

        {card.status === 'ready' && (
          <div className={styles.metadataSection}>
            <div className={styles.metaRow}>
              <span className={styles.metaLabel}>Caption</span>
              <span className={styles.metaValue}>{card.caption || '—'}</span>
            </div>
            <div className={styles.metaRow}>
              <span className={styles.metaLabel}>Timestamp</span>
              <span className={styles.metaValue}>{formatTimestamp(card.capture_timestamp)}</span>
            </div>
            <div className={styles.metaRow}>
              <span className={styles.metaLabel}>GPS</span>
              <span className={styles.metaValue}>{formatGps(card.lat, card.lon)}</span>
            </div>
          </div>
        )}

        {card.status === 'ready' && activeFaces.length > 0 && (
          <div className={styles.facesSection}>
            <div className={styles.facesTitle}>
              Faces detected ({activeFaces.length})
              {dismissedCount > 0 && (
                <span className={styles.dismissedCount}> · {dismissedCount} dismissed</span>
              )}
            </div>
            {activeFaces.map((face) => (
              <div key={face.face_id} className={styles.faceRow}>
                <FaceCrop filePath={card.file_path} bbox={face.bbox} />
                <div className={styles.faceInfo}>
                  <input
                    className={styles.nameInput}
                    type="text"
                    placeholder="Name this person"
                    value={faceNames[face.face_id] || ''}
                    onChange={(e) =>
                      setFaceNames((prev) => ({ ...prev, [face.face_id]: e.target.value }))
                    }
                  />
                  <span className={styles.confidence}>{(face.confidence * 100).toFixed(0)}% conf</span>
                </div>
                <button
                  className={styles.dismissBtn}
                  onClick={() => handleDismiss(face.face_id)}
                  disabled={!!dismissing[face.face_id]}
                  title="Not a face"
                >
                  {dismissing[face.face_id] ? '...' : '✕ Not a face'}
                </button>
              </div>
            ))}
            <button className={styles.saveBtn} onClick={handleSaveNames} disabled={saving}>
              {saving ? 'Saving...' : saved ? 'Saved!' : 'Save Names'}
            </button>
          </div>
        )}

        {card.status === 'ready' && activeFaces.length === 0 && (
          <div className={styles.noFaces}>
            No faces detected
            {dismissedCount > 0 && ` (${dismissedCount} dismissed)`}
          </div>
        )}
      </div>
    </article>
  )
}

/** Renders a cropped face thumbnail using the bbox from the original image. */
function FaceCrop({ filePath, bbox }: { filePath: string; bbox: number[] }) {
  const [x1, y1, x2, y2] = bbox
  const w = x2 - x1
  const h = y2 - y1

  if (w <= 0 || h <= 0) {
    return <div className={styles.faceCropPlaceholder} />
  }

  const displaySize = 48
  const scale = displaySize / Math.max(w, h)

  return (
    <div className={styles.faceCrop} style={{ width: displaySize, height: displaySize }}>
      <img
        src={`/image-preview?path=${encodeURIComponent(filePath)}`}
        alt="face"
        style={{
          position: 'absolute',
          left: -x1 * scale,
          top: -y1 * scale,
          width: 'auto',
          height: 'auto',
          transform: `scale(${scale})`,
          transformOrigin: '0 0',
        }}
      />
    </div>
  )
}
