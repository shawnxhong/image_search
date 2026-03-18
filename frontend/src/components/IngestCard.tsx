import { useState } from 'react'
import type { IngestCardState, DetectedFace } from '../types'
import { updateFaces, dismissFace } from '../api'
import styles from './IngestCard.module.css'

interface IngestCardProps {
  card: IngestCardState
  onUpdate: (update: Partial<IngestCardState>) => void
  onImageClick?: (filePath: string) => void
}

const STATUS_DISPLAY: Record<string, { label: string; className: string }> = {
  pending: { label: 'Pending', className: 'pending' },
  processing: { label: 'Processing...', className: 'processing' },
  ready: { label: 'Ready', className: 'ready' },
  pending_labels: { label: 'Needs Labels', className: 'pendingLabels' },
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

export default function IngestCard({ card, onUpdate, onImageClick }: IngestCardProps) {
  const [faceNames, setFaceNames] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {}
    for (const f of card.faces) {
      initial[f.face_id] = f.name || ''
    }
    return initial
  })
  const [customInput, setCustomInput] = useState<Record<string, boolean>>({})
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [dismissing, setDismissing] = useState<Record<string, boolean>>({})

  const statusInfo = STATUS_DISPLAY[card.status] || STATUS_DISPLAY.pending
  const activeFaces = card.faces.filter((f) => !f.dismissed)
  const dismissedCount = card.faces.length - activeFaces.length

  function handlePickCandidate(faceId: string, name: string) {
    setFaceNames((prev) => ({ ...prev, [faceId]: name }))
    setCustomInput((prev) => ({ ...prev, [faceId]: false }))
  }

  function handleShowCustomInput(faceId: string) {
    setCustomInput((prev) => ({ ...prev, [faceId]: true }))
    setFaceNames((prev) => ({ ...prev, [faceId]: '' }))
  }

  async function handleSaveNames() {
    if (!card.image_id) return
    setSaving(true)
    setSaved(false)

    const entries = activeFaces
      .filter((f) => faceNames[f.face_id]?.trim())
      .map((f) => ({ face_id: f.face_id, name: faceNames[f.face_id].trim() }))

    try {
      const result = await updateFaces(card.image_id, entries)
      const updatedFaces: DetectedFace[] = card.faces.map((f) => ({
        ...f,
        name: faceNames[f.face_id]?.trim() || f.name,
      }))
      const update: Partial<IngestCardState> = { faces: updatedFaces }
      if (result.ingestion_status) {
        update.status = result.ingestion_status as IngestCardState['status']
      }
      if (result.caption) {
        update.caption = result.caption
      }
      onUpdate(update)
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
      const result = await dismissFace(card.image_id, faceId)
      const updatedFaces: DetectedFace[] = card.faces.map((f) =>
        f.face_id === faceId ? { ...f, dismissed: true } : f,
      )
      const update: Partial<IngestCardState> = { faces: updatedFaces }
      if (result.ingestion_status) {
        update.status = result.ingestion_status as IngestCardState['status']
      }
      if (result.caption) {
        update.caption = result.caption
      }
      onUpdate(update)
    } catch {
      // silently fail for demo
    } finally {
      setDismissing((prev) => ({ ...prev, [faceId]: false }))
    }
  }

  const showDetails = card.status === 'ready' || card.status === 'pending_labels'
  const showThumb = showDetails || card.status === 'failed'

  return (
    <article className={styles.card}>
      {showThumb ? (
        <img
          className={`${styles.thumb} ${onImageClick ? styles.clickable : ''}`}
          src={`/image-preview?path=${encodeURIComponent(card.file_path)}`}
          alt={card.file_path}
          loading="lazy"
          onClick={onImageClick ? () => onImageClick(card.file_path) : undefined}
        />
      ) : (
        <div className={styles.thumbPlaceholder}>
          {card.status === 'processing' && <span className={styles.spinner} />}
        </div>
      )}

      <div className={styles.body}>
        <div className={styles.filePath}>{card.file_path}</div>
        <span className={`${styles.badge} ${styles[statusInfo.className]}`}>{statusInfo.label}</span>

        {card.status === 'failed' && card.error && (
          <div className={styles.errorMsg}>{card.error}</div>
        )}

        {showDetails && (
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

        {showDetails && activeFaces.length > 0 && (
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
                  {face.name ? (
                    <span className={styles.matchedName}>{face.name}</span>
                  ) : (
                    <FaceNamePicker
                      face={face}
                      value={faceNames[face.face_id] || ''}
                      showCustom={!!customInput[face.face_id]}
                      onPick={(name) => handlePickCandidate(face.face_id, name)}
                      onCustom={() => handleShowCustomInput(face.face_id)}
                      onChange={(val) =>
                        setFaceNames((prev) => ({ ...prev, [face.face_id]: val }))
                      }
                    />
                  )}
                  <span className={styles.confidence}>{(face.confidence * 100).toFixed(0)}% conf</span>
                </div>
                {!face.name && (
                  <button
                    className={styles.dismissBtn}
                    onClick={() => handleDismiss(face.face_id)}
                    disabled={!!dismissing[face.face_id]}
                    title="Not a face"
                  >
                    {dismissing[face.face_id] ? '...' : '✕ Not a face'}
                  </button>
                )}
              </div>
            ))}
            {activeFaces.some((f) => !f.name) && (
              <button className={styles.saveBtn} onClick={handleSaveNames} disabled={saving}>
                {saving ? 'Saving...' : saved ? 'Saved!' : 'Save Names'}
              </button>
            )}
          </div>
        )}

        {showDetails && activeFaces.length === 0 && (
          <div className={styles.noFaces}>
            No faces detected
            {dismissedCount > 0 && ` (${dismissedCount} dismissed)`}
          </div>
        )}
      </div>
    </article>
  )
}

/** Shows candidate buttons if available, otherwise a text input. */
function FaceNamePicker({
  face,
  value,
  showCustom,
  onPick,
  onCustom,
  onChange,
}: {
  face: DetectedFace
  value: string
  showCustom: boolean
  onPick: (name: string) => void
  onCustom: () => void
  onChange: (val: string) => void
}) {
  const candidates = face.candidates || []
  const hasCandidates = candidates.length > 0

  // If user picked a candidate, show it as selected
  if (value && !showCustom && hasCandidates) {
    return (
      <div className={styles.pickedRow}>
        <span className={styles.pickedName}>{value}</span>
        <button className={styles.changeBtn} onClick={onCustom} type="button">
          change
        </button>
      </div>
    )
  }

  // Show custom input if: no candidates, user clicked "Other", or showCustom
  if (!hasCandidates || showCustom) {
    return (
      <div>
        <input
          className={styles.nameInput}
          type="text"
          placeholder="Name this person"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
        {hasCandidates && (
          <button
            className={styles.backBtn}
            onClick={() => onPick('')}
            type="button"
          >
            back to suggestions
          </button>
        )}
      </div>
    )
  }

  // Show candidate buttons
  return (
    <div className={styles.candidateList}>
      {candidates.map((c) => (
        <button
          key={c.name}
          className={styles.candidateBtn}
          onClick={() => onPick(c.name)}
          type="button"
          title={`Similarity: ${((1 - c.distance) * 100).toFixed(0)}%`}
        >
          {c.name}
          <span className={styles.candidatePct}>{((1 - c.distance) * 100).toFixed(0)}%</span>
        </button>
      ))}
      <button className={styles.candidateOther} onClick={onCustom} type="button">
        Other...
      </button>
    </div>
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
