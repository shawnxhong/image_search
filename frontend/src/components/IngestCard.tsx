import { useState } from 'react'
import type { IngestCardState, DetectedFace } from '../types'
import { updateFaces } from '../api'
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

  const statusInfo = STATUS_DISPLAY[card.status] || STATUS_DISPLAY.pending

  async function handleSaveNames() {
    if (!card.image_id) return
    setSaving(true)
    setSaved(false)

    const entries = card.faces
      .filter((f) => faceNames[f.face_id]?.trim())
      .map((f) => ({ face_id: f.face_id, name: faceNames[f.face_id].trim() }))

    try {
      await updateFaces(card.image_id, entries)
      // Update face names in parent state
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

  // Update local face names when card.faces changes (e.g. after ingest completes)
  // This is handled by the useState initializer; for subsequent face data,
  // we rely on key-based remounting from the parent.

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

        {card.status === 'ready' && card.faces.length > 0 && (
          <div className={styles.facesSection}>
            <div className={styles.facesTitle}>Faces detected ({card.faces.length})</div>
            {card.faces.map((face) => (
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
              </div>
            ))}
            <button className={styles.saveBtn} onClick={handleSaveNames} disabled={saving}>
              {saving ? 'Saving...' : saved ? 'Saved!' : 'Save Names'}
            </button>
          </div>
        )}

        {card.status === 'ready' && card.faces.length === 0 && (
          <div className={styles.noFaces}>No faces detected</div>
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

  // We use object-fit + object-position via a canvas-like crop approach.
  // For simplicity, use a container with overflow:hidden and position the full image.
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
