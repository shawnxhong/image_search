import { useEffect, useCallback } from 'react'
import styles from './ImageLightbox.module.css'

interface Props {
  filePath: string
  onClose: () => void
}

export default function ImageLightbox({ filePath, onClose }: Props) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    },
    [onClose],
  )

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = ''
    }
  }, [handleKeyDown])

  return (
    <div className={styles.backdrop} onClick={onClose}>
      <div className={styles.imageWrapper} onClick={(e) => e.stopPropagation()}>
        <button className={styles.closeBtn} onClick={onClose} aria-label="Close">
          &times;
        </button>
        <img
          className={styles.image}
          src={`/image-preview?path=${encodeURIComponent(filePath)}`}
          alt={filePath}
        />
      </div>
    </div>
  )
}
