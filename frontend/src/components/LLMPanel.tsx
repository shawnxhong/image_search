import { useEffect, useState, useCallback } from 'react'
import type { LLMStatus, LLMAvailable, ModelServiceStatus } from '../types'
import {
  fetchLLMStatus,
  fetchLLMAvailable,
  loadLLM,
  unloadLLM,
  fetchAllModelsStatus,
  loadModel,
  unloadModel,
} from '../api'
import styles from './LLMPanel.module.css'

export type ModelKey = 'vlm' | 'embeddings' | 'face_detection'

const MODEL_LABELS: Record<ModelKey, string> = {
  vlm: 'VL Captioner',
  embeddings: 'Embeddings',
  face_detection: 'Face Detection',
}

const SEARCH_NEEDED: ModelKey[] = ['embeddings']
const SEARCH_LLM_NEEDED = true
const INGEST_NEEDED: ModelKey[] = ['vlm', 'embeddings', 'face_detection']
const INGEST_LLM_NEEDED = false

export interface ModelStatusSnapshot {
  llm: LLMStatus
  models: Record<ModelKey, ModelServiceStatus>
}

interface Props {
  activeTab: string
  onStatusChange?: (status: ModelStatusSnapshot) => void
}

export default function LLMPanel({ activeTab, onStatusChange }: Props) {
  // LLM state (has model/device selection)
  const [llmStatus, setLlmStatus] = useState<LLMStatus>({
    loaded: false,
    model_path: null,
    model_name: null,
    device: null,
  })
  const [available, setAvailable] = useState<LLMAvailable>({ models: [], devices: ['CPU', 'GPU'] })
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedDevice, setSelectedDevice] = useState('GPU')

  // Other model statuses
  const [modelStatus, setModelStatus] = useState<Record<ModelKey, ModelServiceStatus>>({
    vlm: { loaded: false, name: 'VL Captioner' },
    embeddings: { loaded: false, name: 'Embeddings' },
    face_detection: { loaded: false, name: 'Face Detection' },
  })

  const [busy, setBusy] = useState<Record<string, boolean>>({})
  const [error, setError] = useState('')

  useEffect(() => {
    async function init() {
      try {
        const [allStatus, llm, avail] = await Promise.all([
          fetchAllModelsStatus(),
          fetchLLMStatus(),
          fetchLLMAvailable(),
        ])
        setLlmStatus(llm)
        setAvailable(avail)
        setModelStatus({
          vlm: allStatus.vlm,
          embeddings: allStatus.embeddings,
          face_detection: allStatus.face_detection,
        })
        if (llm.loaded && llm.model_name) {
          setSelectedModel(llm.model_name)
          setSelectedDevice(llm.device || 'GPU')
        } else if (avail.models.length > 0) {
          // Prefer Qwen3 model as default (case-insensitive)
          const qwen3 = avail.models.find((m) => m.toLowerCase().includes('qwen3'))
          setSelectedModel(qwen3 || avail.models[0])
        }
        if (avail.devices.length > 0 && !llm.loaded) {
          setSelectedDevice(avail.devices[avail.devices.length - 1])
        }
      } catch {
        setError('Failed to connect to server')
      }
    }
    init()
  }, [])

  async function handleLLMLoad() {
    if (!selectedModel) return
    setBusy((p) => ({ ...p, llm: true }))
    setError('')
    try {
      const s = await loadLLM(selectedModel, selectedDevice)
      setLlmStatus(s)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Load failed')
    } finally {
      setBusy((p) => ({ ...p, llm: false }))
    }
  }

  async function handleLLMUnload() {
    setBusy((p) => ({ ...p, llm: true }))
    setError('')
    try {
      const s = await unloadLLM()
      setLlmStatus(s)
    } catch {
      const s = await fetchLLMStatus().catch(() => ({
        loaded: false,
        model_path: null,
        model_name: null,
        device: null,
      }))
      setLlmStatus(s)
    } finally {
      setBusy((p) => ({ ...p, llm: false }))
    }
  }

  async function handleModelLoad(key: ModelKey) {
    setBusy((p) => ({ ...p, [key]: true }))
    setError('')
    try {
      const s = await loadModel(key)
      setModelStatus((p) => ({ ...p, [key]: s }))
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to load ${MODEL_LABELS[key]}`)
    } finally {
      setBusy((p) => ({ ...p, [key]: false }))
    }
  }

  async function handleModelUnload(key: ModelKey) {
    setBusy((p) => ({ ...p, [key]: true }))
    setError('')
    try {
      const s = await unloadModel(key)
      setModelStatus((p) => ({ ...p, [key]: s }))
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to unload ${MODEL_LABELS[key]}`)
    } finally {
      setBusy((p) => ({ ...p, [key]: false }))
    }
  }

  // Report status to parent whenever it changes
  const reportStatus = useCallback(() => {
    onStatusChange?.({ llm: llmStatus, models: modelStatus })
  }, [llmStatus, modelStatus, onStatusChange])

  useEffect(() => {
    reportStatus()
  }, [reportStatus])

  // Preset: load models for search mode (LLM + Embeddings, unload VLM + Face Detection)
  async function handleSearchPreset() {
    setError('')
    const ops: Promise<void>[] = []

    // Load LLM if not loaded
    if (!llmStatus.loaded && selectedModel) {
      setBusy((p) => ({ ...p, llm: true }))
      ops.push(
        loadLLM(selectedModel, selectedDevice)
          .then((s) => setLlmStatus(s))
          .catch((err) => setError(err instanceof Error ? err.message : 'LLM load failed'))
          .finally(() => setBusy((p) => ({ ...p, llm: false }))),
      )
    }

    // Load embeddings if not loaded
    if (!modelStatus.embeddings.loaded) {
      setBusy((p) => ({ ...p, embeddings: true }))
      ops.push(
        loadModel('embeddings')
          .then((s) => setModelStatus((p) => ({ ...p, embeddings: s })))
          .catch((err) => setError(err instanceof Error ? err.message : 'Embeddings load failed'))
          .finally(() => setBusy((p) => ({ ...p, embeddings: false }))),
      )
    }

    // Unload VLM if loaded
    if (modelStatus.vlm.loaded) {
      setBusy((p) => ({ ...p, vlm: true }))
      ops.push(
        unloadModel('vlm')
          .then((s) => setModelStatus((p) => ({ ...p, vlm: s })))
          .catch(() => {})
          .finally(() => setBusy((p) => ({ ...p, vlm: false }))),
      )
    }

    // Unload face detection if loaded
    if (modelStatus.face_detection.loaded) {
      setBusy((p) => ({ ...p, face_detection: true }))
      ops.push(
        unloadModel('face_detection')
          .then((s) => setModelStatus((p) => ({ ...p, face_detection: s })))
          .catch(() => {})
          .finally(() => setBusy((p) => ({ ...p, face_detection: false }))),
      )
    }

    await Promise.all(ops)
  }

  // Preset: load models for ingest mode (VLM + Embeddings + Face Detection, unload LLM)
  async function handleIngestPreset() {
    setError('')
    const ops: Promise<void>[] = []

    // Unload LLM if loaded
    if (llmStatus.loaded) {
      setBusy((p) => ({ ...p, llm: true }))
      ops.push(
        unloadLLM()
          .then((s) => setLlmStatus(s))
          .catch(() => fetchLLMStatus().then((s) => setLlmStatus(s)).catch(() => {}))
          .finally(() => setBusy((p) => ({ ...p, llm: false }))),
      )
    }

    // Load VLM, embeddings, face_detection if not loaded
    for (const key of INGEST_NEEDED) {
      if (!modelStatus[key].loaded) {
        setBusy((p) => ({ ...p, [key]: true }))
        ops.push(
          loadModel(key)
            .then((s) => setModelStatus((p) => ({ ...p, [key]: s })))
            .catch((err) =>
              setError(err instanceof Error ? err.message : `${MODEL_LABELS[key]} load failed`),
            )
            .finally(() => setBusy((p) => ({ ...p, [key]: false }))),
        )
      }
    }

    await Promise.all(ops)
  }

  const anyBusy = Object.values(busy).some(Boolean)

  // Build hints based on active tab
  function getHint(): string | null {
    if (activeTab === 'search') {
      const needed: string[] = []
      const canRelease: string[] = []
      if (SEARCH_LLM_NEEDED && !llmStatus.loaded) needed.push('LLM')
      if (!SEARCH_LLM_NEEDED && llmStatus.loaded) canRelease.push('LLM')
      for (const key of (['vlm', 'embeddings', 'face_detection'] as ModelKey[])) {
        if (SEARCH_NEEDED.includes(key) && !modelStatus[key].loaded) {
          needed.push(MODEL_LABELS[key])
        }
        if (!SEARCH_NEEDED.includes(key) && modelStatus[key].loaded) {
          canRelease.push(MODEL_LABELS[key])
        }
      }
      const parts: string[] = []
      if (needed.length) parts.push(`Load: ${needed.join(', ')}`)
      if (canRelease.length) parts.push(`Can release: ${canRelease.join(', ')}`)
      return parts.length ? `Search needs LLM + Embeddings. ${parts.join('. ')}` : null
    }
    if (activeTab === 'ingest') {
      const needed: string[] = []
      const canRelease: string[] = []
      if (INGEST_LLM_NEEDED && !llmStatus.loaded) needed.push('LLM')
      if (!INGEST_LLM_NEEDED && llmStatus.loaded) canRelease.push('LLM')
      for (const key of (['vlm', 'embeddings', 'face_detection'] as ModelKey[])) {
        if (INGEST_NEEDED.includes(key) && !modelStatus[key].loaded) {
          needed.push(MODEL_LABELS[key])
        }
        if (!INGEST_NEEDED.includes(key) && modelStatus[key].loaded) {
          canRelease.push(MODEL_LABELS[key])
        }
      }
      const parts: string[] = []
      if (needed.length) parts.push(`Load: ${needed.join(', ')}`)
      if (canRelease.length) parts.push(`Can release: ${canRelease.join(', ')}`)
      return parts.length
        ? `Ingestion needs VL Captioner + Embeddings + Face Detection. ${parts.join('. ')}`
        : null
    }
    return null
  }

  function indicatorClass(loaded: boolean, isBusy: boolean) {
    if (isBusy) return `${styles.indicator} ${styles.loading}`
    if (loaded) return `${styles.indicator} ${styles.on}`
    return `${styles.indicator} ${styles.off}`
  }

  const hint = getHint()

  return (
    <section className={styles.panel}>
      <h2>Model Control</h2>

      {hint && <p className={styles.hint}>{hint}</p>}
      {error && <p className={styles.error}>{error}</p>}

      <div className={styles.presetRow}>
        <button
          className={`${styles.presetBtn} ${activeTab === 'search' ? styles.presetActive : ''}`}
          onClick={handleSearchPreset}
          disabled={anyBusy}
        >
          Search Mode
        </button>
        <button
          className={`${styles.presetBtn} ${activeTab === 'ingest' ? styles.presetActive : ''}`}
          onClick={handleIngestPreset}
          disabled={anyBusy}
        >
          Ingest Mode
        </button>
      </div>

      {/* LLM row — has model/device selection */}
      <div className={styles.modelRow}>
        <span data-testid="llm-indicator" className={indicatorClass(llmStatus.loaded, !!busy.llm)} />
        <span className={styles.modelLabel}>LLM</span>
        <span data-testid="llm-status-text" className={styles.modelInfo}>
          {busy.llm
            ? llmStatus.loaded
              ? 'Unloading...'
              : `Loading ${selectedModel}...`
            : llmStatus.loaded
              ? `${llmStatus.model_name} on ${llmStatus.device}`
              : 'Not loaded'}
        </span>
        <select
          className={styles.inlineSelect}
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          disabled={!!busy.llm || llmStatus.loaded}
        >
          {available.models.length === 0 && <option value="">No models</option>}
          {available.models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        <select
          className={styles.inlineSelect}
          value={selectedDevice}
          onChange={(e) => setSelectedDevice(e.target.value)}
          disabled={!!busy.llm || llmStatus.loaded}
        >
          {available.devices.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
        <button
          className={styles.loadBtn}
          onClick={handleLLMLoad}
          disabled={!!busy.llm || llmStatus.loaded || !selectedModel}
        >
          Load
        </button>
        <button
          className={styles.unloadBtn}
          onClick={handleLLMUnload}
          disabled={!!busy.llm || !llmStatus.loaded}
        >
          Unload
        </button>
      </div>

      {/* VLM and Embeddings rows — model/device display with selection */}
      {(['vlm', 'embeddings'] as ModelKey[]).map((key) => {
        const st = modelStatus[key]
        const modelName = st.model_name || st.name
        const device = st.device || 'CPU'
        return (
          <div key={key} className={styles.modelRow}>
            <span className={indicatorClass(st.loaded, !!busy[key])} />
            <span className={styles.modelLabel}>{MODEL_LABELS[key]}</span>
            <span data-testid={`${key}-status-text`} className={styles.modelInfo}>
              {busy[key]
                ? st.loaded
                  ? 'Unloading...'
                  : `Loading ${modelName}...`
                : st.loaded
                  ? `${modelName} on ${device}`
                  : 'Not loaded'}
            </span>
            <select
              className={styles.inlineSelect}
              defaultValue={modelName}
              disabled
            >
              <option value={modelName}>{modelName}</option>
            </select>
            <select
              className={styles.inlineSelect}
              defaultValue={device}
              disabled
            >
              <option value={device}>{device}</option>
            </select>
            <button
              className={styles.loadBtn}
              onClick={() => handleModelLoad(key)}
              disabled={!!busy[key] || st.loaded}
            >
              Load
            </button>
            <button
              className={styles.unloadBtn}
              onClick={() => handleModelUnload(key)}
              disabled={!!busy[key] || !st.loaded}
            >
              Unload
            </button>
          </div>
        )
      })}

      {/* Face Detection row — status display only */}
      <div className={styles.modelRow}>
        <span className={indicatorClass(modelStatus.face_detection.loaded, !!busy.face_detection)} />
        <span className={styles.modelLabel}>{MODEL_LABELS.face_detection}</span>
        <span data-testid="face_detection-status-text" className={styles.modelInfo}>
          {busy.face_detection
            ? modelStatus.face_detection.loaded
              ? 'Unloading...'
              : `Loading ${modelStatus.face_detection.model_name || 'models'}...`
            : modelStatus.face_detection.loaded
              ? `${modelStatus.face_detection.model_name || 'Intel OMZ Face Pipeline'} on ${modelStatus.face_detection.device || 'CPU'}`
              : 'Not loaded'}
        </span>
        <button
          className={styles.loadBtn}
          onClick={() => handleModelLoad('face_detection')}
          disabled={!!busy.face_detection || modelStatus.face_detection.loaded}
        >
          Load
        </button>
        <button
          className={styles.unloadBtn}
          onClick={() => handleModelUnload('face_detection')}
          disabled={!!busy.face_detection || !modelStatus.face_detection.loaded}
        >
          Unload
        </button>
      </div>
    </section>
  )
}
