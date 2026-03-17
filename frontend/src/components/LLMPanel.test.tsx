import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import LLMPanel from './LLMPanel'
import type { ModelStatusSnapshot } from './LLMPanel'
import * as api from '../api'

vi.mock('../api')
const mockedApi = vi.mocked(api)

const UNLOADED_STATUS = { loaded: false, model_path: null, model_name: null, device: null }
const LOADED_STATUS = { loaded: true, model_path: '/models/Qwen3-4B', model_name: 'Qwen3-4B', device: 'GPU' }
const AVAILABLE = { models: ['Qwen3-4B', 'Llama-7B'], devices: ['CPU', 'GPU'] }

const ALL_UNLOADED = {
  llm: UNLOADED_STATUS,
  vlm: { loaded: false, name: 'VL Captioner' },
  embeddings: { loaded: false, name: 'Embeddings' },
  face_detection: { loaded: false, name: 'Face Detection' },
}

const ALL_LOADED = {
  llm: LOADED_STATUS,
  vlm: { loaded: true, name: 'VL Captioner' },
  embeddings: { loaded: true, name: 'Embeddings' },
  face_detection: { loaded: true, name: 'Face Detection' },
}

function setupDefaultMocks(llmStatus = UNLOADED_STATUS, allModels = ALL_UNLOADED) {
  mockedApi.fetchLLMStatus.mockResolvedValue(llmStatus)
  mockedApi.fetchLLMAvailable.mockResolvedValue(AVAILABLE)
  mockedApi.fetchAllModelsStatus.mockResolvedValue(allModels)
}

beforeEach(() => {
  vi.resetAllMocks()
})

describe('LLMPanel initialization', () => {
  it('shows "Not loaded" when LLM is not loaded', async () => {
    setupDefaultMocks()

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => {
      expect(screen.getByTestId('llm-status-text')).toHaveTextContent('Not loaded')
    })
  })

  it('shows loaded model info when LLM is loaded', async () => {
    setupDefaultMocks(LOADED_STATUS, ALL_LOADED)

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => {
      expect(screen.getByTestId('llm-status-text')).toHaveTextContent('Qwen3-4B on GPU')
    })
  })

  it('populates model dropdown with available models', async () => {
    setupDefaultMocks()

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => {
      const options = screen.getAllByRole('option')
      const modelOptions = options.filter((o) => AVAILABLE.models.includes(o.textContent || ''))
      expect(modelOptions).toHaveLength(2)
    })
  })

  it('shows error when server connection fails', async () => {
    mockedApi.fetchLLMStatus.mockRejectedValue(new Error('Network error'))
    mockedApi.fetchLLMAvailable.mockRejectedValue(new Error('Network error'))
    mockedApi.fetchAllModelsStatus.mockRejectedValue(new Error('Network error'))

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => {
      expect(screen.getByText('Failed to connect to server')).toBeInTheDocument()
    })
  })
})

describe('LLMPanel indicator', () => {
  it('has "off" class when not loaded', async () => {
    setupDefaultMocks()

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => {
      const indicator = screen.getByTestId('llm-indicator')
      expect(indicator.className).toContain('off')
    })
  })

  it('has "on" class when loaded', async () => {
    setupDefaultMocks(LOADED_STATUS, ALL_LOADED)

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => {
      const indicator = screen.getByTestId('llm-indicator')
      expect(indicator.className).toContain('on')
    })
  })
})

describe('LLMPanel button states', () => {
  it('enables Load and disables Unload when not loaded', async () => {
    setupDefaultMocks()

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => {
      const loadBtns = screen.getAllByText('Load')
      const unloadBtns = screen.getAllByText('Unload')
      // First Load/Unload pair is for LLM
      expect(loadBtns[0]).not.toBeDisabled()
      expect(unloadBtns[0]).toBeDisabled()
    })
  })

  it('disables Load and enables Unload when loaded', async () => {
    setupDefaultMocks(LOADED_STATUS, ALL_LOADED)

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => {
      const loadBtns = screen.getAllByText('Load')
      const unloadBtns = screen.getAllByText('Unload')
      expect(loadBtns[0]).toBeDisabled()
      expect(unloadBtns[0]).not.toBeDisabled()
    })
  })
})

describe('LLMPanel load action', () => {
  it('calls loadLLM with selected model and device', async () => {
    setupDefaultMocks()
    mockedApi.loadLLM.mockResolvedValue(LOADED_STATUS)

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => {
      const loadBtns = screen.getAllByText('Load')
      expect(loadBtns[0]).not.toBeDisabled()
    })

    fireEvent.click(screen.getAllByText('Load')[0])

    await waitFor(() => {
      expect(mockedApi.loadLLM).toHaveBeenCalledWith('Qwen3-4B', 'GPU')
    })
  })

  it('shows error message when load fails', async () => {
    setupDefaultMocks()
    mockedApi.loadLLM.mockRejectedValue(new Error('GPU out of memory'))

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => expect(screen.getAllByText('Load')[0]).not.toBeDisabled())

    fireEvent.click(screen.getAllByText('Load')[0])

    await waitFor(() => {
      expect(screen.getByText('GPU out of memory')).toBeInTheDocument()
    })
  })
})

describe('LLMPanel unload action', () => {
  it('calls unloadLLM and updates status', async () => {
    setupDefaultMocks(LOADED_STATUS, ALL_LOADED)
    mockedApi.unloadLLM.mockResolvedValue(UNLOADED_STATUS)

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => expect(screen.getAllByText('Unload')[0]).not.toBeDisabled())

    fireEvent.click(screen.getAllByText('Unload')[0])

    await waitFor(() => {
      expect(mockedApi.unloadLLM).toHaveBeenCalled()
      expect(screen.getByTestId('llm-status-text')).toHaveTextContent('Not loaded')
    })
  })
})

describe('LLMPanel preset buttons', () => {
  it('renders Search Mode and Ingest Mode preset buttons', async () => {
    setupDefaultMocks()

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => {
      expect(screen.getByText('Search Mode')).toBeInTheDocument()
      expect(screen.getByText('Ingest Mode')).toBeInTheDocument()
    })
  })

  it('highlights Search Mode button when on search tab', async () => {
    setupDefaultMocks()

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => {
      const btn = screen.getByText('Search Mode')
      expect(btn.className).toContain('presetActive')
    })
  })

  it('highlights Ingest Mode button when on ingest tab', async () => {
    setupDefaultMocks()

    render(<LLMPanel activeTab="ingest" />)

    await waitFor(() => {
      const btn = screen.getByText('Ingest Mode')
      expect(btn.className).toContain('presetActive')
    })
  })

  it('Search Mode preset loads LLM + embeddings and unloads vlm + face_detection', async () => {
    setupDefaultMocks()
    mockedApi.loadLLM.mockResolvedValue(LOADED_STATUS)
    mockedApi.loadModel.mockResolvedValue({ loaded: true, name: 'Embeddings' })
    // vlm and face_detection already unloaded, so no unload calls expected

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => expect(screen.getByText('Search Mode')).not.toBeDisabled())

    fireEvent.click(screen.getByText('Search Mode'))

    await waitFor(() => {
      expect(mockedApi.loadLLM).toHaveBeenCalledWith('Qwen3-4B', 'GPU')
      expect(mockedApi.loadModel).toHaveBeenCalledWith('embeddings')
    })
  })

  it('Ingest Mode preset loads vlm + embeddings + face_detection and unloads LLM', async () => {
    setupDefaultMocks(LOADED_STATUS, {
      ...ALL_UNLOADED,
      llm: LOADED_STATUS,
    })
    mockedApi.unloadLLM.mockResolvedValue(UNLOADED_STATUS)
    mockedApi.loadModel.mockImplementation(async (name) => ({
      loaded: true,
      name: name === 'vlm' ? 'VL Captioner' : name === 'embeddings' ? 'Embeddings' : 'Face Detection',
    }))

    render(<LLMPanel activeTab="ingest" />)

    await waitFor(() => expect(screen.getByText('Ingest Mode')).not.toBeDisabled())

    fireEvent.click(screen.getByText('Ingest Mode'))

    await waitFor(() => {
      expect(mockedApi.unloadLLM).toHaveBeenCalled()
      expect(mockedApi.loadModel).toHaveBeenCalledWith('vlm')
      expect(mockedApi.loadModel).toHaveBeenCalledWith('embeddings')
      expect(mockedApi.loadModel).toHaveBeenCalledWith('face_detection')
    })
  })

  it('Search Mode preset skips loading already-loaded models', async () => {
    setupDefaultMocks(LOADED_STATUS, {
      ...ALL_UNLOADED,
      llm: LOADED_STATUS,
      embeddings: { loaded: true, name: 'Embeddings' },
    })

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => expect(screen.getByText('Search Mode')).not.toBeDisabled())

    fireEvent.click(screen.getByText('Search Mode'))

    // Should not call loadLLM or loadModel since they're already loaded
    await waitFor(() => {
      expect(mockedApi.loadLLM).not.toHaveBeenCalled()
      expect(mockedApi.loadModel).not.toHaveBeenCalledWith('embeddings')
    })
  })
})

describe('LLMPanel onStatusChange callback', () => {
  it('calls onStatusChange with initial status after init', async () => {
    setupDefaultMocks()
    const onStatusChange = vi.fn()

    render(<LLMPanel activeTab="search" onStatusChange={onStatusChange} />)

    await waitFor(() => {
      expect(onStatusChange).toHaveBeenCalled()
      const lastCall = onStatusChange.mock.calls[onStatusChange.mock.calls.length - 1][0] as ModelStatusSnapshot
      expect(lastCall.llm.loaded).toBe(false)
      expect(lastCall.models.vlm.loaded).toBe(false)
      expect(lastCall.models.embeddings.loaded).toBe(false)
      expect(lastCall.models.face_detection.loaded).toBe(false)
    })
  })

  it('calls onStatusChange with updated status after load', async () => {
    setupDefaultMocks()
    mockedApi.loadLLM.mockResolvedValue(LOADED_STATUS)
    const onStatusChange = vi.fn()

    render(<LLMPanel activeTab="search" onStatusChange={onStatusChange} />)

    await waitFor(() => expect(screen.getAllByText('Load')[0]).not.toBeDisabled())

    fireEvent.click(screen.getAllByText('Load')[0])

    await waitFor(() => {
      const calls = onStatusChange.mock.calls
      const lastCall = calls[calls.length - 1][0] as ModelStatusSnapshot
      expect(lastCall.llm.loaded).toBe(true)
    })
  })
})

describe('LLMPanel hints', () => {
  it('shows hint for search tab when LLM is not loaded', async () => {
    setupDefaultMocks()

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => {
      expect(screen.getByText(/Search needs LLM \+ Embeddings/)).toBeInTheDocument()
    })
  })

  it('shows hint for ingest tab when models are not loaded', async () => {
    setupDefaultMocks()

    render(<LLMPanel activeTab="ingest" />)

    await waitFor(() => {
      expect(screen.getByText(/Ingestion needs/)).toBeInTheDocument()
    })
  })

  it('shows no hint when all needed models are loaded for search', async () => {
    setupDefaultMocks(LOADED_STATUS, {
      ...ALL_UNLOADED,
      llm: LOADED_STATUS,
      embeddings: { loaded: true, name: 'Embeddings' },
    })

    render(<LLMPanel activeTab="search" />)

    await waitFor(() => {
      expect(screen.getByTestId('llm-status-text')).toHaveTextContent('Qwen3-4B on GPU')
    })

    // The hint should mention "Can release" for vlm/face_detection but not "Load"
    // Actually when vlm and face_detection are unloaded and LLM + embeddings are loaded,
    // there should be no hint at all since nothing needs to change
    const hints = screen.queryAllByText(/Load:/)
    expect(hints).toHaveLength(0)
  })
})
