import type { QueryMode, SearchResponse, IngestResponse, UpdateFacesResponse, LLMStatus, LLMAvailable, AgentStep, AllModelsStatus, ModelServiceStatus, LibraryResponse } from './types'

export async function fetchLibrary(limit = 50, cursor?: string): Promise<LibraryResponse> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (cursor) params.set('cursor', cursor)
  const res = await fetch(`/library?${params}`)
  if (!res.ok) throw new Error(`Library fetch failed: ${res.status}`)
  return res.json()
}

export async function search(
  mode: QueryMode,
  textQuery: string,
  imagePath: string,
  topK: number,
): Promise<SearchResponse> {
  let endpoint: string
  let payload: Record<string, unknown>

  if (mode === 'text') {
    endpoint = '/search/text'
    payload = { query: textQuery, top_k: topK }
  } else {
    endpoint = '/search/image'
    payload = {
      image_path: imagePath,
      query: mode === 'image+text' ? textQuery : null,
      top_k: topK,
    }
  }

  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    throw new Error(`Search failed: ${res.status}`)
  }

  return res.json()
}

export async function browseImages(): Promise<string[]> {
  const res = await fetch('/browse-images')
  if (!res.ok) {
    throw new Error(`Browse failed: ${res.status}`)
  }
  const data: { paths: string[] } = await res.json()
  return data.paths
}

export async function ingestImage(imagePath: string): Promise<IngestResponse> {
  const res = await fetch('/ingest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_path: imagePath }),
  })

  if (!res.ok) {
    throw new Error(`Ingest failed: ${res.status}`)
  }

  return res.json()
}

export async function updateFaces(
  imageId: string,
  faces: { face_id: string; name: string }[],
): Promise<UpdateFacesResponse> {
  const res = await fetch(`/images/${imageId}/faces`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ faces }),
  })

  if (!res.ok) {
    throw new Error(`Failed to update faces: ${res.status}`)
  }

  return res.json()
}

export async function dismissFace(
  imageId: string,
  faceId: string,
): Promise<{ image_id: string; face_id: string; dismissed: boolean; caption: string | null; ingestion_status: string | null }> {
  const res = await fetch(`/images/${imageId}/faces/${faceId}/dismiss`, {
    method: 'PUT',
  })

  if (!res.ok) {
    throw new Error(`Failed to dismiss face: ${res.status}`)
  }

  return res.json()
}

// LLM lifecycle

export async function fetchLLMStatus(): Promise<LLMStatus> {
  const res = await fetch('/llm/status')
  if (!res.ok) throw new Error(`Failed to fetch LLM status: ${res.status}`)
  return res.json()
}

export async function fetchLLMAvailable(): Promise<LLMAvailable> {
  const res = await fetch('/llm/available')
  if (!res.ok) throw new Error(`Failed to fetch available models: ${res.status}`)
  return res.json()
}

export async function loadLLM(modelName: string, device: string): Promise<LLMStatus> {
  const res = await fetch('/llm/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: modelName, device }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Load failed: ${res.status}`)
  }
  return res.json()
}

export async function unloadLLM(): Promise<LLMStatus> {
  const res = await fetch('/llm/unload', { method: 'POST' })
  if (!res.ok) throw new Error(`Unload failed: ${res.status}`)
  return res.json()
}

// Unified model management

export async function fetchAllModelsStatus(): Promise<AllModelsStatus> {
  const res = await fetch('/models/status')
  if (!res.ok) throw new Error(`Failed to fetch models status: ${res.status}`)
  return res.json()
}

export async function loadModel(name: string): Promise<ModelServiceStatus> {
  const res = await fetch(`/models/${name}/load`, { method: 'POST' })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to load ${name}: ${res.status}`)
  }
  return res.json()
}

export async function unloadModel(name: string): Promise<ModelServiceStatus> {
  const res = await fetch(`/models/${name}/unload`, { method: 'POST' })
  if (!res.ok) throw new Error(`Failed to unload ${name}: ${res.status}`)
  return res.json()
}

// Streaming search

export async function searchTextStream(
  query: string,
  topK: number,
  onStep: (step: AgentStep) => void,
  onResult: (response: SearchResponse) => void,
): Promise<void> {
  const res = await fetch('/search/text/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Search failed: ${res.status}`)
  }

  const reader = res.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // Parse SSE events from buffer
    const parts = buffer.split('\n\n')
    // Keep the last incomplete part in the buffer
    buffer = parts.pop() || ''

    for (const part of parts) {
      const lines = part.split('\n')
      let eventType = ''
      let data = ''

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim()
        } else if (line.startsWith('data: ')) {
          data = line.slice(6)
        }
      }

      if (!data) continue

      try {
        if (eventType === 'step') {
          onStep(JSON.parse(data) as AgentStep)
        } else if (eventType === 'result') {
          onResult(JSON.parse(data) as SearchResponse)
        }
      } catch {
        // Skip malformed JSON
      }
    }
  }
}

/** Parse SSE text into events. Exported for testing. */
export function parseSSEEvents(text: string): Array<{ event: string; data: string }> {
  const events: Array<{ event: string; data: string }> = []
  const parts = text.split('\n\n')

  for (const part of parts) {
    if (!part.trim()) continue
    const lines = part.split('\n')
    let eventType = ''
    let data = ''

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        eventType = line.slice(7).trim()
      } else if (line.startsWith('data: ')) {
        data = line.slice(6)
      }
    }

    if (eventType && data) {
      events.push({ event: eventType, data })
    }
  }

  return events
}
