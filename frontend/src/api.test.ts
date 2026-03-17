import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fetchLLMStatus, fetchLLMAvailable, loadLLM, unloadLLM, search, parseSSEEvents, searchTextStream } from './api'

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('fetchLLMStatus', () => {
  it('returns status when server responds ok', async () => {
    const mockStatus = { loaded: true, model_path: '/models/Qwen3', model_name: 'Qwen3', device: 'GPU' }
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(mockStatus), { status: 200 }))

    const result = await fetchLLMStatus()
    expect(result).toEqual(mockStatus)
    expect(fetch).toHaveBeenCalledWith('/llm/status')
  })

  it('throws on non-ok response', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', { status: 500 }))
    await expect(fetchLLMStatus()).rejects.toThrow('Failed to fetch LLM status: 500')
  })
})

describe('fetchLLMAvailable', () => {
  it('returns available models and devices', async () => {
    const mockAvailable = { models: ['ModelA', 'ModelB'], devices: ['CPU', 'GPU'] }
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(mockAvailable), { status: 200 }))

    const result = await fetchLLMAvailable()
    expect(result).toEqual(mockAvailable)
    expect(fetch).toHaveBeenCalledWith('/llm/available')
  })

  it('throws on non-ok response', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', { status: 503 }))
    await expect(fetchLLMAvailable()).rejects.toThrow('Failed to fetch available models: 503')
  })
})

describe('loadLLM', () => {
  it('sends model_name and device in POST body', async () => {
    const mockStatus = { loaded: true, model_path: '/models/Qwen3', model_name: 'Qwen3', device: 'GPU' }
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(mockStatus), { status: 200 }))

    const result = await loadLLM('Qwen3', 'GPU')
    expect(result).toEqual(mockStatus)
    expect(fetch).toHaveBeenCalledWith('/llm/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: 'Qwen3', device: 'GPU' }),
    })
  })

  it('throws with detail from error response', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: 'Model not found' }), { status: 500 }),
    )
    await expect(loadLLM('BadModel', 'CPU')).rejects.toThrow('Model not found')
  })

  it('throws generic message when error body is not JSON', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('internal error', { status: 500 }))
    await expect(loadLLM('BadModel', 'CPU')).rejects.toThrow('Load failed: 500')
  })
})

describe('unloadLLM', () => {
  it('sends POST and returns status', async () => {
    const mockStatus = { loaded: false, model_path: null, model_name: null, device: null }
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(mockStatus), { status: 200 }))

    const result = await unloadLLM()
    expect(result).toEqual(mockStatus)
    expect(fetch).toHaveBeenCalledWith('/llm/unload', { method: 'POST' })
  })

  it('throws on non-ok response', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', { status: 500 }))
    await expect(unloadLLM()).rejects.toThrow('Unload failed: 500')
  })
})

describe('search', () => {
  it('calls /search/text for text mode', async () => {
    const mockResponse = { solid_results: [], soft_results: [] }
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(mockResponse), { status: 200 }))

    await search('text', 'sunset beach', '', 10)
    expect(fetch).toHaveBeenCalledWith('/search/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: 'sunset beach', top_k: 10 }),
    })
  })

  it('calls /search/image for image mode', async () => {
    const mockResponse = { solid_results: [], soft_results: [] }
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(mockResponse), { status: 200 }))

    await search('image', '', '/path/to/img.jpg', 5)
    expect(fetch).toHaveBeenCalledWith('/search/image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_path: '/path/to/img.jpg', query: null, top_k: 5 }),
    })
  })

  it('throws on non-ok response', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', { status: 422 }))
    await expect(search('text', 'test', '', 10)).rejects.toThrow('Search failed: 422')
  })
})

describe('parseSSEEvents', () => {
  it('parses a single step event', () => {
    const text = 'event: step\ndata: {"step_type":"thinking","message":"Analyzing..."}\n\n'
    const events = parseSSEEvents(text)
    expect(events).toHaveLength(1)
    expect(events[0].event).toBe('step')
    expect(JSON.parse(events[0].data).step_type).toBe('thinking')
  })

  it('parses multiple events', () => {
    const text = [
      'event: step\ndata: {"step_type":"thinking","message":"Analyzing..."}',
      'event: step\ndata: {"step_type":"tool_call","tool_name":"search_by_caption","message":"Calling search_by_caption"}',
      'event: result\ndata: {"solid_results":[],"soft_results":[]}',
    ].join('\n\n') + '\n\n'

    const events = parseSSEEvents(text)
    expect(events).toHaveLength(3)
    expect(events[0].event).toBe('step')
    expect(events[1].event).toBe('step')
    expect(events[2].event).toBe('result')
  })

  it('returns empty array for empty text', () => {
    expect(parseSSEEvents('')).toHaveLength(0)
  })

  it('skips events without data', () => {
    const text = 'event: step\n\nevent: step\ndata: {"step_type":"done","message":"Done"}\n\n'
    const events = parseSSEEvents(text)
    expect(events).toHaveLength(1)
  })
})

describe('searchTextStream', () => {
  it('calls /search/text/stream with correct payload', async () => {
    const sseBody = [
      'event: step\ndata: {"step_type":"thinking","message":"Analyzing..."}',
      'event: step\ndata: {"step_type":"done","message":"Done"}',
      'event: result\ndata: {"solid_results":[],"soft_results":[]}',
    ].join('\n\n') + '\n\n'

    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(sseBody, { status: 200 }))

    const steps: unknown[] = []
    let result: unknown = null

    await searchTextStream(
      'beach sunset',
      10,
      (step) => steps.push(step),
      (res) => { result = res },
    )

    expect(fetch).toHaveBeenCalledWith('/search/text/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: 'beach sunset', top_k: 10 }),
    })

    expect(steps).toHaveLength(2)
    expect(result).toEqual({ solid_results: [], soft_results: [] })
  })

  it('throws on non-ok response', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: 'LLM not loaded' }), { status: 500 }),
    )

    await expect(
      searchTextStream('test', 10, () => {}, () => {}),
    ).rejects.toThrow('LLM not loaded')
  })
})
