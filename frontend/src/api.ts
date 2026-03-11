import type { QueryMode, SearchResponse, IngestResponse, UpdateFacesResponse } from './types'

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
): Promise<{ image_id: string; face_id: string; dismissed: boolean }> {
  const res = await fetch(`/images/${imageId}/faces/${faceId}/dismiss`, {
    method: 'PUT',
  })

  if (!res.ok) {
    throw new Error(`Failed to dismiss face: ${res.status}`)
  }

  return res.json()
}
