export type QueryMode = 'text' | 'image' | 'image+text'

export interface MatchExplanation {
  image_id: string
  reason: string
  matched_constraints: string[]
  missing_metadata: string[]
}

export interface SearchResultItem {
  image_id: string
  file_path: string
  score: number
  explanation: MatchExplanation
}

export interface SearchResponse {
  solid_results: SearchResultItem[]
  soft_results: SearchResultItem[]
}

// Ingestion types

export type IngestionStatus = 'pending' | 'processing' | 'ready' | 'failed'

export interface DetectedFace {
  face_id: string
  bbox: number[]
  confidence: number
  name: string | null
  dismissed: boolean
}

export interface IngestResponse {
  image_id: string
  file_path: string
  ingestion_status: string
  caption: string | null
  capture_timestamp: string | null
  lat: number | null
  lon: number | null
  faces: DetectedFace[]
}

export interface IngestCardState {
  file_path: string
  status: IngestionStatus
  image_id: string | null
  caption: string | null
  capture_timestamp: string | null
  lat: number | null
  lon: number | null
  faces: DetectedFace[]
}

export interface UpdateFacesResponse {
  image_id: string
  updated: number
}
