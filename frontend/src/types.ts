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

export type IngestionStatus = 'pending' | 'processing' | 'ready' | 'pending_labels' | 'failed'

export interface FaceCandidate {
  name: string
  distance: number
}

export interface DetectedFace {
  face_id: string
  bbox: number[]
  confidence: number
  name: string | null
  dismissed: boolean
  candidates: FaceCandidate[]
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
  error: string | null
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
  error: string | null
}

export interface UpdateFacesResponse {
  image_id: string
  updated: number
  caption: string | null
  ingestion_status: string | null
}

// LLM lifecycle types

export interface LLMStatus {
  loaded: boolean
  model_path: string | null
  model_name: string | null
  device: string | null
}

export interface LLMAvailable {
  models: string[]
  devices: string[]
}

// Unified model status types

export interface ModelServiceStatus {
  loaded: boolean
  name: string
}

export interface AllModelsStatus {
  llm: LLMStatus
  vlm: ModelServiceStatus
  embeddings: ModelServiceStatus
  face_detection: ModelServiceStatus
}

// Agent step types

export type AgentStepType = 'thinking' | 'tool_call' | 'tool_result' | 'done' | 'error'

export interface AgentStep {
  step_type: AgentStepType
  tool_name: string | null
  tool_args: Record<string, unknown> | null
  result_count: number | null
  message: string
}
