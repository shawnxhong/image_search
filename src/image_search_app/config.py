from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic Image Search"
    sqlite_url: str = "sqlite:///./image_search.db"
    chroma_path: str = "./.chroma"
    caption_collection: str = "caption_embeddings"
    image_collection: str = "image_embeddings"
    default_top_k: int = 20
    solid_score_threshold: float = 0.25  # Minimum semantic score (1-cosine_distance) to appear in solid results
    default_timezone: str = "UTC"
    vlm_model_path: str = "models/Qwen2.5-VL-3B-Instruct/INT4"
    vlm_device: str = "GPU"
    text_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_embedding_model_dir: str = "models/all-MiniLM-L6-v2-ov"
    text_embedding_device: str = "GPU"
    embedding_dim: int = 384

    # OpenVINO face detection models (paths relative to project root)
    face_models_dir: str = "models"
    face_detection_model: str = "intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml"
    face_landmarks_model: str = "intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"
    face_reidentification_model: str = "intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml"
    face_detection_confidence: float = 0.6
    face_identity_threshold: float = 0.5  # Cosine distance threshold for face matching

    # OpenVINO GenAI LLM for agentic search
    llm_model_path: str = "models/Qwen3-4B-Instruct-ov"
    llm_device: str = "GPU"
    llm_max_agent_iterations: int = 3
    llm_models_dir: str = "models"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="IMG_SEARCH_")


settings = Settings()
