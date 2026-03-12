from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic Image Search"
    sqlite_url: str = "sqlite:///./image_search.db"
    chroma_path: str = "./.chroma"
    caption_collection: str = "caption_embeddings"
    image_collection: str = "image_embeddings"
    default_top_k: int = 20
    default_timezone: str = "UTC"
    caption_model_name: str = "Salesforce/blip-image-captioning-base"
    text_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # OpenVINO face detection models (paths relative to project root)
    face_detection_model: str = "face_recognition/models/intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml"
    face_landmarks_model: str = "face_recognition/models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"
    face_reidentification_model: str = "face_recognition/models/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml"
    face_detection_confidence: float = 0.6
    face_identity_threshold: float = 0.5  # Cosine distance threshold for face matching

    model_config = SettingsConfigDict(env_file=".env", env_prefix="IMG_SEARCH_")


settings = Settings()
