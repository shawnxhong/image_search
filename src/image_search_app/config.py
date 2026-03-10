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
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_embedding_dim: int = 512

    model_config = SettingsConfigDict(env_file=".env", env_prefix="IMG_SEARCH_")


settings = Settings()
