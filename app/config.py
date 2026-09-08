from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "trakheesi-compliance-detector"
    app_env: str = "local"
    log_level: str = "info"

    pexels_api_key: str = ""
    azure_vision_endpoint: str = ""
    azure_vision_key: str = ""

    yolo_onnx_path: Path = Path("models/yolov8n_watermark.onnx")
    detect_conf_threshold: float = 0.42
    phash_hamming_threshold: int = 8
    duplicate_index_path: Path = Path("data/phash_index.json")

    @property
    def azure_vision_enabled(self) -> bool:
        return bool(self.azure_vision_endpoint and self.azure_vision_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
