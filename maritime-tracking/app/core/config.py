from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "Maritime Tracking API"
    API_V1_STR: str = "/api/v1"
    
    # Data paths
    DATA_PATH: str = "./data"
    MODEL_PATH: str = "./app/api/models"
    
    # Model settings
    USE_GPU: bool = True
    SEQUENCE_LENGTH: int = 10
    BATCH_SIZE: int = 32
    
    # API settings
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    class Config:
        case_sensitive = True

settings = Settings()