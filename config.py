from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATASET_PATH: str
    MODEL_PATH: str
    ENCODER_NAME: str
    ONNX_ENCODER_NAME: str
    THRESHOLD: float
    LATENT_DIM: int
    SEED: int
    HF_TOKEN: str
    
    MAX_BATCH_SIZE: int
    BATCH_WINDOW_MS: int
    
    DELAY_FAST: float
    DELAY_SLOW: float

    SERVER_HOST: str
    SERVER_PORT: str

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()