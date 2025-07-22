from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):

    Qdrant_db_path:str
    Qdrant_distance_method:str
    Embedding_Model_Size:int
    Data_Path:str
    DO_RESET:bool
    DB_NAME:str
    TOP_K:int
    
    Embedding_Model_Name:str
    BATCH_SIZE:int
    
    CHUNK_SIZE:int
    CHUNK_OVERLAP:int
    
    LLM_MODEL_NAME: str
    API_KEY :str
    API_URL :str
    
    EVALUATION_DATA_PATH :str
    class Config:
        env_file = ".env"

def get_settings():
    return Settings()