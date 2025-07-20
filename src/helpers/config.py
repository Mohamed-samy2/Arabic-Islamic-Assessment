from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):

    Qdrant_db_path:str
    Qdrant_distance_method:str
    Embedding_Model_Size:int
    Data_Path:str
    DO_RESET:bool
    DB_NAME:str

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()