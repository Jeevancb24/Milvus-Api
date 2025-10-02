from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MILVUS_HOST:str
    COLLECTION_NAME:str
    MILVUS_ROOT_PASSWORD:str
    AZURE_CONTAINER_NAME:str
    AZURE_ACCOUNT_URL:str
    AZURE_CONNECTION_STRING:str

    class Config:
        env_file = ".env"   # load from .env
        env_file_encoding = "utf-8"


# Singleton settings instance
settings = Settings()
