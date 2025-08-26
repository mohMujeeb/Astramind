from pydantic import BaseModel
import os

class Settings(BaseModel):
    model_name: str = "llama-3.1-8b-instant"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    tavily_api_key: str | None = None
    index_dir: str = "data/index"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_name=os.getenv("MODEL_NAME", "llama-3.1-8b-instant"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            index_dir=os.getenv("INDEX_DIR", "data/index"),
        )
