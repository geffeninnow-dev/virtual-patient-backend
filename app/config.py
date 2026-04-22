from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.engine import URL
import os

load_dotenv()


class Settings(BaseModel):
    db_host: str = os.getenv("DB_HOST", "")
    db_port: str = os.getenv("DB_PORT", "5432")
    db_name: str = os.getenv("DB_NAME", "")
    db_user: str = os.getenv("DB_USER", "")
    db_password: str = os.getenv("DB_PASSWORD", "")

    secret_key: str = os.getenv("SECRET_KEY", "change_me")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))

    ai_api_key: str = os.getenv("AI_API_KEY", "")
    ai_base_url: str = os.getenv("AI_BASE_URL", "")
    ai_model_name: str = os.getenv("AI_MODEL_NAME", "")

    @property
    def sqlalchemy_database_url(self):
        return URL.create(
            drivername="postgresql+psycopg2",
            username=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=int(self.db_port),
            database=self.db_name,
        )


settings = Settings()