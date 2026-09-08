from typing import Literal

from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

from src.settings.rate_limiter import RateLimiterSettings


class DatabaseSettings(BaseModel):
    """Postgres configuration."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    db_name: str = Field(default="reml_copilot_vk", description="Database name")
    user: str = Field(default="", description="Database username")
    db_schema: str = Field(
        default="public", description="Database schema used for table initialization"
    )
    password: str = Field(default="", description="Database password")
    echo: bool = Field(default=False, description="Database Echo")
    max_overflow: int = Field(default=10, description="Database max overflow")
    pool_size: int = Field(default=5, description="Database pool size")

    @computed_field
    @property
    def url(self) -> str:
        return URL.create(
            "postgresql+asyncpg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.db_name,
        ).render_as_string(hide_password=False)


class QdrantSettings(BaseModel):
    """Qdrant configuration."""

    host: str = Field(default="localhost", description="Qdrant host")
    port: int = Field(default=6333, description="Qdrant port")
    grpc_port: int = Field(default=6334, description="Qdrant gRPC port")
    prefer_grpc: bool = Field(default=False, description="Prefer gRPC over REST")
    https: bool = Field(default=False, description="Use HTTPS for connection")
    api_key: str = Field(default="", description="Qdrant API key")
    prefix: str = Field(default="", description="Qdrant URL prefix")
    timeout: float | None = Field(default=None, description="Request timeout in seconds")
    collection_name: str = Field(
        default="wine_vectors", description="Default collection name"
    )

    @computed_field
    @property
    def url(self) -> str:
        scheme = "https" if self.https else "http"
        return f"{scheme}://{self.host}:{self.port}"


class RedisSettings(BaseModel):
    """Redis configuration."""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    username: str | None = Field(default=None, description="Redis username")
    password: str | None = Field(default=None, description="Redis password")
    ssl: bool = Field(default=False, description="Use TLS for Redis connection")
    socket_timeout: float = Field(default=5.0)
    socket_connect_timeout: float = Field(default=5.0)
    retry_on_timeout: bool = Field(default=True)
    max_connections: int = Field(default=50)


class AllSettings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_ignore_empty=True,
        env_nested_delimiter="__",
    )
    log_level: str = Field(default="INFO")
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    rate_limiter: RateLimiterSettings = Field(default_factory=RateLimiterSettings)

all_settings = AllSettings()
