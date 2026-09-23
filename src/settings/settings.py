from typing import Literal

from pydantic import BaseModel, Field, computed_field, SecretStr
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


class YoloSettings(BaseModel):
    """YOLO cropper configuration."""

    model_path: str = Field(
        default="/models/yolo/label.pt", description="Path or Ultralytics reference for label crop model"
    )
    bottle_model_path: str = Field(
        default="/models/yolo/yolo26x.pt",
        description="Path or Ultralytics reference for bottle detection model",
    )
    label_model_path: str = Field(
        default="/models/yolo/label.pt",
        description="Path or Ultralytics reference for label crop model",
    )


class EmbeddingSettings(BaseModel):
    """SigLIP2 local model configuration."""

    model_id: str = Field(
        default="google/siglip2-base-patch16-224",
        description="Hugging Face model id",
    )
    model_dir: str = Field(
        default="/models/siglip2", description="Path to mounted SigLIP2 model files"
    )
    device: str = Field(
        default="auto", description="Embedding device: auto, cpu, cuda, or mps"
    )


class DinoV3Settings(BaseModel):
    """DINOv3 local model configuration."""

    model_id: str = Field(
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        description="Hugging Face model id",
    )
    model_dir: str = Field(
        default="/models/dinov3", description="Path to mounted DINOv3 model files"
    )
    device: str = Field(
        default="auto", description="DINOv3 device: auto, cpu, cuda, or mps"
    )
    patch_batch_size: int = Field(
        default=32, description="Batch size for on-the-fly DINOv3 patch-token encoding"
    )


class SearchSettings(BaseModel):
    """Runtime image search configuration."""

    global_encoder: Literal["siglip2", "dinov3"] = Field(
        default="siglip2",
        description="Encoder used for query global vectors in runtime search",
    )
    collection_encoder: Literal["siglip2", "dinov3"] = Field(
        default="siglip2",
        description="Qdrant collection suffix used for global vector search",
    )


class LlmSettings(BaseModel):
    """Remote LLM API configuration."""

    base_url: str = Field(
        default="http://localhost:1234/v1", description="LLM base URL"
    )
    model_name: str = Field(default="gpt", description="LLM model name")
    api_key: SecretStr = Field(default="", description="LLM API key")
    max_tokens: int = Field(default=20000)
    temperature: float = Field(default=1)
    timeout: float = Field(default=60.0, description="Request timeout")
    max_retries: int = Field(default=3, description="Max LLM call retries")
    backoff: float = Field(default=0.5, description="Backoff factor")
    max_connections: int = Field(default=100, description="Max HTTP connections")
    max_keepalive_connections: int = Field(
        default=20, description="Max keepalive connections"
    )
    keepalive_expiry: float = Field(default=30.0, description="Keepalive expiry")
    is_ollama_infer: bool = Field(default=False, description="Inference mode - default is vllm")

class MinioSettings(BaseModel):
    host: str | None = Field(default=None, description="Minio host")
    port: int = Field(default=None, description="Minio port")
    access_key: str | None = Field(default=None, description="Minio access key")
    secret_key: str | None = Field(default=None, description="Minio secret key")
    bucket: str | None = Field(default=None, description="Minio bucket")
    verify_ssl: bool = Field(
        default=False, description="Whether Minio connection uses SSL"
    )
    ca_cert_path: str | None = Field(
        default=None, description="Path to ca certs for Minio"
    )

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
    yolo: YoloSettings = Field(default_factory=YoloSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    dinov3: DinoV3Settings = Field(default_factory=DinoV3Settings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    rate_limiter: RateLimiterSettings = Field(default_factory=RateLimiterSettings)
    minio: MinioSettings = Field(default_factory=MinioSettings)

all_settings = AllSettings()
