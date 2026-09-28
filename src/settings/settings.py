
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
    device: str = Field(
        default="auto", description="YOLO device: auto (cuda -> mps -> cpu), cpu, cuda, or mps"
    )


class EmbeddingSettings(BaseModel):
    """SigLIP2 local model configuration."""

    model_id: str = Field(
        default="google/siglip2-base-patch16-384",
        description="Hugging Face model id",
    )
    model_dir: str = Field(
        default="/models/siglip2_384", description="Path to mounted SigLIP2 model files"
    )
    adapter_path: str = Field(
        default="/models/adapter/siglip2_384_views.npz",
        description=(
            "Дообученный адаптер векторов по ракурсам (scripts/train_view_adapter.py); "
            "пусто — поиск по исходным векторам SigLIP2"
        ),
    )
    device: str = Field(
        default="auto", description="Embedding device: auto, cpu, cuda, or mps"
    )


class SearchSettings(BaseModel):
    """Runtime image search configuration."""

    collection_encoder: str = Field(
        default="siglip2_384",
        description=(
            "Suffix of Qdrant collections: wine_<view>_<collection_encoder>, "
            "с адаптером — wine_<view>_<collection_encoder>_<версия адаптера>"
        ),
    )
    ocr_skip_visual_gap: float = Field(
        default=0.15,
        description=(
            "Не ждать OCR, если визуальный top-1 лучше top-2 больше чем на эту долю ((v1 - v2) / v2). "
            "На отложенных фото Vivino (SigLIP2-384 + адаптер) текст этикетки менял top-1 только при отрыве "
            "до 0.077; при 0.15 — вдвое больше запаса, OCR пропускается в ~2/3 поисков без потери точности. "
            "0 — всегда OCR."
        ),
    )
    ocr_skip_min_score: float = Field(
        default=0.50,
        description=(
            "…и только если визуальный скор top-1 не ниже этого: слабый top-1 с большим отрывом (вина нет "
            "в каталоге, сложное фото) всё равно проверяем текстом этикетки. На точность по Vivino не влияет, "
            "касается ~1% поисков."
        ),
    )
    ocr_budget_seconds: float = Field(
        default=7.5,
        description=(
            "Сколько секунд от начала поиска ждать OCR (LLM); не успела — ответ по изображению. "
            "Проверочный скрипт организаторов ждёт ответ не дольше 10 с. 0 — ждать сколько угодно."
        ),
    )


class LlmSettings(BaseModel):
    """Remote LLM API configuration."""

    base_url: str = Field(
        default="",
        description="LLM base URL; пусто — распознавание текста этикеток выключено, поиск только по изображению",
    )
    model_name: str = Field(default="", description="LLM model name; пусто — OCR выключен")

    @property
    def configured(self) -> bool:
        return bool(self.base_url.strip() and self.model_name.strip())
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
    disable_reasoning: bool = Field(
        default=False,
        description=(
            "Выключить размышления у моделей с reasoning (OpenRouter/RouterAI: reasoning.enabled=false): "
            "для OCR они не нужны и добавляют секунды"
        ),
    )

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

DEFAULT_JWT_SECRET = "insecure-default-secret-set-AUTH__JWT_SECRET-in-env"


class AuthSettings(BaseModel):
    """Упрощённая авторизация: email + пароль, JWT access-токен."""

    jwt_secret: str = Field(
        default=DEFAULT_JWT_SECRET, description="Секрет подписи JWT (AUTH__JWT_SECRET), от 32 байт"
    )
    jwt_algorithm: str = Field(default="HS256")
    access_token_ttl_minutes: int = Field(
        default=7 * 24 * 60, description="Время жизни access-токена"
    )
    anon_cookie_name: str = Field(
        default="wine_anon_id", description="Cookie анонимного пользователя (временная история)"
    )
    anon_history_ttl_hours: int = Field(
        default=72, description="Сколько хранится история анонимного пользователя"
    )


class NotificationSettings(BaseModel):
    """Напоминания "вы недавно смотрели вино, как вам?"."""

    enabled: bool = Field(default=True)
    delay_minutes: float = Field(
        default=60,
        description="Через сколько минут после последнего поиска (конец сессии) присылать сводку",
    )
    max_age_hours: float = Field(
        default=24, description="Вина из поисков старше этого в сводку не попадают"
    )
    check_interval_seconds: float = Field(
        default=60, description="Как часто фоновая задача ищет поиски для напоминания"
    )
    message_template: str = Field(
        default="Вы недавно смотрели вина ({count}). Что-то взяли?",
        description="Текст сводки; {count} — сколько вин в ней",
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
    search: SearchSettings = Field(default_factory=SearchSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    rate_limiter: RateLimiterSettings = Field(default_factory=RateLimiterSettings)
    minio: MinioSettings = Field(default_factory=MinioSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)

all_settings = AllSettings()
