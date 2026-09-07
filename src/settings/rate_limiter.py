from typing import Literal

from pydantic import BaseModel, Field


class SemaphoreSettings(BaseModel):
    expiry: int = Field(
        30, description="How long to keep the semaphore resources alive, in seconds."
    )


class TokenBucketSettings(BaseModel):
    refill_frequency: float = Field(
        1.0, description="How often tokens are added to the bucket, in seconds."
    )
    refill_amount: int = Field(
        2, description="Number of tokens added per refill interval."
    )

class LimiterConfig(BaseModel):
    prefix: str = Field(..., description="Corporate prefix for shared redis")
    name: str = Field(
        ..., description="Unique name of the rate limiter (e.g., 'openai-api', 'rkvk')."
    )
    mode: Literal["semaphore", "token_bucket"] = Field(
        "semaphore", description="Limiter mode: 'semaphore' or 'token_bucket'."
    )
    capacity: int = Field(
        1, description="Maximum number of concurrent requests or tokens in the bucket."
    )
    max_sleep: float = Field(
        30.0, description="Maximum time to sleep when rate‑limited, in seconds."
    )
    semaphore: SemaphoreSettings | None= Field(
        None,
        description="Semaphore‑specific settings (only used when mode='semaphore').",
    )
    token_bucket: TokenBucketSettings | None = Field(
        None,
        description="Token‑bucket‑specific settings (only used when mode='token_bucket').",
    )


class RateLimiterSettings(BaseModel):
    limiters: list[LimiterConfig] = Field(
        default_factory=list,
        description="List of individual rate limiters, each with its own mode and settings.",
    )
