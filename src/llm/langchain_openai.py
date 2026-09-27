import logging
from contextlib import nullcontext
from http import HTTPMethod
from io import BytesIO
from typing import Optional, Union
import base64

import httpx
from httpx import AsyncHTTPTransport
from httpx_retries import Retry, RetryTransport
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import LLMResult
from limiters import AsyncSemaphore, AsyncTokenBucket
from PIL import Image, ImageOps
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from src.connections.rate_limiter import RateLimiterManager

from src.settings.settings import LlmSettings

logger = logging.getLogger(__name__)


class SemaphoredOpenAI(ChatOpenAI):
    def __init__(
        self,
        *args,
        rate_limiter: Optional[Union[AsyncSemaphore, AsyncTokenBucket]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._rate_limiter = rate_limiter

    async def agenerate(
        self, messages: list[list[BaseMessage]], *args, **kwargs
    ) -> LLMResult:
        metadata = kwargs.get("metadata")
        response_format = kwargs.get("response_format")
        if metadata and response_format and issubclass(response_format, BaseModel):
            metadata["schema"] = response_format.model_json_schema()
        async with self._rate_limiter:
            logger.info("Invoking ChatOpenAI")
            try:
                result = await super().agenerate(messages, *args, **kwargs)
                logger.info("Successful generation")
                return result
            except Exception:
                logger.exception("ChatOpenAI invoke failed")
                raise

class ChatOpenAIWrapper:
    """
    LangChain ChatOpenAI wrapper with automatic Langfuse tracing via callback handler,
    dynamic API key (callable), custom HTTP client, rate limiting, and controlled lifecycle.
    """

    def __init__(
        self,
        config: LlmSettings,
        rate_limiter_manager: Optional[RateLimiterManager] = None,
    ):
        self.config = config

        self._rate_limiter_manager = rate_limiter_manager
        self._rate_limiter = None

        self._chat: ChatOpenAI | None = None
        self._http_client: httpx.AsyncClient | None = None
        logger.debug("ChatOpenAIWrapper initialized")

    async def start(self) -> None:
        if self._chat is not None:
            logger.warning("ChatOpenAIWrapper already started, skipping")
            return

        if self._rate_limiter_manager:
            if not self._rate_limiter_manager.started:
                logger.warning(
                    "Rate limiter not started, will continue without rate limiter"
                )
                self._rate_limiter = nullcontext()
            else:
                self._rate_limiter = self._rate_limiter_manager.get_limiter('llm')
        else:
            self._rate_limiter = nullcontext()


        base_transport = AsyncHTTPTransport()
        transport = RetryTransport(
            transport=base_transport,
            retry=Retry(
                total=self.config.max_retries,
                backoff_factor=self.config.backoff,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods={
                    HTTPMethod.HEAD,
                    HTTPMethod.GET,
                    HTTPMethod.PUT,
                    HTTPMethod.DELETE,
                    HTTPMethod.OPTIONS,
                    HTTPMethod.TRACE,
                    HTTPMethod.POST,
                },
            ),
        )
        self._http_client = httpx.AsyncClient(
            transport=transport,
            limits=httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive_connections,
                keepalive_expiry=self.config.keepalive_expiry,
            ),
            follow_redirects=True,
        )

        # Create ChatOpenAI with dynamic api_key callable, custom http_async_client, etc.
        self._chat = SemaphoredOpenAI(
            rate_limiter=self._rate_limiter,
            model=self.config.model_name,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=min(self.config.max_tokens, 1000),
            base_url=self.config.base_url,
            http_async_client=self._http_client,
            max_retries=0,  # логикой управляет http_async_client
            timeout=httpx.Timeout(self.config.timeout),
            extra_body={
                "repetition_penalty": 1.15
            }
        )
        logger.info("ChatOpenAIWrapper created")
        # Test connection
        try:
            models = await self._chat.root_async_client.models.list()
            available_models = {model.id for model in models.data}
        except Exception:
            logger.exception("Connection check failed: API is not reachable")
            raise
        if self.config.model_name not in available_models:

            raise Exception("Configured LLM model %s is unavailable",
                           self.config.model_name,)
        logger.info("Connection check succeeded: API reachable and key valid.")
        logger.debug(f"Available models {models=}")

    async def stop(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
            self._chat = None
            logger.debug("ChatOpenAIWrapper stopped")

    @property
    def chat(self) -> ChatOpenAI:
        if self._chat is None:
            raise RuntimeError("ChatOpenAIWrapper not started. Call start() first.")
        return self._chat

    @staticmethod
    def _image_data_url(image_bytes: bytes, max_side: int = 1024) -> str:

        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((max_side, max_side))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=90, optimize=True)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

    @staticmethod
    def _strip_empty_think(content: str) -> str:
        content = content.strip()
        if content.startswith("<think>") and "</think>" in content:
            return content.split("</think>", 1)[1].strip()
        return content

    async def ocr_image_text(self, image_bytes: bytes) -> str:
        prompt = (
            "Recognize all visible text on this wine bottle or wine label. "
            "Pay special attention to producer, brand, wine name, grape variety, color, "
            "sweetness/style words such as брют, сухое, полусухое, полусладкое, сладкое, "
            "and Latin/Cyrillic lookalikes. Use only Russian Cyrillic letters, English ASCII letters, "
            "digits, spaces, and basic punctuation. Do not use accented Latin letters or other "
            "diacritics; transliterate them to plain English letters. Return only the recognized text, "
            "preserving line breaks."
        )
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt,
                },
                {"type": "image_url", "image_url": {"url": self._image_data_url(image_bytes)}},
            ]
        )
        result = await self.chat.ainvoke([message])
        content = getattr(result, "content", "") or ""
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
            content = "\n".join(parts)
        else:
            content = str(content or "")
        return self._strip_empty_think(content).strip()
