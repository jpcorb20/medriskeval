"""OpenAI API model implementation.

This module provides ChatModel implementations for OpenAI's API,
supporting both the standard chat completions and batch API.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

from medriskeval.core.types import ChatMessage
from medriskeval.models.base import (
    ChatModel,
    ContentFilterError,
    ContextLengthError,
    GenerationParams,
    ModelError,
    ModelOutput,
    RateLimitError,
    UsageStats,
)
from medriskeval.models.retry import RateLimiter, RetryConfig, retry_sync

logger = logging.getLogger(__name__)


def _lazy_import_openai():
    """Lazy import openai to avoid import errors when not installed."""
    try:
        import openai
        return openai
    except ImportError:
        raise ImportError(
            "openai package not installed. Install with: pip install openai"
        )


class OpenAIModel(ChatModel):
    """OpenAI API ChatModel implementation.
    
    Supports GPT-4, GPT-4-Turbo, GPT-3.5-Turbo and other OpenAI models.
    Handles rate limiting, retries, and provides raw response for auditability.
    
    Attributes:
        model: The OpenAI model identifier (e.g., 'gpt-4', 'gpt-4-turbo').
        client: The OpenAI client instance.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: float = 60.0,
        retry_config: Optional[RetryConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """Initialize OpenAI model.
        
        Args:
            model: OpenAI model identifier.
            api_key: API key (defaults to OPENAI_API_KEY env var).
            base_url: Custom API base URL (for Azure or proxies).
            organization: OpenAI organization ID.
            timeout: Request timeout in seconds.
            retry_config: Configuration for retry behavior.
            rate_limiter: Optional rate limiter instance.
        """
        openai = _lazy_import_openai()
        
        self._model = model
        self._retry_config = retry_config or RetryConfig()
        self._rate_limiter = rate_limiter

        # Initialize client
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
            organization=organization,
            timeout=timeout,
        )

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def supports_seed(self) -> bool:
        # OpenAI supports seed for newer models
        return True

    @property
    def supports_batching(self) -> bool:
        # OpenAI has batch API, but we use concurrent requests instead
        return False

    def generate(
        self,
        messages: Sequence[ChatMessage],
        gen_params: Optional[GenerationParams] = None,
    ) -> ModelOutput:
        """Generate a response using OpenAI's chat completions API.
        
        Args:
            messages: Conversation messages.
            gen_params: Generation parameters.
            
        Returns:
            ModelOutput with generated text and metadata.
        """
        params = gen_params or GenerationParams()
        formatted_messages = self._format_messages(messages)

        # Apply rate limiting if configured
        if self._rate_limiter:
            from medriskeval.models.batching import estimate_messages_tokens
            estimated_tokens = estimate_messages_tokens(messages)
            self._rate_limiter.acquire_sync(estimated_tokens)

        @retry_sync(self._retry_config)
        def _call_api() -> ModelOutput:
            return self._make_request(formatted_messages, params)

        return _call_api()

    def _make_request(
        self,
        messages: list[dict[str, Any]],
        params: GenerationParams,
    ) -> ModelOutput:
        """Make the actual API request.
        
        Args:
            messages: Formatted messages for the API.
            params: Generation parameters.
            
        Returns:
            ModelOutput from the response.
            
        Raises:
            RateLimitError: On rate limit exceeded.
            ContentFilterError: On content filter triggered.
            ContextLengthError: On context length exceeded.
            ModelError: On other API errors.
        """
        openai = _lazy_import_openai()

        try:
            # Build request parameters
            request_params: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "max_tokens": params.max_tokens,
            }

            if params.seed is not None:
                request_params["seed"] = params.seed
            if params.stop:
                request_params["stop"] = params.stop
            if params.presence_penalty != 0.0:
                request_params["presence_penalty"] = params.presence_penalty
            if params.frequency_penalty != 0.0:
                request_params["frequency_penalty"] = params.frequency_penalty

            response = self._client.chat.completions.create(**request_params)

            # Extract response data
            choice = response.choices[0]
            text = choice.message.content or ""
            finish_reason = choice.finish_reason

            # Build usage stats
            usage = UsageStats()
            if response.usage:
                usage = UsageStats(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            # Record actual usage for rate limiter
            if self._rate_limiter and response.usage:
                self._rate_limiter.record_usage(response.usage.total_tokens)

            # Convert response to dict for raw payload
            raw_response = response.model_dump() if hasattr(response, 'model_dump') else {}

            return ModelOutput(
                text=text,
                finish_reason=finish_reason,
                usage=usage,
                raw=raw_response,
                model=response.model,
            )

        except openai.RateLimitError as e:
            retry_after = None
            if hasattr(e, 'response') and e.response is not None:
                retry_after_header = e.response.headers.get('retry-after')
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        pass
            raise RateLimitError(str(e), retry_after=retry_after) from e

        except openai.BadRequestError as e:
            error_message = str(e).lower()
            if "content_filter" in error_message or "content filter" in error_message:
                raise ContentFilterError(str(e)) from e
            if "context_length" in error_message or "maximum context length" in error_message:
                raise ContextLengthError(str(e)) from e
            raise ModelError(str(e)) from e

        except openai.APIError as e:
            raise ModelError(str(e)) from e

    def generate_batch(
        self,
        messages_list: Sequence[Sequence[ChatMessage]],
        gen_params: Optional[GenerationParams] = None,
    ) -> list[ModelOutput]:
        """Generate responses for multiple conversations.
        
        Uses concurrent execution via the batching module.
        
        Args:
            messages_list: List of conversations.
            gen_params: Generation parameters for all requests.
            
        Returns:
            List of ModelOutput objects.
        """
        from medriskeval.models.batching import batch_generate_sync, BatchConfig

        config = BatchConfig(
            max_concurrent=5,  # Conservative for rate limits
            show_progress=True,
        )

        result = batch_generate_sync(
            model=self,
            messages_list=messages_list,
            gen_params=gen_params,
            config=config,
        )

        if not result.all_successful:
            failed_indices = [i for i, e in enumerate(result.errors) if e is not None]
            logger.warning(f"Batch had {result.failure_count} failures at indices: {failed_indices}")

        # Return outputs, replacing None with error outputs
        outputs = []
        for i, (output, error) in enumerate(zip(result.outputs, result.errors)):
            if output is not None:
                outputs.append(output)
            else:
                # Create error output for failed items
                outputs.append(ModelOutput(
                    text="",
                    finish_reason="error",
                    raw={"error": str(error)},
                    model=self._model,
                ))
        return outputs


class AzureOpenAIModel(ChatModel):
    """Azure OpenAI Service ChatModel implementation.
    
    Supports Azure-hosted OpenAI models with Azure-specific authentication
    and endpoint configuration.
    """

    def __init__(
        self,
        deployment: str,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        azure_endpoint: Optional[str] = None,
        timeout: float = 60.0,
        retry_config: Optional[RetryConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """Initialize Azure OpenAI model.
        
        Args:
            deployment: Azure deployment name.
            api_key: API key (defaults to AZURE_OPENAI_API_KEY env var).
            api_version: Azure API version.
            azure_endpoint: Azure endpoint URL (defaults to AZURE_OPENAI_ENDPOINT env var).
            timeout: Request timeout in seconds.
            retry_config: Configuration for retry behavior.
            rate_limiter: Optional rate limiter instance.
        """
        openai = _lazy_import_openai()

        self._deployment = deployment
        self._retry_config = retry_config or RetryConfig()
        self._rate_limiter = rate_limiter

        self._client = openai.AzureOpenAI(
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=api_version,
            azure_endpoint=azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            timeout=timeout,
        )

    @property
    def model_id(self) -> str:
        return self._deployment

    @property
    def supports_seed(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[ChatMessage],
        gen_params: Optional[GenerationParams] = None,
    ) -> ModelOutput:
        """Generate a response using Azure OpenAI API."""
        params = gen_params or GenerationParams()
        formatted_messages = self._format_messages(messages)

        if self._rate_limiter:
            from medriskeval.models.batching import estimate_messages_tokens
            estimated_tokens = estimate_messages_tokens(messages)
            self._rate_limiter.acquire_sync(estimated_tokens)

        @retry_sync(self._retry_config)
        def _call_api() -> ModelOutput:
            return self._make_request(formatted_messages, params)

        return _call_api()

    def _make_request(
        self,
        messages: list[dict[str, Any]],
        params: GenerationParams,
    ) -> ModelOutput:
        """Make the actual API request to Azure."""
        openai = _lazy_import_openai()

        try:
            request_params: dict[str, Any] = {
                "model": self._deployment,
                "messages": messages,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "max_tokens": params.max_tokens,
            }

            if params.seed is not None:
                request_params["seed"] = params.seed
            if params.stop:
                request_params["stop"] = params.stop

            response = self._client.chat.completions.create(**request_params)

            choice = response.choices[0]
            text = choice.message.content or ""
            finish_reason = choice.finish_reason

            usage = UsageStats()
            if response.usage:
                usage = UsageStats(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            if self._rate_limiter and response.usage:
                self._rate_limiter.record_usage(response.usage.total_tokens)

            raw_response = response.model_dump() if hasattr(response, 'model_dump') else {}

            return ModelOutput(
                text=text,
                finish_reason=finish_reason,
                usage=usage,
                raw=raw_response,
                model=response.model,
            )

        except openai.RateLimitError as e:
            retry_after = None
            if hasattr(e, 'response') and e.response is not None:
                retry_after_header = e.response.headers.get('retry-after')
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        pass
            raise RateLimitError(str(e), retry_after=retry_after) from e

        except openai.BadRequestError as e:
            error_message = str(e).lower()
            if "content_filter" in error_message:
                raise ContentFilterError(str(e)) from e
            if "context_length" in error_message:
                raise ContextLengthError(str(e)) from e
            raise ModelError(str(e)) from e

        except openai.APIError as e:
            raise ModelError(str(e)) from e
