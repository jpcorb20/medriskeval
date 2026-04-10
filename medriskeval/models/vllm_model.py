"""vLLM model implementation.

This module provides ChatModel implementations for vLLM,
supporting both local inference and remote OpenAI-compatible API servers.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

from medriskeval.core.types import ChatMessage
from medriskeval.models.base import (
    ChatModel,
    GenerationParams,
    ModelError,
    ModelOutput,
    UsageStats,
)
from medriskeval.models.retry import RetryConfig, retry_sync

logger = logging.getLogger(__name__)


def _lazy_import_vllm():
    """Lazy import vllm to avoid import errors when not installed."""
    try:
        from vllm import LLM, SamplingParams
        return LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vllm package not installed. Install with: pip install vllm"
        )


class VLLMModel(ChatModel):
    """Local vLLM model for high-throughput inference.
    
    This implementation uses vLLM's offline batched inference mode,
    which is highly efficient for processing large batches of requests.
    
    Attributes:
        model: The Hugging Face model identifier.
        llm: The vLLM LLM engine instance.
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        seed: int = 42,
        **vllm_kwargs: Any,
    ):
        """Initialize vLLM model.
        
        Args:
            model: Hugging Face model identifier or local path.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use.
            max_model_len: Maximum sequence length (auto-detected if None).
            dtype: Data type for model weights ('auto', 'float16', 'bfloat16').
            trust_remote_code: Whether to trust remote code in model config.
            seed: Default random seed for reproducibility.
            **vllm_kwargs: Additional arguments passed to vLLM.
        """
        LLM, _ = _lazy_import_vllm()

        self._model_id = model
        self._default_seed = seed

        logger.info(f"Loading vLLM model: {model}")
        self._llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            seed=seed,
            **vllm_kwargs,
        )
        logger.info(f"vLLM model loaded: {model}")

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def supports_seed(self) -> bool:
        return True

    @property
    def supports_batching(self) -> bool:
        # vLLM is optimized for batch inference
        return True

    def _build_sampling_params(
        self, gen_params: Optional[GenerationParams] = None
    ) -> Any:
        """Convert GenerationParams to vLLM SamplingParams."""
        _, SamplingParams = _lazy_import_vllm()
        params = gen_params or GenerationParams()

        return SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            seed=params.seed if params.seed is not None else self._default_seed,
            stop=params.stop,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
        )

    def _format_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """Format messages into a prompt string.
        
        Uses the tokenizer's chat template if available.
        Falls back to a simple format otherwise.
        """
        try:
            # Try to use the tokenizer's chat template
            tokenizer = self._llm.get_tokenizer()
            formatted_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            if hasattr(tokenizer, 'apply_chat_template'):
                return tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception as e:
            logger.debug(f"Chat template failed, using fallback: {e}")

        # Fallback to simple format
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
            else:
                parts.append(f"{msg.role}: {msg.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def generate(
        self,
        messages: Sequence[ChatMessage],
        gen_params: Optional[GenerationParams] = None,
    ) -> ModelOutput:
        """Generate a response for a single conversation.
        
        Note: For efficiency, prefer using generate_batch() with vLLM.
        """
        outputs = self.generate_batch([messages], gen_params)
        return outputs[0]

    def generate_batch(
        self,
        messages_list: Sequence[Sequence[ChatMessage]],
        gen_params: Optional[GenerationParams] = None,
    ) -> list[ModelOutput]:
        """Generate responses for multiple conversations efficiently.
        
        vLLM's batched inference is highly optimized and should be
        preferred over sequential generation.
        
        Args:
            messages_list: List of conversations.
            gen_params: Generation parameters for all requests.
            
        Returns:
            List of ModelOutput objects.
        """
        sampling_params = self._build_sampling_params(gen_params)

        # Format all prompts
        prompts = [self._format_prompt(msgs) for msgs in messages_list]

        logger.info(f"Running vLLM batch inference on {len(prompts)} prompts")
        
        try:
            outputs = self._llm.generate(prompts, sampling_params)
        except Exception as e:
            raise ModelError(f"vLLM generation failed: {e}") from e

        results = []
        for i, output in enumerate(outputs):
            if not output.outputs:
                results.append(ModelOutput(
                    text="",
                    finish_reason="error",
                    raw={"error": "No output generated"},
                    model=self._model_id,
                ))
                continue

            completion = output.outputs[0]
            
            # Build raw response for auditability
            raw_response = {
                "prompt": prompts[i],
                "prompt_token_ids": output.prompt_token_ids,
                "outputs": [
                    {
                        "text": o.text,
                        "token_ids": o.token_ids,
                        "finish_reason": o.finish_reason,
                        "stop_reason": o.stop_reason,
                    }
                    for o in output.outputs
                ],
            }

            # Estimate token counts
            prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0
            completion_tokens = len(completion.token_ids) if completion.token_ids else 0

            results.append(ModelOutput(
                text=completion.text,
                finish_reason=completion.finish_reason,
                usage=UsageStats(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
                raw=raw_response,
                model=self._model_id,
            ))

        logger.info(f"vLLM batch inference complete: {len(results)} outputs")
        return results


class VLLMServerModel(ChatModel):
    """vLLM OpenAI-compatible API server client.
    
    This implementation connects to a vLLM server running with
    --api-key and --served-model-name options, using the OpenAI-compatible
    /v1/chat/completions endpoint.
    
    Use this when you have a vLLM server already running, rather than
    loading the model locally.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Initialize vLLM server client.
        
        Args:
            model: Model name as configured on the vLLM server.
            base_url: Base URL of the vLLM server's OpenAI-compatible API.
            api_key: API key if server requires authentication.
            timeout: Request timeout in seconds.
            retry_config: Configuration for retry behavior.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required for VLLMServerModel. "
                "Install with: pip install openai"
            )

        self._model = model
        self._retry_config = retry_config or RetryConfig()

        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("VLLM_API_KEY", "EMPTY"),
            base_url=base_url,
            timeout=timeout,
        )

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def supports_seed(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[ChatMessage],
        gen_params: Optional[GenerationParams] = None,
    ) -> ModelOutput:
        """Generate a response via vLLM server."""
        params = gen_params or GenerationParams()
        formatted_messages = self._format_messages(messages)

        @retry_sync(self._retry_config)
        def _call_api() -> ModelOutput:
            return self._make_request(formatted_messages, params)

        return _call_api()

    def _make_request(
        self,
        messages: list[dict[str, Any]],
        params: GenerationParams,
    ) -> ModelOutput:
        """Make request to vLLM server."""
        import openai

        try:
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

            raw_response = response.model_dump() if hasattr(response, 'model_dump') else {}

            return ModelOutput(
                text=text,
                finish_reason=finish_reason,
                usage=usage,
                raw=raw_response,
                model=response.model,
            )

        except openai.APIError as e:
            raise ModelError(f"vLLM server error: {e}") from e

    def generate_batch(
        self,
        messages_list: Sequence[Sequence[ChatMessage]],
        gen_params: Optional[GenerationParams] = None,
    ) -> list[ModelOutput]:
        """Generate responses for multiple conversations.
        
        Uses concurrent execution for efficiency.
        """
        from medriskeval.models.batching import batch_generate_sync, BatchConfig

        config = BatchConfig(
            max_concurrent=10,  # vLLM servers can handle more concurrency
            show_progress=True,
        )

        result = batch_generate_sync(
            model=self,
            messages_list=messages_list,
            gen_params=gen_params,
            config=config,
        )

        outputs = []
        for output, error in zip(result.outputs, result.errors):
            if output is not None:
                outputs.append(output)
            else:
                outputs.append(ModelOutput(
                    text="",
                    finish_reason="error",
                    raw={"error": str(error)},
                    model=self._model,
                ))
        return outputs
