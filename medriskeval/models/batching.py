"""Batch processing utilities for efficient model inference.

This module provides utilities for batching requests to models,
handling concurrent execution, and aggregating results.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, TypeVar

from medriskeval.core.types import ChatMessage
from medriskeval.models.base import (
    ChatModel,
    GenerationParams,
    ModelError,
    ModelOutput,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class BatchConfig:
    """Configuration for batch processing.
    
    Attributes:
        batch_size: Number of items to process in each batch.
        max_concurrent: Maximum concurrent requests (for parallel execution).
        show_progress: Whether to log progress updates.
        fail_fast: Whether to stop on first failure or continue.
    """
    batch_size: int = 10
    max_concurrent: int = 5
    show_progress: bool = True
    fail_fast: bool = False


@dataclass
class BatchResult:
    """Result from batch processing.
    
    Attributes:
        outputs: List of successful outputs (None for failed items).
        errors: List of errors (None for successful items).
        success_count: Number of successful generations.
        failure_count: Number of failed generations.
    """
    outputs: list[Optional[ModelOutput]]
    errors: list[Optional[Exception]]
    success_count: int
    failure_count: int

    @property
    def all_successful(self) -> bool:
        """Whether all items succeeded."""
        return self.failure_count == 0

    def get_successful_outputs(self) -> list[ModelOutput]:
        """Return only successful outputs."""
        return [o for o in self.outputs if o is not None]


def batch_generate_sync(
    model: ChatModel,
    messages_list: Sequence[Sequence[ChatMessage]],
    gen_params: Optional[GenerationParams] = None,
    config: Optional[BatchConfig] = None,
) -> BatchResult:
    """Process multiple generation requests with batching.
    
    Uses thread pool for concurrent execution when model doesn't
    have native batch support.
    
    Args:
        model: The chat model to use for generation.
        messages_list: List of conversations to process.
        gen_params: Generation parameters for all requests.
        config: Batch processing configuration.
        
    Returns:
        BatchResult containing outputs and any errors.
    """
    cfg = config or BatchConfig()
    total = len(messages_list)
    
    outputs: list[Optional[ModelOutput]] = [None] * total
    errors: list[Optional[Exception]] = [None] * total
    success_count = 0
    failure_count = 0

    # Check if model has native batch support
    if model.supports_batching:
        try:
            batch_outputs = model.generate_batch(messages_list, gen_params)
            for i, output in enumerate(batch_outputs):
                outputs[i] = output
                success_count += 1
            return BatchResult(
                outputs=outputs,
                errors=errors,
                success_count=success_count,
                failure_count=failure_count,
            )
        except ModelError as e:
            logger.warning(f"Native batch failed, falling back to sequential: {e}")

    # Fall back to concurrent execution via thread pool
    def process_item(index: int) -> tuple[int, ModelOutput]:
        return index, model.generate(messages_list[index], gen_params)

    with ThreadPoolExecutor(max_workers=cfg.max_concurrent) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_item, i): i 
            for i in range(total)
        }

        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, output = future.result()
                outputs[idx] = output
                success_count += 1
            except Exception as e:
                errors[idx] = e
                failure_count += 1
                logger.error(f"Batch item {idx} failed: {e}")
                
                if cfg.fail_fast:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

            completed += 1
            if cfg.show_progress and completed % cfg.batch_size == 0:
                logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

    if cfg.show_progress:
        logger.info(
            f"Batch complete: {success_count} succeeded, {failure_count} failed"
        )

    return BatchResult(
        outputs=outputs,
        errors=errors,
        success_count=success_count,
        failure_count=failure_count,
    )


async def batch_generate_async(
    model: ChatModel,
    messages_list: Sequence[Sequence[ChatMessage]],
    gen_params: Optional[GenerationParams] = None,
    config: Optional[BatchConfig] = None,
    generate_fn: Optional[Callable[..., Any]] = None,
) -> BatchResult:
    """Process multiple generation requests asynchronously.
    
    For models with async generate methods, this provides efficient
    concurrent execution with semaphore-based concurrency control.
    
    Args:
        model: The chat model to use.
        messages_list: List of conversations to process.
        gen_params: Generation parameters for all requests.
        config: Batch processing configuration.
        generate_fn: Optional async generate function to use instead of model.generate.
        
    Returns:
        BatchResult containing outputs and any errors.
    """
    cfg = config or BatchConfig()
    total = len(messages_list)

    outputs: list[Optional[ModelOutput]] = [None] * total
    errors: list[Optional[Exception]] = [None] * total
    success_count = 0
    failure_count = 0

    semaphore = asyncio.Semaphore(cfg.max_concurrent)
    stop_event = asyncio.Event() if cfg.fail_fast else None

    async def process_item(index: int) -> None:
        nonlocal success_count, failure_count

        if stop_event and stop_event.is_set():
            return

        async with semaphore:
            try:
                if generate_fn is not None:
                    output = await generate_fn(messages_list[index], gen_params)
                else:
                    # Wrap sync call in executor
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        None,
                        lambda: model.generate(messages_list[index], gen_params),
                    )
                outputs[index] = output
                success_count += 1
            except Exception as e:
                errors[index] = e
                failure_count += 1
                logger.error(f"Batch item {index} failed: {e}")

                if stop_event:
                    stop_event.set()

    # Create and run all tasks
    tasks = [asyncio.create_task(process_item(i)) for i in range(total)]
    
    # Track progress
    if cfg.show_progress:
        async def progress_reporter() -> None:
            while not all(t.done() for t in tasks):
                done = sum(1 for t in tasks if t.done())
                if done > 0 and done % cfg.batch_size == 0:
                    logger.info(f"Progress: {done}/{total} ({100*done/total:.1f}%)")
                await asyncio.sleep(1.0)

        progress_task = asyncio.create_task(progress_reporter())
        await asyncio.gather(*tasks, return_exceptions=True)
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
    else:
        await asyncio.gather(*tasks, return_exceptions=True)

    if cfg.show_progress:
        logger.info(
            f"Batch complete: {success_count} succeeded, {failure_count} failed"
        )

    return BatchResult(
        outputs=outputs,
        errors=errors,
        success_count=success_count,
        failure_count=failure_count,
    )


def chunk_list(items: Sequence[T], chunk_size: int) -> list[list[T]]:
    """Split a sequence into chunks of specified size.
    
    Args:
        items: Sequence to split.
        chunk_size: Maximum size of each chunk.
        
    Returns:
        List of chunks.
    """
    return [
        list(items[i : i + chunk_size])
        for i in range(0, len(items), chunk_size)
    ]


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Rough estimation of token count from text.
    
    This is a simple heuristic; for accurate counts use the model's tokenizer.
    
    Args:
        text: Input text.
        chars_per_token: Average characters per token (default ~4 for English).
        
    Returns:
        Estimated token count.
    """
    return max(1, int(len(text) / chars_per_token))


def estimate_messages_tokens(
    messages: Sequence[ChatMessage],
    chars_per_token: float = 4.0,
    overhead_per_message: int = 4,
) -> int:
    """Estimate token count for a conversation.
    
    Args:
        messages: Sequence of chat messages.
        chars_per_token: Average characters per token.
        overhead_per_message: Additional tokens per message for formatting.
        
    Returns:
        Estimated total token count.
    """
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content, chars_per_token)
        total += overhead_per_message
    return total
