"""Retry utilities with exponential backoff for rate-limited APIs.

This module provides decorators and utilities for handling transient failures
and rate limits when calling model APIs.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Optional, Sequence, Type, TypeVar

from medriskeval.models.base import ModelError, RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior.
    
    Attributes:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay between retries.
        exponential_base: Base for exponential backoff calculation.
        jitter: Whether to add random jitter to delays.
        retryable_exceptions: Exception types that should trigger retry.
    """

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Sequence[Type[Exception]]] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions: tuple[Type[Exception], ...] = tuple(
            retryable_exceptions or (RateLimitError, ConnectionError, TimeoutError)
        )

    def compute_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Compute delay before next retry attempt.
        
        Args:
            attempt: Current attempt number (0-indexed).
            retry_after: Server-suggested retry delay (takes precedence).
            
        Returns:
            Delay in seconds before next retry.
        """
        if retry_after is not None:
            return min(retry_after, self.max_delay)

        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        # Add jitter to prevent thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay


# Default retry configuration
DEFAULT_RETRY_CONFIG = RetryConfig()


def retry_sync(
    config: Optional[RetryConfig] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for synchronous functions with retry logic.
    
    Args:
        config: Retry configuration. Uses defaults if not provided.
        
    Returns:
        Decorated function that automatically retries on transient failures.
        
    Example:
        @retry_sync()
        def call_api():
            return api.generate(...)
    """
    cfg = config or DEFAULT_RETRY_CONFIG

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(cfg.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except cfg.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == cfg.max_retries:
                        logger.error(
                            f"Max retries ({cfg.max_retries}) exceeded for {func.__name__}: {e}"
                        )
                        break

                    # Get retry_after hint if available (for rate limits)
                    retry_after = getattr(e, "retry_after", None)
                    delay = cfg.compute_delay(attempt, retry_after)

                    logger.warning(
                        f"Retry {attempt + 1}/{cfg.max_retries} for {func.__name__} "
                        f"after {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)

            raise last_exception or ModelError("Retry failed with no exception")

        return wrapper

    return decorator


def retry_async(
    config: Optional[RetryConfig] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for async functions with retry logic.
    
    Args:
        config: Retry configuration. Uses defaults if not provided.
        
    Returns:
        Decorated async function that automatically retries on transient failures.
        
    Example:
        @retry_async()
        async def call_api():
            return await api.generate(...)
    """
    cfg = config or DEFAULT_RETRY_CONFIG

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(cfg.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except cfg.retryable_exceptions as e:
                    last_exception = e

                    if attempt == cfg.max_retries:
                        logger.error(
                            f"Max retries ({cfg.max_retries}) exceeded for {func.__name__}: {e}"
                        )
                        break

                    retry_after = getattr(e, "retry_after", None)
                    delay = cfg.compute_delay(attempt, retry_after)

                    logger.warning(
                        f"Retry {attempt + 1}/{cfg.max_retries} for {func.__name__} "
                        f"after {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)

            raise last_exception or ModelError("Retry failed with no exception")

        return wrapper

    return decorator


class RateLimiter:
    """Token bucket rate limiter for controlling API request rates.
    
    This implementation uses a sliding window approach to limit
    requests per minute (RPM) and tokens per minute (TPM).
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 90000,
    ):
        """Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests allowed per minute.
            tokens_per_minute: Maximum tokens allowed per minute.
        """
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        
        self._request_times: list[float] = []
        self._token_counts: list[tuple[float, int]] = []
        self._lock = None  # Lazy initialization for async

    def _cleanup_old_entries(self, current_time: float) -> None:
        """Remove entries older than 1 minute."""
        cutoff = current_time - 60.0
        
        self._request_times = [t for t in self._request_times if t > cutoff]
        self._token_counts = [(t, c) for t, c in self._token_counts if t > cutoff]

    def _get_wait_time(self, estimated_tokens: int = 0) -> float:
        """Calculate wait time before next request is allowed.
        
        Args:
            estimated_tokens: Estimated tokens for the upcoming request.
            
        Returns:
            Seconds to wait (0 if request can proceed immediately).
        """
        current_time = time.time()
        self._cleanup_old_entries(current_time)

        wait_time = 0.0

        # Check RPM limit
        if len(self._request_times) >= self.rpm:
            oldest = min(self._request_times)
            wait_time = max(wait_time, oldest + 60.0 - current_time)

        # Check TPM limit
        current_tokens = sum(c for _, c in self._token_counts)
        if current_tokens + estimated_tokens > self.tpm:
            # Find when enough tokens will be freed
            sorted_counts = sorted(self._token_counts, key=lambda x: x[0])
            tokens_to_free = current_tokens + estimated_tokens - self.tpm
            freed = 0
            for t, c in sorted_counts:
                freed += c
                if freed >= tokens_to_free:
                    wait_time = max(wait_time, t + 60.0 - current_time)
                    break

        return max(0.0, wait_time)

    def acquire_sync(self, estimated_tokens: int = 0) -> None:
        """Block until rate limit allows a request (synchronous).
        
        Args:
            estimated_tokens: Estimated tokens for the request.
        """
        wait_time = self._get_wait_time(estimated_tokens)
        if wait_time > 0:
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
            time.sleep(wait_time)

        current_time = time.time()
        self._request_times.append(current_time)
        if estimated_tokens > 0:
            self._token_counts.append((current_time, estimated_tokens))

    async def acquire_async(self, estimated_tokens: int = 0) -> None:
        """Block until rate limit allows a request (asynchronous).
        
        Args:
            estimated_tokens: Estimated tokens for the request.
        """
        wait_time = self._get_wait_time(estimated_tokens)
        if wait_time > 0:
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        current_time = time.time()
        self._request_times.append(current_time)
        if estimated_tokens > 0:
            self._token_counts.append((current_time, estimated_tokens))

    def record_usage(self, tokens_used: int) -> None:
        """Record actual token usage after a request completes.
        
        Call this to update the rate limiter with actual token counts
        if they differ from the estimate provided to acquire_*.
        
        Args:
            tokens_used: Actual number of tokens used.
        """
        current_time = time.time()
        self._token_counts.append((current_time, tokens_used))
