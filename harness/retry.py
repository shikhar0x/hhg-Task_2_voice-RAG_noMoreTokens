import time
import random
import functools
from config.logger import logger

try:
    from groq import RateLimitError as GroqRateLimitError
except ImportError:
    GroqRateLimitError = ()

def retry_step(max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 4.0):
    """
    Rate-limit aware exponential backoff decorator with jitter for resilient API calls.
    Respects HTTP 429 'Retry-After' response headers from API providers (Groq, ElevenLabs, Sarvam).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    if attempt == max_retries - 1:
                        raise e

                    # Check for rate limit / Retry-After headers from API providers
                    retry_after = None
                    is_rate_limit = False

                    # Check exception type & status code
                    status_code = getattr(e, 'status_code', None)
                    err_str = str(e).lower()
                    if (isinstance(GroqRateLimitError, type) and isinstance(e, GroqRateLimitError)) or status_code == 429 or "rate limit" in err_str or "429" in err_str:
                        is_rate_limit = True

                    # Check HTTP response headers for 'retry-after'
                    if hasattr(e, 'response') and getattr(e, 'response', None) is not None:
                        headers = getattr(e.response, 'headers', {})
                        if 'retry-after' in headers:
                            try:
                                retry_after = float(headers['retry-after'])
                            except (ValueError, TypeError):
                                pass

                    if retry_after is not None and retry_after > 0:
                        sleep_time = min(max_delay, retry_after + random.uniform(0.05, 0.20))
                        logger.warning(
                            f"⚠️ Retry attempt {attempt + 1}/{max_retries} for {func.__name__}: "
                            f"Groq 429 RateLimitError hit! Respecting Retry-After header: sleeping {sleep_time:.2f}s..."
                        )
                    elif is_rate_limit:
                        sleep_time = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0.1, 0.3)
                        logger.warning(
                            f"⚠️ Retry attempt {attempt + 1}/{max_retries} for {func.__name__}: "
                            f"Groq 429 RateLimitError hit! Backing off for {sleep_time:.2f}s (base_delay={base_delay}s)..."
                        )
                    else:
                        sleep_time = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0.01, 0.05)
                        logger.warning(
                            f"⚠️ Retry attempt {attempt + 1}/{max_retries} for {func.__name__} after {sleep_time*1000:.1f}ms due to: {e}"
                        )

                    time.sleep(sleep_time)
            raise last_err
        return wrapper
    return decorator
