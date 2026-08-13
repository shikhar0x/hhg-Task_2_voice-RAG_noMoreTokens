import time
import random
import functools
from config.logger import logger

def retry_step(max_retries: int = 3, base_delay: float = 0.05, max_delay: float = 0.5):
    """Exponential backoff decorator with jitter for resilient network/API calls."""
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
                    sleep_time = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 0.02)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {sleep_time*1000:.1f}ms due to: {e}")
                    time.sleep(sleep_time)
            raise last_err
        return wrapper
    return decorator
