import logging
from rich.logging import RichHandler
from config.settings import settings

def setup_logger(name: str = "voice_rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = getattr(logging, settings.log_level.upper(), logging.INFO)
        logger.setLevel(level)
        handler = RichHandler(rich_tracebacks=True, markup=True)
        formatter = logging.Formatter("[%(name)s] %(message)s", datefmt="[%X]")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()
