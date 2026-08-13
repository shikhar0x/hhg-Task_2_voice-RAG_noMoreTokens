import time
from dataclasses import dataclass, field
from typing import Any, Optional
from config.logger import logger

@dataclass
class StepResult:
    """Standardized result returned by all pipeline steps (ported from Nexa's SkillResult)."""
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0
    refused: bool = False
    refusal_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

class BaseStep:
    """Base class for structured pipeline harness steps with built-in telemetry."""
    name: str = "base_step"

    def run(self, input_data: dict[str, Any]) -> StepResult:
        start_time = time.perf_counter()
        try:
            result = self.execute(input_data)
            result.duration_ms = (time.perf_counter() - start_time) * 1000.0
            return result
        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Step '{self.name}' failed after {duration:.2f}ms: {e}")
            return StepResult(
                success=False,
                error=str(e),
                duration_ms=duration
            )

    def execute(self, input_data: dict[str, Any]) -> StepResult:
        raise NotImplementedError
