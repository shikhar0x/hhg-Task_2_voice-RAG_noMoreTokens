import pytest
import time
from harness.base import BaseStep, StepResult
from harness.retry import retry_step

class DummySuccessStep(BaseStep):
    name = "dummy_success"

    def execute(self, input_data: dict) -> StepResult:
        time.sleep(0.01)
        return StepResult(success=True, data={"value": input_data.get("input", 0) * 2})

class DummyFailingStep(BaseStep):
    name = "dummy_failing"

    def execute(self, input_data: dict) -> StepResult:
        raise ValueError("Simulated step failure")

def test_base_step_success_timing():
    step = DummySuccessStep()
    res = step.run({"input": 21})
    assert res.success is True
    assert res.data == {"value": 42}
    assert res.duration_ms > 0.0
    assert res.error is None

def test_base_step_exception_handling():
    step = DummyFailingStep()
    res = step.run({})
    assert res.success is False
    assert res.error == "Simulated step failure"
    assert res.duration_ms > 0.0

def test_retry_step_eventual_failure():
    attempts = 0

    @retry_step(max_retries=3, base_delay=0.01, max_delay=0.02)
    def always_fails():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("Persistent error")

    with pytest.raises(RuntimeError) as exc_info:
        always_fails()

    assert "Persistent error" in str(exc_info.value)
    assert attempts == 3

def test_retry_step_success_after_retries():
    attempts = 0

    @retry_step(max_retries=3, base_delay=0.01, max_delay=0.02)
    def succeeds_on_third_try():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError(f"Temporary failure {attempts}")
        return "success"

    result = succeeds_on_third_try()
    assert result == "success"
    assert attempts == 3
