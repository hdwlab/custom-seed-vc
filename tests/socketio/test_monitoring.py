"""Tests for operational health and metrics endpoints."""

from typing import Any, Callable
from unittest.mock import MagicMock

from fastapi.routing import APIRoute

from seed_vc.socketio.monitoring import create_monitoring_router
from seed_vc.socketio.runtime import ServerRuntimeCoordinator


def _endpoints(
    runtime: ServerRuntimeCoordinator | None = None,
) -> dict[str, Callable[[], Any]]:
    model = MagicMock()
    model.conversion_mode.value = "convert"
    model.input_sampling_rate = 44100
    model.block_time = 0.18
    model.get_reference_audio_path.return_value = None
    router = create_monitoring_router(model, runtime or ServerRuntimeCoordinator(), 1)
    return {route.path: route.endpoint for route in router.routes if isinstance(route, APIRoute)}


def test_liveness_endpoint() -> None:
    """Liveness reports that the ASGI process is serving requests."""
    response = _endpoints()["/health/live"]()

    assert response == {"status": "ok"}


def test_readiness_endpoint_reports_model_state() -> None:
    """Readiness includes the initialized model and runtime capacity."""
    response = _endpoints()["/health/ready"]()

    assert response == {
        "status": "ready",
        "engine": "MagicMock",
        "conversion_mode": "convert",
        "input_sampling_rate": 44100,
        "block_time": 0.18,
        "reference_configured": False,
        "active_clients": 0,
        "max_clients": 1,
        "offline_job_active": False,
    }


def test_metrics_endpoint_reports_runtime_counters() -> None:
    """Metrics expose session, failure, drop, timing, and RTF values."""
    runtime = ServerRuntimeCoordinator()
    assert runtime.try_register_client("sid-1", 1) is None
    runtime.record_realtime_chunk(0.02, 0.01, failed=True)
    runtime.record_dropped_chunk()

    response = _endpoints(runtime)["/metrics"]()
    body = response.body.decode()

    assert response.status_code == 200
    assert response.media_type == "text/plain"
    assert "realtime_vc_active_clients 1" in body
    assert "realtime_vc_connections_total 1" in body
    assert "realtime_vc_realtime_chunks_total 1" in body
    assert "realtime_vc_conversion_errors_total 1" in body
    assert "realtime_vc_dropped_chunks_total 1" in body
    assert 'realtime_vc_realtime_factor{quantile="0.95"} 2.0' in body
