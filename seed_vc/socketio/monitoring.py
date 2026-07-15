# Copyright (C) 2026 Human Dataware Lab.
# Created by HDL members
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Operational health and metrics endpoints for the Seed-VC server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from seed_vc.socketio.runtime import ServerRuntimeCoordinator

if TYPE_CHECKING:
    from seed_vc.socketio.model import VoiceConverter


def create_monitoring_router(
    model: VoiceConverter,
    runtime: ServerRuntimeCoordinator,
    max_clients: int,
) -> APIRouter:
    """Create health and Prometheus-compatible metrics routes.

    Args:
        model: Initialized Seed-VC converter.
        runtime: Shared server runtime coordinator.
        max_clients: Configured realtime client capacity.

    Returns:
        Router mounted at the application root.
    """
    router = APIRouter(tags=["operations"])

    @router.get("/health/live")
    def live() -> dict[str, str]:
        """Report that the ASGI process is serving requests."""
        return {"status": "ok"}

    @router.get("/health/ready")
    def ready() -> dict[str, Any]:
        """Report initialized model and current runtime state."""
        snapshot = runtime.metrics_snapshot()
        return {
            "status": "ready",
            "engine": type(model).__name__,
            "conversion_mode": model.conversion_mode.value,
            "input_sampling_rate": model.input_sampling_rate,
            "block_time": model.block_time,
            "reference_configured": model.get_reference_audio_path() is not None,
            "active_clients": snapshot["active_clients"],
            "max_clients": max_clients,
            "offline_job_active": snapshot["offline_job_active"],
        }

    @router.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        """Expose runtime counters and recent latency samples in Prometheus format."""
        return PlainTextResponse(_prometheus_metrics(runtime.metrics_snapshot(), max_clients))

    return router


def _prometheus_metrics(snapshot: dict[str, Any], max_clients: int) -> str:
    """Render a runtime metrics snapshot in Prometheus text exposition format."""
    scalar_metrics = {
        "realtime_vc_uptime_seconds": snapshot["uptime_seconds"],
        "realtime_vc_active_clients": snapshot["active_clients"],
        "realtime_vc_max_clients": max_clients,
        "realtime_vc_offline_job_active": int(snapshot["offline_job_active"]),
        "realtime_vc_voice_session_active": int(snapshot["voice_session_active"]),
        "realtime_vc_connections_total": snapshot["connections_total"],
        "realtime_vc_connection_rejections_total": snapshot["connection_rejections_total"],
        "realtime_vc_offline_jobs_total": snapshot["offline_jobs_total"],
        "realtime_vc_realtime_chunks_total": snapshot["realtime_chunks_total"],
        "realtime_vc_conversion_errors_total": snapshot["conversion_errors_total"],
        "realtime_vc_dropped_chunks_total": snapshot["dropped_chunks_total"],
    }
    lines = [f"{name} {value}" for name, value in scalar_metrics.items()]
    for metric_name, summary in (
        ("realtime_vc_processing_seconds", snapshot["processing_seconds"]),
        ("realtime_vc_realtime_factor", snapshot["rtf"]),
    ):
        lines.extend(
            [
                f'{metric_name}{{quantile="0.5"}} {summary["p50"]}',
                f'{metric_name}{{quantile="0.95"}} {summary["p95"]}',
                f"{metric_name}_count {summary['count']}",
                f"{metric_name}_avg {summary['average']}",
                f"{metric_name}_max {summary['max']}",
            ]
        )
    return "\n".join(lines) + "\n"
