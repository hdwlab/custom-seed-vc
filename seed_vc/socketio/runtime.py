# Copyright (C) 2025 Human Dataware Lab.
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

"""Shared runtime coordination for API and Socket.IO access."""

import threading
import time
from collections import deque
from typing import Any, Optional

from seed_vc.socketio.schemas import ConnectionErrorType


class ServerRuntimeCoordinator:
    """Coordinate shared model access between realtime and offline paths."""

    def __init__(self) -> None:
        """Initialize shared runtime state."""
        self.model_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active_client_sids: set = set()
        self._offline_job_active = False
        self._started_at = time.monotonic()
        self._connections_total = 0
        self._connection_rejections_total = 0
        self._offline_jobs_total = 0
        self._realtime_chunks_total = 0
        self._conversion_errors_total = 0
        self._dropped_chunks_total = 0
        self._processing_seconds: deque[float] = deque(maxlen=1000)
        self._rtf_values: deque[float] = deque(maxlen=1000)
        self._voice_session_sid: Optional[str] = None
        self._voice_session_restore_path: Optional[str] = None

    def client_count(self) -> int:
        """Return the number of active Socket.IO clients."""
        with self._state_lock:
            return len(self._active_client_sids)

    def has_client(self, sid: str) -> bool:
        """Return whether the client SID is currently registered.

        Args:
            sid: Client session ID.
        """
        with self._state_lock:
            return sid in self._active_client_sids

    def try_register_client(self, sid: str, max_clients: int) -> Optional[ConnectionErrorType]:
        """Try to register a realtime client.

        Args:
            sid: Client session ID.
            max_clients: Maximum number of concurrent clients allowed.

        Returns:
            The blocking error type when registration is rejected, otherwise None.
        """
        with self._state_lock:
            if self._offline_job_active:
                self._connection_rejections_total += 1
                return ConnectionErrorType.OFFLINE_BUSY
            if len(self._active_client_sids) >= max_clients:
                self._connection_rejections_total += 1
                return ConnectionErrorType.MAX_CLIENTS_REACHED
            self._active_client_sids.add(sid)
            self._connections_total += 1
        return None

    def unregister_client(self, sid: str) -> None:
        """Remove a realtime client registration if present.

        Args:
            sid: Client session ID.
        """
        with self._state_lock:
            self._active_client_sids.discard(sid)

    def try_begin_offline_job(self) -> bool:
        """Try to reserve exclusive access for an offline conversion job.

        Returns:
            True if the reservation succeeded, False otherwise.
        """
        with self._state_lock:
            if self._offline_job_active or self._active_client_sids:
                return False
            self._offline_job_active = True
            self._offline_jobs_total += 1
        return True

    def finish_offline_job(self) -> None:
        """Release the offline conversion reservation."""
        with self._state_lock:
            self._offline_job_active = False

    def record_queued_offline_job(self) -> None:
        """Count another job accepted under an existing queue reservation."""
        with self._state_lock:
            self._offline_jobs_total += 1

    def is_offline_job_active(self) -> bool:
        """Return whether an offline conversion job is currently active."""
        with self._state_lock:
            return self._offline_job_active

    def begin_voice_session(self, sid: str, restore_path: Optional[str]) -> None:
        """Record an applied operator voice session.

        Must be called while holding model_lock in the same critical section as
        the model voice application.
        """
        with self._state_lock:
            self._voice_session_sid = sid
            self._voice_session_restore_path = restore_path

    def end_voice_session(self, sid: str) -> tuple[bool, Optional[str]]:
        """Clear and return the restore path when the SID owns the voice session."""
        with self._state_lock:
            if self._voice_session_sid != sid:
                return False, None
            restore_path = self._voice_session_restore_path
            self._voice_session_sid = None
            self._voice_session_restore_path = None
            return True, restore_path

    def voice_session_active(self) -> bool:
        """Return whether an operator voice session currently owns the model voice."""
        with self._state_lock:
            return self._voice_session_sid is not None

    def record_dropped_chunk(self) -> None:
        """Increment the number of realtime chunks dropped before conversion."""
        with self._state_lock:
            self._dropped_chunks_total += 1

    def record_realtime_chunk(
        self,
        processing_seconds: float,
        audio_duration_seconds: float,
        failed: bool,
    ) -> None:
        """Record bounded realtime processing metrics for one audio chunk.

        Args:
            processing_seconds: Wall-clock conversion time.
            audio_duration_seconds: Duration represented by the chunk.
            failed: Whether conversion failed and silence was emitted.
        """
        with self._state_lock:
            self._realtime_chunks_total += 1
            if failed:
                self._conversion_errors_total += 1
            self._processing_seconds.append(max(0.0, processing_seconds))
            if audio_duration_seconds > 0:
                self._rtf_values.append(max(0.0, processing_seconds / audio_duration_seconds))

    def metrics_snapshot(self) -> dict[str, Any]:
        """Return an immutable snapshot of runtime state and recent latency metrics."""
        with self._state_lock:
            processing = tuple(self._processing_seconds)
            rtf_values = tuple(self._rtf_values)
            snapshot = {
                "uptime_seconds": max(0.0, time.monotonic() - self._started_at),
                "active_clients": len(self._active_client_sids),
                "offline_job_active": self._offline_job_active,
                "voice_session_active": self._voice_session_sid is not None,
                "connections_total": self._connections_total,
                "connection_rejections_total": self._connection_rejections_total,
                "offline_jobs_total": self._offline_jobs_total,
                "realtime_chunks_total": self._realtime_chunks_total,
                "conversion_errors_total": self._conversion_errors_total,
                "dropped_chunks_total": self._dropped_chunks_total,
            }
        snapshot["processing_seconds"] = _distribution_summary(processing)
        snapshot["rtf"] = _distribution_summary(rtf_values)
        return snapshot


def _distribution_summary(values: tuple[float, ...]) -> dict[str, float | int]:
    """Summarize a bounded metric sample without third-party dependencies."""
    if not values:
        return {"count": 0, "average": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "average": sum(values) / len(values),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    """Return a nearest-rank percentile from an ordered non-empty sample."""
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * quantile)))
    return ordered[index]
