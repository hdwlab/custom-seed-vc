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

"""Single-worker in-process queue for asynchronous offline conversions."""

from __future__ import annotations

import copy
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from seed_vc.socketio.runtime import ServerRuntimeCoordinator

if TYPE_CHECKING:
    from seed_vc.socketio.model import VoiceConverter


class OfflineJobBusyError(RuntimeError):
    """Raised when realtime or synchronous offline work owns the model."""


class OfflineJobQueueFullError(RuntimeError):
    """Raised when the bounded in-process queue has no remaining capacity."""


class OfflineJobManager:
    """Run queued offline jobs sequentially against the shared SeedVC model."""

    def __init__(
        self,
        runtime: ServerRuntimeCoordinator,
        model: VoiceConverter,
        max_records: int = 100,
        max_pending_jobs: int = 100,
    ) -> None:
        """Initialize the queue without starting a worker until first submit."""
        if max_records < 1:
            raise ValueError("max_records must be at least 1")
        if max_pending_jobs < 1:
            raise ValueError("max_pending_jobs must be at least 1")
        self.runtime = runtime
        self.model = model
        self.max_records = max_records
        self.max_pending_jobs = max_pending_jobs
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="offline-vc")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future[None]] = {}
        self._outstanding = 0
        self._closed = False

    def submit(self, operation: Callable[[], dict[str, Any]]) -> str:
        """Queue an operation and reserve the model until all queued work completes."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Offline job manager is closed")
            if self._outstanding >= self.max_pending_jobs:
                raise OfflineJobQueueFullError("Offline job queue is full")
            owns_reservation = self._outstanding > 0
            if not owns_reservation and not self.runtime.try_begin_offline_job():
                raise OfflineJobBusyError("Voice conversion model is busy")
            if owns_reservation:
                self.runtime.record_queued_offline_job()
            job_id = uuid.uuid4().hex
            self._jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "created_at": _utc_now_iso(),
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None,
            }
            self._outstanding += 1
            try:
                self._futures[job_id] = self._executor.submit(self._run_job, job_id, operation)
            except Exception:
                self._outstanding -= 1
                self._jobs.pop(job_id, None)
                if not owns_reservation:
                    self.runtime.finish_offline_job()
                raise
            self._prune_records()
            return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Return a detached job snapshot, or None for an unknown ID."""
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else copy.deepcopy(job)

    def wait(self, job_id: str, timeout: float | None = None) -> dict[str, Any] | None:
        """Wait for a job to finish and return its snapshot."""
        with self._lock:
            future = self._futures.get(job_id)
        if future is None:
            return self.get(job_id)
        future.result(timeout=timeout)
        return self.get(job_id)

    def shutdown(self) -> None:
        """Stop accepting work and wait for queued jobs during graceful shutdown."""
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)

    def _run_job(self, job_id: str, operation: Callable[[], dict[str, Any]]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
            job["started_at"] = _utc_now_iso()
        try:
            with self.runtime.model_lock:
                self.model._init_buffers()
                try:
                    result = operation()
                finally:
                    self.model._init_buffers()
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "succeeded"
                job["result"] = result
        except Exception as error:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["error"] = str(error)
        finally:
            with self._lock:
                self._jobs[job_id]["completed_at"] = _utc_now_iso()
                self._outstanding -= 1
                if self._outstanding == 0:
                    self.runtime.finish_offline_job()
                self._prune_records()

    def _prune_records(self) -> None:
        if len(self._jobs) <= self.max_records:
            return
        for job_id, job in tuple(self._jobs.items()):
            if len(self._jobs) <= self.max_records:
                break
            if job["status"] in {"succeeded", "failed"}:
                self._jobs.pop(job_id, None)
                self._futures.pop(job_id, None)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
