"""Tests for the asynchronous offline conversion queue."""

import threading

import pytest

from seed_vc.socketio.offline_jobs import OfflineJobManager, OfflineJobQueueFullError
from seed_vc.socketio.runtime import ServerRuntimeCoordinator
from seed_vc.socketio.schemas import ConnectionErrorType


class ModelStub:
    """Minimal model state needed by OfflineJobManager."""

    def __init__(self):
        """Initialize the buffer reset count."""
        self.init_buffers_calls = 0

    def _init_buffers(self):
        """Record a clean-state reset."""
        self.init_buffers_calls += 1


def make_manager():
    """Create a manager and its shared test dependencies."""
    runtime = ServerRuntimeCoordinator()
    model = ModelStub()
    return OfflineJobManager(runtime, model), runtime, model


class TestOfflineJobManager:
    """Tests for job state, ordering, and runtime exclusion."""

    def test_successful_job_records_result_and_releases_runtime(self):
        """A completed operation exposes its result and releases the reservation."""
        manager, runtime, model = make_manager()
        try:
            job_id = manager.submit(lambda: {"message": "ok"})
            job = manager.wait(job_id, timeout=5)

            assert job is not None
            assert job["status"] == "succeeded"
            assert job["result"] == {"message": "ok"}
            assert job["error"] is None
            assert job["started_at"] is not None
            assert job["completed_at"] is not None
            assert runtime.is_offline_job_active() is False
            assert model.init_buffers_calls == 2
        finally:
            manager.shutdown()

    def test_jobs_run_in_submission_order_under_one_reservation(self):
        """Queued operations are sequential and block realtime connections as a group."""
        manager, runtime, _ = make_manager()
        started = threading.Event()
        release = threading.Event()
        execution_order = []

        def first_operation():
            started.set()
            assert release.wait(timeout=5)
            execution_order.append("first")
            return {"item": "first"}

        def second_operation():
            execution_order.append("second")
            return {"item": "second"}

        try:
            first_id = manager.submit(first_operation)
            assert started.wait(timeout=5)
            second_id = manager.submit(second_operation)

            assert runtime.try_register_client("sid-1", max_clients=1) is (
                ConnectionErrorType.OFFLINE_BUSY
            )

            release.set()
            assert manager.wait(first_id, timeout=5)["status"] == "succeeded"
            assert manager.wait(second_id, timeout=5)["status"] == "succeeded"
            assert execution_order == ["first", "second"]
            assert runtime.is_offline_job_active() is False
            assert runtime.metrics_snapshot()["offline_jobs_total"] == 2
        finally:
            release.set()
            manager.shutdown()

    def test_failed_operation_becomes_failed_job_state(self):
        """Worker exceptions are retained instead of escaping the worker thread."""
        manager, runtime, _ = make_manager()

        def fail():
            raise RuntimeError("conversion exploded")

        try:
            job_id = manager.submit(fail)
            job = manager.wait(job_id, timeout=5)

            assert job is not None
            assert job["status"] == "failed"
            assert job["result"] is None
            assert job["error"] == "conversion exploded"
            assert runtime.is_offline_job_active() is False
        finally:
            manager.shutdown()

    def test_rejects_work_beyond_pending_job_capacity(self):
        """The executor queue is bounded independently of retained job records."""
        runtime = ServerRuntimeCoordinator()
        model = ModelStub()
        manager = OfflineJobManager(runtime, model, max_pending_jobs=1)
        started = threading.Event()
        release = threading.Event()

        def blocking_operation():
            started.set()
            assert release.wait(timeout=5)
            return {"message": "ok"}

        try:
            job_id = manager.submit(blocking_operation)
            assert started.wait(timeout=5)
            with pytest.raises(OfflineJobQueueFullError, match="queue is full"):
                manager.submit(lambda: {"message": "too many"})
            release.set()
            manager.wait(job_id, timeout=5)
        finally:
            release.set()
            manager.shutdown()

    def test_get_returns_a_detached_result_snapshot(self):
        """Callers cannot mutate a retained job through a returned result object."""
        manager, _, _ = make_manager()
        try:
            job_id = manager.submit(lambda: {"items": [{"status": "succeeded"}]})
            snapshot = manager.wait(job_id, timeout=5)
            assert snapshot is not None
            snapshot["result"]["items"][0]["status"] = "changed"

            retained = manager.get(job_id)
            assert retained is not None
            assert retained["result"]["items"][0]["status"] == "succeeded"
        finally:
            manager.shutdown()
