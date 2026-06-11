"""Tests for the server runtime coordinator."""

from seed_vc.socketio.runtime import ServerRuntimeCoordinator
from seed_vc.socketio.schemas import ConnectionErrorType


class TestClientRegistration:
    """Test realtime client registration."""

    def test_register_client(self):
        """Test that a client can be registered."""
        runtime = ServerRuntimeCoordinator()

        error = runtime.try_register_client("client1", max_clients=1)

        assert error is None
        assert runtime.client_count() == 1
        assert runtime.has_client("client1")

    def test_register_client_rejected_when_max_clients_reached(self):
        """Test that registration is rejected when max clients is reached."""
        runtime = ServerRuntimeCoordinator()
        runtime.try_register_client("client1", max_clients=1)

        error = runtime.try_register_client("client2", max_clients=1)

        assert error == ConnectionErrorType.MAX_CLIENTS_REACHED
        assert runtime.client_count() == 1
        assert not runtime.has_client("client2")

    def test_unregister_client(self):
        """Test that a client can be unregistered and a new one can register."""
        runtime = ServerRuntimeCoordinator()
        runtime.try_register_client("client1", max_clients=1)

        runtime.unregister_client("client1")

        assert runtime.client_count() == 0
        assert runtime.try_register_client("client2", max_clients=1) is None

    def test_unregister_unknown_client_is_noop(self):
        """Test that unregistering an unknown client does not raise."""
        runtime = ServerRuntimeCoordinator()

        runtime.unregister_client("unknown")

        assert runtime.client_count() == 0


class TestOfflineJobExclusion:
    """Test exclusion between offline jobs and realtime clients."""

    def test_begin_offline_job(self):
        """Test that an offline job can begin when idle."""
        runtime = ServerRuntimeCoordinator()

        assert runtime.try_begin_offline_job() is True
        assert runtime.is_offline_job_active()

    def test_second_offline_job_rejected(self):
        """Test that a second offline job is rejected while one is active."""
        runtime = ServerRuntimeCoordinator()
        runtime.try_begin_offline_job()

        assert runtime.try_begin_offline_job() is False

    def test_offline_job_rejected_while_clients_connected(self):
        """Test that an offline job is rejected while a client is connected."""
        runtime = ServerRuntimeCoordinator()
        runtime.try_register_client("client1", max_clients=1)

        assert runtime.try_begin_offline_job() is False

    def test_client_rejected_while_offline_job_active(self):
        """Test that client registration is rejected while an offline job is active."""
        runtime = ServerRuntimeCoordinator()
        runtime.try_begin_offline_job()

        error = runtime.try_register_client("client1", max_clients=1)

        assert error == ConnectionErrorType.OFFLINE_BUSY
        assert runtime.client_count() == 0

    def test_finish_offline_job_releases_exclusion(self):
        """Test that finishing an offline job allows clients and new jobs."""
        runtime = ServerRuntimeCoordinator()
        runtime.try_begin_offline_job()

        runtime.finish_offline_job()

        assert not runtime.is_offline_job_active()
        assert runtime.try_register_client("client1", max_clients=1) is None
