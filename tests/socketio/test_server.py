"""Tests for Socket.IO server connection limits."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from socketio.exceptions import ConnectionRefusedError as SocketIOConnectionRefused

from seed_vc.socketio import server as server_module
from seed_vc.socketio.runtime import ServerRuntimeCoordinator
from seed_vc.socketio.schemas import ConnectionErrorType
from seed_vc.socketio.server import (
    client_converters,
    connect,
    converter_init_status,
    disconnect,
)


class TestServerConnectionLimits:
    """Test cases for server connection limits."""

    def setup_method(self):
        """Setup server state before each test."""
        # Clear any existing client state
        client_converters.clear()
        converter_init_status.clear()
        server_module.runtime = ServerRuntimeCoordinator()

    def test_single_client_connection_allowed(self):
        """Test that a single client can connect successfully."""
        # Mock MAX_CLIENT = 1
        with patch("seed_vc.socketio.server.MAX_CLIENT", 1):
            with patch("seed_vc.socketio.server.global_converter", MagicMock()):
                # First client should connect successfully
                result = asyncio.run(connect("client1", {}, None))
                assert result is True
                assert "client1" in client_converters

    def test_second_client_connection_rejected(self):
        """Test that a second client is rejected when MAX_CLIENT=1."""
        # Mock MAX_CLIENT = 1
        with patch("seed_vc.socketio.server.MAX_CLIENT", 1):
            with patch("seed_vc.socketio.server.global_converter", MagicMock()):
                # First client connects successfully
                result1 = asyncio.run(connect("client1", {}, None))
                assert result1 is True

                # Second client should be rejected with ConnectionRefusedError
                with pytest.raises(SocketIOConnectionRefused) as exc_info:
                    asyncio.run(connect("client2", {}, None))

                # Check the error message content
                error_data = exc_info.value.args[0]
                assert error_data["error"] == ConnectionErrorType.MAX_CLIENTS_REACHED.value
                assert "Maximum number of clients (1) already connected" in error_data["message"]

                # Check that client2 is not in the converters dict
                assert "client2" not in client_converters

    def test_client_can_connect_after_disconnect(self):
        """Test that a new client can connect after another client disconnects."""
        # Mock MAX_CLIENT = 1
        with patch("seed_vc.socketio.server.MAX_CLIENT", 1):
            with patch("seed_vc.socketio.server.global_converter", MagicMock()):
                mock_sio = MagicMock()

                # Create a coroutine mock
                async def mock_emit(*args, **kwargs):
                    return None

                async def mock_disconnect(*args, **kwargs):
                    return None

                mock_sio.emit = MagicMock(side_effect=mock_emit)
                mock_sio.disconnect = MagicMock(side_effect=mock_disconnect)

                with patch("seed_vc.socketio.server.sio", mock_sio):
                    # First client connects
                    result1 = asyncio.run(connect("client1", {}, None))
                    assert result1 is True

                    # Disconnect first client
                    asyncio.run(disconnect("client1"))
                    assert "client1" not in client_converters

                    # Second client should now be able to connect
                    result2 = asyncio.run(connect("client2", {}, None))
                    assert result2 is True
                    assert "client2" in client_converters

    def test_multiple_clients_allowed_with_higher_limit(self):
        """Test that multiple clients can connect when MAX_CLIENT > 1."""
        # Mock MAX_CLIENT = 3
        with patch("seed_vc.socketio.server.MAX_CLIENT", 3):
            with patch("seed_vc.socketio.server.global_converter", MagicMock()):
                # Three clients should connect successfully
                result1 = asyncio.run(connect("client1", {}, None))
                result2 = asyncio.run(connect("client2", {}, None))
                result3 = asyncio.run(connect("client3", {}, None))

                assert result1 is True
                assert result2 is True
                assert result3 is True
                assert len(client_converters) == 3

    def test_fourth_client_rejected_with_limit_three(self):
        """Test that the fourth client is rejected when MAX_CLIENT=3."""
        # Mock MAX_CLIENT = 3
        with patch("seed_vc.socketio.server.MAX_CLIENT", 3):
            with patch("seed_vc.socketio.server.global_converter", MagicMock()):
                # Three clients connect successfully
                asyncio.run(connect("client1", {}, None))
                asyncio.run(connect("client2", {}, None))
                asyncio.run(connect("client3", {}, None))

                # Fourth client should be rejected with ConnectionRefusedError
                with pytest.raises(SocketIOConnectionRefused) as exc_info:
                    asyncio.run(connect("client4", {}, None))

                # Check the error message content
                error_data = exc_info.value.args[0]
                assert error_data["error"] == ConnectionErrorType.MAX_CLIENTS_REACHED.value
                assert "Maximum number of clients (3) already connected" in error_data["message"]

    def test_connection_with_chunk_size_validation_and_limit(self):
        """Test that chunk size validation still works with connection limits."""
        # Mock MAX_CLIENT = 1
        with patch("seed_vc.socketio.server.MAX_CLIENT", 1):
            mock_converter = MagicMock()
            mock_converter.block_time = 0.18
            mock_converter.block_frame = 7938
            with patch("seed_vc.socketio.server.global_converter", mock_converter):
                # Client with invalid chunk size should be rejected
                auth = {"chunk_size": 1000, "sample_rate": 44100}
                with pytest.raises(SocketIOConnectionRefused) as exc_info:
                    asyncio.run(connect("client1", {}, auth))

                # Check the error message content
                error_data = exc_info.value.args[0]
                assert error_data["error"] == ConnectionErrorType.CHUNK_SIZE_MISMATCH.value
                assert error_data["client_chunk_size"] == 1000
                assert error_data["expected_chunk_size"] == 7938

    def test_chunk_size_uses_zc_aligned_block_frame(self):
        """Test that chunk size validation uses zc-aligned block_frame from converter."""
        with patch("seed_vc.socketio.server.MAX_CLIENT", 1):
            mock_converter = MagicMock()
            mock_converter.block_time = 0.25
            # zc-aligned: int(round(0.25 * 44100 / 882)) * 882 = round(12.5) * 882 = 10584
            mock_converter.block_frame = 10584
            with patch("seed_vc.socketio.server.global_converter", mock_converter):
                # Client sending zc-aligned chunk size should be accepted
                auth = {"chunk_size": 10584, "sample_rate": 44100}
                result = asyncio.run(connect("client_aligned", {}, auth))
                assert result is True
                asyncio.run(disconnect("client_aligned"))

                # Client sending simple int(0.25*44100)=11025 should be rejected
                auth_wrong = {"chunk_size": 11025, "sample_rate": 44100}
                with pytest.raises(SocketIOConnectionRefused) as exc_info:
                    asyncio.run(connect("client_wrong", {}, auth_wrong))
                error_data = exc_info.value.args[0]
                assert error_data["expected_chunk_size"] == 10584

    def test_client_disconnect_reduces_connection_count(self):
        """Test that client disconnection properly reduces the connection count."""
        # Mock MAX_CLIENT = 2
        with patch("seed_vc.socketio.server.MAX_CLIENT", 2):
            with patch("seed_vc.socketio.server.global_converter", MagicMock()):
                # Connect two clients
                result1 = asyncio.run(connect("client1", {}, None))
                result2 = asyncio.run(connect("client2", {}, None))
                assert result1 is True
                assert result2 is True
                assert len(client_converters) == 2

                # Disconnect first client
                asyncio.run(disconnect("client1"))
                assert len(client_converters) == 1
                assert "client1" not in client_converters
                assert "client2" in client_converters

                # Third client should now be able to connect
                result3 = asyncio.run(connect("client3", {}, None))
                assert result3 is True
                assert len(client_converters) == 2
                assert "client3" in client_converters

    def test_multiple_disconnects_maintain_correct_count(self):
        """Test that multiple disconnections maintain correct connection count."""
        # Mock MAX_CLIENT = 3
        with patch("seed_vc.socketio.server.MAX_CLIENT", 3):
            with patch("seed_vc.socketio.server.global_converter", MagicMock()):
                # Connect three clients
                asyncio.run(connect("client1", {}, None))
                asyncio.run(connect("client2", {}, None))
                asyncio.run(connect("client3", {}, None))
                assert len(client_converters) == 3

                # Disconnect all clients one by one
                asyncio.run(disconnect("client1"))
                assert len(client_converters) == 2

                asyncio.run(disconnect("client2"))
                assert len(client_converters) == 1

                asyncio.run(disconnect("client3"))
                assert len(client_converters) == 0

                # All three clients should be able to reconnect now
                result1 = asyncio.run(connect("client1", {}, None))
                result2 = asyncio.run(connect("client2", {}, None))
                result3 = asyncio.run(connect("client3", {}, None))
                assert result1 is True
                assert result2 is True
                assert result3 is True
                assert len(client_converters) == 3


class TestOfflineJobConnectionExclusion:
    """Test cases for connection exclusion during offline conversion jobs."""

    def setup_method(self):
        """Setup server state before each test."""
        client_converters.clear()
        converter_init_status.clear()
        server_module.runtime = ServerRuntimeCoordinator()

    def test_connection_rejected_while_offline_job_active(self):
        """Test that a client is rejected while an offline conversion job is active."""
        with patch("seed_vc.socketio.server.MAX_CLIENT", 1):
            with patch("seed_vc.socketio.server.global_converter", MagicMock()):
                assert server_module.runtime.try_begin_offline_job()

                with pytest.raises(SocketIOConnectionRefused) as exc_info:
                    asyncio.run(connect("client1", {}, None))

                error_data = exc_info.value.args[0]
                assert error_data["error"] == ConnectionErrorType.OFFLINE_BUSY.value
                assert "client1" not in client_converters

    def test_connection_allowed_after_offline_job_finished(self):
        """Test that a client can connect after the offline job finishes."""
        with patch("seed_vc.socketio.server.MAX_CLIENT", 1):
            with patch("seed_vc.socketio.server.global_converter", MagicMock()):
                server_module.runtime.try_begin_offline_job()
                server_module.runtime.finish_offline_job()

                result = asyncio.run(connect("client1", {}, None))

                assert result is True
                assert "client1" in client_converters

    def test_offline_job_rejected_while_client_connected(self):
        """Test that an offline job cannot start while a client is connected."""
        with patch("seed_vc.socketio.server.MAX_CLIENT", 1):
            with patch("seed_vc.socketio.server.global_converter", MagicMock()):
                asyncio.run(connect("client1", {}, None))

                assert server_module.runtime.try_begin_offline_job() is False


class TestLoadConverterConfig:
    """Test cases for loading VoiceConverter config from YAML."""

    def test_none_returns_empty_dict(self):
        """Test that None config path returns an empty dict."""
        from seed_vc.socketio.server import load_converter_config

        assert load_converter_config(None) == {}

    def test_yaml_values_loaded(self, tmp_path):
        """Test that YAML values are loaded as a dict."""
        from seed_vc.socketio.server import load_converter_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text("block_time: 0.12\ndiffusion_steps: 6\nuse_vad: false\n")

        config = load_converter_config(str(config_path))

        assert config == {"block_time": 0.12, "diffusion_steps": 6, "use_vad": False}

    def test_empty_file_returns_empty_dict(self, tmp_path):
        """Test that an empty YAML file returns an empty dict."""
        from seed_vc.socketio.server import load_converter_config

        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        assert load_converter_config(str(config_path)) == {}

    def test_missing_file_raises(self):
        """Test that a missing config file raises FileNotFoundError."""
        from seed_vc.socketio.server import load_converter_config

        with pytest.raises(FileNotFoundError):
            load_converter_config("/no/such/config.yaml")


class TestInitializeGlobalConverterWithConfig:
    """Test cases for config overrides in initialize_global_converter."""

    def test_config_overrides_defaults(self, monkeypatch):
        """Test that YAML-provided kwargs override the hardcoded defaults."""
        from seed_vc.socketio import server

        monkeypatch.setattr(server, "global_converter", None)
        with patch("seed_vc.socketio.server.VoiceConverter") as mock_vc:
            mock_vc.return_value = MagicMock()
            result = server.initialize_global_converter(
                log_level="INFO",
                converter_kwargs={"block_time": 0.12, "use_vad": False},
            )

        assert result is not None
        _args, kwargs = mock_vc.call_args
        assert kwargs["block_time"] == 0.12
        assert kwargs["use_vad"] is False
        # Unspecified parameters keep their defaults
        assert kwargs["diffusion_steps"] == 10
        assert kwargs["input_sampling_rate"] == 44100

    def test_defaults_used_without_config(self, monkeypatch):
        """Test that defaults are used when no config is provided."""
        from seed_vc.socketio import server

        monkeypatch.setattr(server, "global_converter", None)
        with patch("seed_vc.socketio.server.VoiceConverter") as mock_vc:
            mock_vc.return_value = MagicMock()
            server.initialize_global_converter(log_level="DEBUG")

        _args, kwargs = mock_vc.call_args
        assert kwargs["block_time"] == 0.18
        assert kwargs["use_vad"] is True
        assert kwargs["log_level"] == "DEBUG"
