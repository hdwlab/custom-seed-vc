"""Integration tests for Socket.IO server and client connection limits."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import socketio
from socketio.exceptions import ConnectionRefusedError as SocketIOConnectionRefusedError

from seed_vc.socketio.schemas import ConnectionErrorType


class TestSocketIOIntegration:
    """Integration test cases for Socket.IO server-client connection limits."""

    @pytest.fixture
    def mock_voice_converter(self):
        """Mock VoiceConverter for testing."""
        mock_converter = MagicMock()
        mock_converter.input_sampling_rate = 44100
        mock_converter.block_time = 0.18
        mock_converter.block_frame = 7938
        return mock_converter

    def test_max_client_connection_limit_integration(self, mock_voice_converter):
        """Test that client receives proper error message when max clients reached."""
        import seed_vc.socketio.server as server_module

        # Mock the global converter and set MAX_CLIENT to 1
        with patch.object(server_module, "global_converter", mock_voice_converter):
            with patch.object(server_module, "MAX_CLIENT", 1):
                # Create two test clients
                client1 = socketio.Client()
                client2 = socketio.Client()

                client1_error = {}
                client2_error = {}

                @client1.on("connection_error")
                def client1_on_error(data):
                    client1_error.update(data)

                @client2.on("connection_error")
                def client2_on_error(data):
                    client2_error.update(data)

                # Test server setup with mocked functions
                with patch("seed_vc.socketio.server.uvicorn.run"):
                    with patch(
                        "seed_vc.socketio.server.initialize_global_converter",
                        return_value=mock_voice_converter,
                    ):
                        # Start server in a separate thread (mocked)
                        _ = "http://localhost:5001"

                        # Manually test the connect function with mocked sio
                        from seed_vc.socketio.server import connect

                        # Mock the sio.emit and sio.disconnect methods
                        mock_sio = MagicMock()

                        async def mock_emit(event, data, to=None):
                            # Simulate sending error to client2
                            if to == "client2" and event == "connection_error":
                                client2_error.update(data)

                        async def mock_disconnect(sid):
                            pass

                        mock_sio.emit = mock_emit
                        mock_sio.disconnect = mock_disconnect

                        # Simulate first client connecting successfully
                        result1 = asyncio.run(
                            connect("client1", {}, {"chunk_size": 7938, "sample_rate": 44100})
                        )
                        assert result1 is True

                        # Simulate second client being rejected with ConnectionRefusedError
                        with pytest.raises(SocketIOConnectionRefusedError) as exc_info:
                            asyncio.run(
                                connect("client2", {}, {"chunk_size": 7938, "sample_rate": 44100})
                            )

                        # Check that the ConnectionRefusedError contains the correct error data
                        error_data = exc_info.value.args[0]
                        assert error_data["error"] == ConnectionErrorType.MAX_CLIENTS_REACHED.value
                        assert (
                            "Maximum number of clients (1) already connected"
                            in error_data["message"]
                        )

    def test_client_error_message_display(self):
        """Test that client properly displays max clients reached error message."""
        import seed_vc.socketio.client as client_module

        # Set up the error scenario
        client_module.connection_error_details.clear()
        client_module.connection_error_details.update(
            {
                "error": ConnectionErrorType.MAX_CLIENTS_REACHED.value,
                "message": "Maximum number of clients (1) already connected",
            }
        )

        # Mock the socketio client and logger
        with patch.object(client_module, "sio") as mock_sio:
            with patch.object(client_module, "logger") as mock_logger:
                # Simulate connection failure
                mock_sio.connect.side_effect = socketio.exceptions.ConnectionError(
                    "Connection refused"
                )

                # Mock command line arguments
                with patch("sys.argv", ["client.py", "--host", "localhost", "--port", "5001"]):
                    try:
                        from seed_vc.socketio.client import main

                        main()
                    except SystemExit:
                        pass  # Expected when connection fails

                # Verify that the appropriate error messages were logged
                error_calls = [
                    call
                    for call in mock_logger.error.call_args_list
                    if "Connection rejected" in str(call) or "wait for another client" in str(call)
                ]
                assert len(error_calls) >= 2  # Should have both error messages

    def test_multiple_client_limits(self, mock_voice_converter):
        """Test that server can handle different MAX_CLIENT values."""
        import seed_vc.socketio.server as server_module

        # Test with MAX_CLIENT = 3
        with patch.object(server_module, "global_converter", mock_voice_converter):
            with patch.object(server_module, "MAX_CLIENT", 3):
                # Mock sio for testing
                mock_sio = MagicMock()

                async def mock_emit(event, data, to=None):
                    pass

                async def mock_disconnect(sid):
                    pass

                mock_sio.emit = mock_emit
                mock_sio.disconnect = mock_disconnect

                with patch.object(server_module, "sio", mock_sio):
                    from seed_vc.socketio.server import connect

                    # First three clients should connect successfully
                    result1 = asyncio.run(
                        connect("client1", {}, {"chunk_size": 7938, "sample_rate": 44100})
                    )
                    result2 = asyncio.run(
                        connect("client2", {}, {"chunk_size": 7938, "sample_rate": 44100})
                    )
                    result3 = asyncio.run(
                        connect("client3", {}, {"chunk_size": 7938, "sample_rate": 44100})
                    )

                    assert result1 is True
                    assert result2 is True
                    assert result3 is True

                    # Fourth client should be rejected with ConnectionRefusedError
                    with pytest.raises(SocketIOConnectionRefusedError) as exc_info:
                        asyncio.run(
                            connect("client4", {}, {"chunk_size": 7938, "sample_rate": 44100})
                        )

                    # Check that the ConnectionRefusedError contains the correct error data
                    error_data = exc_info.value.args[0]
                    assert error_data["error"] == ConnectionErrorType.MAX_CLIENTS_REACHED.value
                    assert (
                        "Maximum number of clients (3) already connected" in error_data["message"]
                    )
