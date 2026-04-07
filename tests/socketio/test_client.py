"""Tests for Socket.IO client connection error handling and queue behavior."""

from unittest.mock import patch

import socketio

import seed_vc.socketio.client as client_module
from seed_vc.socketio.client import on_connect_error, on_connection_error
from seed_vc.socketio.schemas import ConnectionErrorType


class TestClientConnectionErrors:
    """Test cases for client connection error handling."""

    def setup_method(self):
        """Setup test state."""
        # Clear global connection error details
        client_module.connection_error_details.clear()

    def test_chunk_size_mismatch_error_handling(self):
        """Test that chunk size mismatch errors are properly handled."""
        error_data = {
            "error": ConnectionErrorType.CHUNK_SIZE_MISMATCH.value,
            "client_chunk_size": 1000,
            "expected_chunk_size": 7938,
            "block_time": 0.18,
            "sample_rate": 44100,
        }

        # Simulate server sending connection_error
        on_connection_error(error_data)

        # Check that error details are stored
        assert (
            client_module.connection_error_details["error"]
            == ConnectionErrorType.CHUNK_SIZE_MISMATCH.value
        )
        assert client_module.connection_error_details["client_chunk_size"] == 1000
        assert client_module.connection_error_details["expected_chunk_size"] == 7938

    def test_max_clients_reached_error_handling(self):
        """Test that max clients reached errors are properly handled."""
        error_data = {
            "error": ConnectionErrorType.MAX_CLIENTS_REACHED.value,
            "message": "Maximum number of clients (1) already connected",
        }

        # Simulate server sending connection_error
        on_connection_error(error_data)

        # Check that error details are stored
        assert (
            client_module.connection_error_details["error"]
            == ConnectionErrorType.MAX_CLIENTS_REACHED.value
        )
        assert client_module.connection_error_details["message"] == (
            "Maximum number of clients (1) already connected"
        )

    def test_connection_error_logging_chunk_size_mismatch(self, caplog):
        """Test that chunk size mismatch errors are logged properly."""
        # Set up error details
        client_module.connection_error_details.update(
            {
                "error": ConnectionErrorType.CHUNK_SIZE_MISMATCH.value,
                "client_chunk_size": 1000,
                "expected_chunk_size": 7938,
                "block_time": 0.18,
                "sample_rate": 44100,
            }
        )

        with patch("seed_vc.socketio.client.sio") as mock_sio:
            # Mock connection failure
            mock_sio.connect.side_effect = socketio.exceptions.ConnectionError("Connection failed")

            # Import here to avoid circular import and capture the actual error handling
            from seed_vc.socketio.client import main

            with patch("sys.argv", ["client.py", "--host", "localhost", "--port", "5000"]):
                with patch("seed_vc.socketio.client.logger") as mock_logger:
                    try:
                        main()
                    except SystemExit:
                        pass  # main() calls return which may cause SystemExit in test

                    # Check that appropriate error messages were logged
                    mock_logger.error.assert_any_call(
                        "❌ Failed to connect to server: %s", mock_sio.connect.side_effect
                    )

                    # Check chunk size mismatch specific logging
                    calls = [
                        call
                        for call in mock_logger.error.call_args_list
                        if "Chunk size mismatch" in str(call)
                    ]
                    assert len(calls) > 0

    def test_connection_error_logging_max_clients_reached(self, caplog):
        """Test that max clients reached errors are logged properly."""
        # Set up error details
        client_module.connection_error_details.update(
            {
                "error": ConnectionErrorType.MAX_CLIENTS_REACHED.value,
                "message": "Maximum number of clients (1) already connected",
            }
        )

        with patch("seed_vc.socketio.client.sio") as mock_sio:
            # Mock connection failure
            mock_sio.connect.side_effect = socketio.exceptions.ConnectionError("Connection failed")

            # Import here to avoid circular import and capture the actual error handling
            from seed_vc.socketio.client import main

            with patch("sys.argv", ["client.py", "--host", "localhost", "--port", "5000"]):
                with patch("seed_vc.socketio.client.logger") as mock_logger:
                    try:
                        main()
                    except SystemExit:
                        pass  # main() calls return which may cause SystemExit in test

                    # Check that appropriate error messages were logged
                    mock_logger.error.assert_any_call(
                        "❌ Failed to connect to server: %s", mock_sio.connect.side_effect
                    )

                    # Check max clients specific logging
                    calls = [
                        call
                        for call in mock_logger.error.call_args_list
                        if "Connection rejected" in str(call)
                    ]
                    assert len(calls) > 0

                    wait_calls = [
                        call
                        for call in mock_logger.error.call_args_list
                        if "wait for another client" in str(call)
                    ]
                    assert len(wait_calls) > 0

    def test_unknown_error_handling(self):
        """Test that unknown errors are handled gracefully."""
        error_data = {"error": "unknown_error", "message": "Some unknown error occurred"}

        # Simulate server sending connection_error
        on_connection_error(error_data)

        # Check that error details are stored
        assert client_module.connection_error_details["error"] == "unknown_error"
        assert client_module.connection_error_details["message"] == "Some unknown error occurred"

    def test_connect_error_max_clients_reached(self):
        """Test that connect_error event handles max_clients_reached properly."""
        error_data = {
            "error": ConnectionErrorType.MAX_CLIENTS_REACHED.value,
            "message": "Maximum number of clients (1) already connected",
        }

        # Simulate server sending connect_error (when connection is refused)
        on_connect_error(error_data)

        # Check that error details are stored
        assert (
            client_module.connection_error_details["error"]
            == ConnectionErrorType.MAX_CLIENTS_REACHED.value
        )
        assert client_module.connection_error_details["message"] == (
            "Maximum number of clients (1) already connected"
        )

    def test_connect_error_chunk_size_mismatch(self):
        """Test that connect_error event handles chunk_size_mismatch properly."""
        error_data = {
            "error": ConnectionErrorType.CHUNK_SIZE_MISMATCH.value,
            "client_chunk_size": 1000,
            "expected_chunk_size": 7938,
            "block_time": 0.18,
            "sample_rate": 44100,
        }

        # Simulate server sending connect_error (when connection is refused)
        on_connect_error(error_data)

        # Check that error details are stored
        assert (
            client_module.connection_error_details["error"]
            == ConnectionErrorType.CHUNK_SIZE_MISMATCH.value
        )
        assert client_module.connection_error_details["client_chunk_size"] == 1000
        assert client_module.connection_error_details["expected_chunk_size"] == 7938


class TestQueueDraining:
    """Test cases for queue size limiting to prevent unbounded latency."""

    def setup_method(self):
        """Clear queues before each test."""
        while not client_module.play_q.empty():
            client_module.play_q.get_nowait()
        while not client_module.send_q.empty():
            client_module.send_q.get_nowait()

    def test_max_queue_size_constant_exists(self):
        """Test that MAX_QUEUE_SIZE constant is defined."""
        assert hasattr(client_module, "MAX_QUEUE_SIZE")
        assert client_module.MAX_QUEUE_SIZE == 1

    def test_play_queue_drains_stale_chunks(self):
        """Test that stale chunks are drained when play_q exceeds limit."""
        # Fill play_q with 5 chunks
        for i in range(5):
            client_module.play_q.put(f"chunk_{i}".encode())

        # Simulate consumer: get first, then drain if over limit
        chunk = client_module.play_q.get()
        assert client_module.play_q.qsize() > client_module.MAX_QUEUE_SIZE

        skipped = 0
        while not client_module.play_q.empty():
            chunk = client_module.play_q.get()
            skipped += 1

        assert skipped == 4
        assert chunk == b"chunk_4"
        assert client_module.play_q.empty()

    def test_send_queue_drains_stale_chunks(self):
        """Test that stale chunks are drained when send_q exceeds limit."""
        # Fill send_q with 5 chunks
        for i in range(5):
            client_module.send_q.put(f"chunk_{i}".encode())

        # Simulate consumer: get first, then drain if over limit
        chunk = client_module.send_q.get()
        assert client_module.send_q.qsize() > client_module.MAX_QUEUE_SIZE

        skipped = 0
        while not client_module.send_q.empty():
            chunk = client_module.send_q.get()
            skipped += 1

        assert skipped == 4
        assert chunk == b"chunk_4"
        assert client_module.send_q.empty()

    def test_no_drain_when_queue_within_limit(self):
        """Test that no draining occurs when queue is within MAX_QUEUE_SIZE."""
        client_module.play_q.put(b"only_chunk")
        chunk = client_module.play_q.get()
        assert client_module.play_q.qsize() <= client_module.MAX_QUEUE_SIZE
        assert chunk == b"only_chunk"
