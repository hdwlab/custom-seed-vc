"""Tests for RTF (Real-Time Factor) monitoring functionality."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np

from seed_vc.socketio.server import (
    audio_chunk,
    warn_if_rtf_slow,
)


class TestRTFMonitoring:
    """Test cases for RTF monitoring and warning functionality."""

    def test_rtf_normal_processing_no_warning(self, caplog):
        """Test RTF calculation for normal processing (RTF < 1.0) - no warning."""
        with patch("seed_vc.socketio.server.logger") as mock_logger:
            sid = "test_client"
            processing_time = 0.05  # 50ms processing time
            audio_duration = 0.18  # 180ms audio chunk (block_time)

            # Expected RTF = 0.05 / 0.18 ≈ 0.28
            warn_if_rtf_slow(sid, processing_time, audio_duration)

            # Should not have warning calls since RTF < 1.0
            mock_logger.warning.assert_not_called()

            # Should have debug call
            mock_logger.debug.assert_called_once()
            debug_call = mock_logger.debug.call_args[0]
            assert "RTF for client" in debug_call[0]

    def test_rtf_slow_processing_warning(self, caplog):
        """Test RTF warning when processing is too slow (RTF > 1.0)."""
        with patch("seed_vc.socketio.server.logger") as mock_logger:
            sid = "test_client"
            processing_time = 0.25  # 250ms processing time
            audio_duration = 0.18  # 180ms audio chunk

            # Expected RTF = 0.25 / 0.18 ≈ 1.39
            warn_if_rtf_slow(sid, processing_time, audio_duration)

            # Check that warning was called
            mock_logger.warning.assert_called_once()

            # Get the call arguments (format string and parameters)
            call_args = mock_logger.warning.call_args[0]
            format_string = call_args[0]
            params = call_args[1:]

            # Check format string and parameters
            assert "RTF Warning" in format_string
            assert "Current RTF=%.2f" in format_string
            assert "Processing too slow!" in format_string
            assert params[0] == "test_client"  # sid
            assert abs(params[1] - 1.39) < 0.01  # RTF value

    def test_rtf_zero_audio_duration_handling(self):
        """Test that zero or negative audio duration is handled gracefully."""
        with patch("seed_vc.socketio.server.logger") as mock_logger:
            sid = "test_client"
            processing_time = 0.05

            # Test with zero duration - should not call any logging
            warn_if_rtf_slow(sid, processing_time, 0.0)

            # Test with negative duration - should not call any logging
            warn_if_rtf_slow(sid, processing_time, -0.1)

            # Should not have called warning or debug
            mock_logger.warning.assert_not_called()
            mock_logger.debug.assert_not_called()

    def test_audio_chunk_rtf_integration(self, caplog):
        """Test RTF monitoring integration in audio_chunk function."""
        # Mock global converter
        mock_converter = MagicMock()
        mock_converter.input_sampling_rate = 44100
        mock_converter.output_wav = []

        # Create test audio data (simulate slow processing)
        frames = 7938  # 180ms at 44100 Hz
        test_data = np.random.randint(-32768, 32767, frames, dtype=np.int16).tobytes()

        def slow_audio_callback(*args, **kwargs):
            """Simulate slow audio processing."""
            time.sleep(0.25)  # 250ms processing time > 180ms audio duration
            # Fill outdata with some dummy data
            outdata = kwargs.get("outdata")
            if outdata is not None:
                outdata.fill(0.1)

        mock_converter.audio_callback = slow_audio_callback

        # Register the client so that the chunk is not dropped
        from seed_vc.socketio import server as server_module
        from seed_vc.socketio.runtime import ServerRuntimeCoordinator

        server_module.runtime = ServerRuntimeCoordinator()
        server_module.runtime.try_register_client("test_client", max_clients=1)

        # Mock global state and logger
        with patch("seed_vc.socketio.server.global_converter", mock_converter):
            with patch("seed_vc.socketio.server.sio") as mock_sio:
                with patch("seed_vc.socketio.server.logger") as mock_logger:
                    # Mock async emit
                    async def mock_emit(*args, **kwargs):
                        return None

                    mock_sio.emit = mock_emit

                    # Run async function
                    asyncio.run(audio_chunk("test_client", test_data))

                    # Check that warning was called
                    mock_logger.warning.assert_called()

                    # Find the RTF warning call
                    warning_calls = [
                        call
                        for call in mock_logger.warning.call_args_list
                        if "RTF Warning" in str(call)
                    ]
                    assert len(warning_calls) > 0

                    # Check the warning call format
                    warning_call = warning_calls[0][0]
                    format_string = warning_call[0]
                    assert "RTF Warning" in format_string
                    assert "Current RTF=%.2f" in format_string

    def test_multiple_clients_rtf_independently(self):
        """Test RTF warning for multiple clients independently."""
        with patch("seed_vc.socketio.server.logger") as mock_logger:
            sid1 = "client1"
            sid2 = "client2"

            # Client1: Normal RTF (no warning)
            warn_if_rtf_slow(sid1, 0.05, 0.18)  # RTF ≈ 0.28

            # Client2: Slow RTF (warning)
            warn_if_rtf_slow(sid2, 0.25, 0.18)  # RTF ≈ 1.39

            # Should have one warning call (for client2 only)
            mock_logger.warning.assert_called_once()

            # Should have two debug calls (for both clients)
            assert mock_logger.debug.call_count == 2
