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

"""Real-time audio streaming server using Socket.IO and FastAPI with uvicorn."""

import argparse
import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import socketio
import uvicorn
from fastapi import FastAPI
from socketio.exceptions import ConnectionRefusedError as SocketIOConnectionRefused

from seed_vc.socketio.schemas import (
    ChunkSizeMismatchError,
    ClientAudioConfig,
    ConnectionErrorType,
    MaxClientsReachedError,
)

# Create logger instance
logger = logging.getLogger(__name__)
logger.propagate = False  # Prevent duplicate logs to root logger
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [SERVER] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger.info("🚀 Starting server imports...")
logger.info("⏳ Importing seed_vc modules (this may take a while)...")
from seed_vc.socketio.api import APIRouterVCModel  # noqa: E402
from seed_vc.socketio.model import VoiceConverter  # noqa: E402

logger.info("✅ All imports completed!")


# Global SocketIO server instance
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# FastAPI app instance
fastapi_app = FastAPI(
    title="Voice Conversion API", description="Real-time voice conversion with Socket.IO"
)

# Global VoiceConverter instance (shared across all clients)
global_converter: Optional[VoiceConverter] = None
converter_lock = threading.Lock()

# Store client-specific state
client_converters: Dict[str, Optional[VoiceConverter]] = {}
converter_init_status: Dict[str, bool] = {}  # Track if initialization was attempted

# Maximum number of concurrent clients
MAX_CLIENT = 1


@sio.event
async def connect(
    sid: str, environ: Dict[str, Any], auth: Optional[ClientAudioConfig] = None
) -> bool:
    """Handle client connection.

    Args:
        sid: Client session ID.
        environ: WSGI environment dictionary.
        auth: Authentication data from client.

    Returns:
        bool: True if connection is accepted, False otherwise.
    """
    logger.info("🔗 Client connection attempt: %s", sid)

    # Check if maximum number of clients is already connected
    with converter_lock:
        current_clients = len(client_converters)

    if current_clients >= MAX_CLIENT:
        logger.error(
            "❌ Connection rejected for client %s: "
            "Maximum number of clients (%d) already connected",
            sid,
            MAX_CLIENT,
        )
        # Use ConnectionRefusedError to send custom error message to client
        error_message: MaxClientsReachedError = {
            "error": ConnectionErrorType.MAX_CLIENTS_REACHED.value,
            "message": f"Maximum number of clients ({MAX_CLIENT}) already connected",
        }
        raise SocketIOConnectionRefused(error_message)

    # Validate chunk size and sample rate if provided
    if auth:
        client_chunk_size = auth.get("chunk_size")
        client_sample_rate = auth.get("sample_rate")

        if client_chunk_size is not None and client_sample_rate is not None:
            # Calculate expected chunk size based on converter's block_frame
            # Use the same zc-aligned formula as model._init_buffers()
            if global_converter is not None:
                block_time = global_converter.block_time
                expected_chunk_size = global_converter.block_frame
            else:
                # Fallback: use zc-aligned formula matching model._init_buffers()
                block_time = 0.18
                zc = client_sample_rate // 50
                expected_chunk_size = int(np.round(block_time * client_sample_rate / zc)) * zc

            if client_chunk_size != expected_chunk_size:
                logger.error(
                    "❌ Chunk size mismatch for client %s: "
                    "client=%d, expected=%d (block_time=%.2fs, sample_rate=%d)",
                    sid,
                    client_chunk_size,
                    expected_chunk_size,
                    block_time,
                    client_sample_rate,
                )
                # Use ConnectionRefusedError to send custom error message to client
                error_data: ChunkSizeMismatchError = {
                    "error": ConnectionErrorType.CHUNK_SIZE_MISMATCH.value,
                    "client_chunk_size": client_chunk_size,
                    "expected_chunk_size": expected_chunk_size,
                    "block_time": block_time,
                    "sample_rate": client_sample_rate,
                }
                raise SocketIOConnectionRefused(error_data)

            logger.info(
                "✅ Chunk size validated for client %s: %d frames (%.2fs @ %dHz)",
                sid,
                client_chunk_size,
                block_time,
                client_sample_rate,
            )
    else:
        logger.warning("⚠️ Client %s connected without authentication data", sid)

    # Assign the global converter to this client
    with converter_lock:
        client_converters[sid] = global_converter
        converter_init_status[sid] = global_converter is not None

        # Log current connection count after successful connection
        current_clients = len(client_converters)
        logger.info("🔗 Client connected successfully: %s", sid)
        logger.info("📊 Current connected clients: %d/%d", current_clients, MAX_CLIENT)

    return True


@sio.event
async def disconnect(sid: str) -> None:
    """Handle client disconnection.

    Args:
        sid: Client session ID.
    """
    logger.info("🔌 Client disconnected: %s", sid)

    # Clean up converter
    with converter_lock:
        if sid in client_converters:
            del client_converters[sid]
        if sid in converter_init_status:
            del converter_init_status[sid]

        # Log current connection count after disconnect
        current_clients = len(client_converters)
        logger.info("📊 Current connected clients: %d/%d", current_clients, MAX_CLIENT)


def initialize_global_converter(log_level: str = "INFO") -> Optional[VoiceConverter]:
    """Initialize the global VoiceConverter instance.

    Returns:
        VoiceConverter instance or None if initialization failed.
    """
    global global_converter

    if global_converter is not None:
        return global_converter

    try:
        logger.info("🔄 Initializing global VoiceConverter...")
        global_converter = VoiceConverter(
            input_sampling_rate=44100,  # Match client's sample rate (44.1kHz)
            block_time=0.18,  # 180ms blocks
            crossfade_time=0.04,
            extra_time_ce=2.5,
            extra_time=0.5,
            extra_time_right=0.02,
            diffusion_steps=10,
            max_prompt_length=3.0,
            inference_cfg_rate=0.7,
            use_vad=True,
            log_level=log_level,
        )
        logger.info("✅ Global VoiceConverter ready!")
        return global_converter

    except Exception as e:
        logger.error("❌ Failed to initialize global VoiceConverter: %s", e)
        import traceback

        traceback.print_exc()
        return None


def warn_if_rtf_slow(sid: str, processing_time: float, audio_duration: float) -> None:
    """Warn if RTF > 1.0 (processing too slow).

    Args:
        sid: Client session ID.
        processing_time: Time taken to process the audio chunk (seconds).
        audio_duration: Duration of the audio chunk (seconds).
    """
    if audio_duration <= 0:
        return

    rtf = processing_time / audio_duration

    # Warn if current RTF > 1.0
    if rtf > 1.0:
        logger.warning(
            "⚠️ RTF Warning for client %s: Current RTF=%.2f (>1.0) - Processing too slow!",
            sid,
            rtf,
        )

    # Debug log for RTF tracking
    logger.debug("📊 RTF for client %s: %.3f", sid, rtf)


def initialize_converter_for_client(sid: str) -> Optional[VoiceConverter]:
    """Get VoiceConverter for a client (returns global converter).

    Args:
        sid: Client session ID.

    Returns:
        VoiceConverter instance or None if initialization failed.
    """
    with converter_lock:
        # Return the global converter
        return client_converters.get(sid, global_converter)


@sio.on("audio_chunk")
async def audio_chunk(sid: str, data: bytes) -> None:
    """Process audio chunk from client with voice conversion.

    Args:
        sid: Client session ID.
        data: Audio chunk data as bytes.
    """
    # Get global converter
    converter = global_converter

    if not converter:
        logger.warning("⚠️ No converter available for %s, echoing original audio", sid)
        await sio.emit("audio_chunk", data, to=sid)
        return

    # Start timing for RTF calculation
    start_time = time.time()

    try:
        # Convert bytes to numpy array (int16 -> float32, normalize to [-1, 1])
        audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Prepare for audio_callback (add channel dimension for sounddevice format)
        audio_2d = audio_array.reshape(-1, 1)  # Shape: (frames, channels)
        frames = len(audio_array)
        outdata = np.zeros_like(audio_2d)  # Must match input shape

        # Clear previous output
        converter.output_wav = []

        converter.audio_callback(
            indata=audio_2d, outdata=outdata, frames=frames, time=None, status=None
        )

        # Get converted audio from outdata
        converted_audio = outdata[:, 0]  # Remove channel dimension

        # Convert back to int16
        converted_int16 = (converted_audio * 32768.0).astype(np.int16)
        processed_data = converted_int16.tobytes()

    except Exception as e:
        logger.error("❌ Error processing audio for %s: %s", sid, e)
        import traceback

        traceback.print_exc()
        # Fallback to echo
        processed_data = data

    # Calculate processing time and RTF
    processing_time = time.time() - start_time

    # Calculate audio duration (frames / sample_rate)
    # Use converter's input sampling rate (should be 44100 Hz)
    sample_rate = converter.input_sampling_rate if converter else 44100
    audio_duration = len(audio_array) / sample_rate

    # Warn if RTF is too slow
    warn_if_rtf_slow(sid, processing_time, audio_duration)

    await sio.emit("audio_chunk", processed_data, to=sid)


def main() -> None:
    """Start the Socket.IO server."""
    parser = argparse.ArgumentParser(description="Voice Conversion Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port number",
    )
    parser.add_argument(
        "--allowed-audio-dirs",
        type=str,
        nargs="*",
        default=["assets/examples/reference"],
        help="List of directories where audio files are allowed to be loaded from",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Configure logger based on the specified level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logger.setLevel(numeric_level)

    logger.info("🎙️ Starting voice conversion server on %s:%s ...", args.host, args.port)

    # Initialize the global converter before starting the server
    if initialize_global_converter(log_level=args.log_level) is None:
        logger.error("❌ Failed to initialize VoiceConverter. Exiting...")
        return

    # Create client count checker function that captures the actual client_converters
    def get_client_count() -> int:
        with converter_lock:
            return len(client_converters)

    # API router for VoiceConverter model
    api_router = APIRouterVCModel(
        model=global_converter,
        log_level=args.log_level,
        allowed_audio_dirs=args.allowed_audio_dirs,
        client_count_checker=get_client_count,
    )

    # Add the VoiceConverter API routes to FastAPI
    fastapi_app.include_router(api_router.api_router, prefix="/api/v1")

    # Create ASGI app combining FastAPI and Socket.IO
    app = socketio.ASGIApp(
        socketio_server=sio,
        other_asgi_app=fastapi_app,
    )

    logger.info("🌟 Ready to accept connections!")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=None,  # Use our logging configuration instead of uvicorn's default
    )


if __name__ == "__main__":
    main()
