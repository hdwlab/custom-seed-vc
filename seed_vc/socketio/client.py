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

"""Client for real-time audio streaming using Socket.IO and sounddevice."""

import argparse
import logging
import queue
import threading
from typing import Any, Dict, Optional, Union

import numpy as np
import socketio
import sounddevice as sd

from seed_vc.socketio.schemas import (
    ClientAudioConfig,
    ConnectionErrorType,
    RealtimeConversionError,
)
from seed_vc.socketio.schemas import (
    ConnectionError as SocketIOConnectionError,
)

logger = logging.getLogger(__name__)
logger.propagate = False  # Prevent duplicate logs to root logger
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [CLIENT] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Audio configuration
SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = "int16"
CHUNK_SIZE = 7938  # Number of frames
MAX_QUEUE_SIZE = 1  # Maximum queued chunks before draining stale audio
AudioDevice = Union[int, str]

# Global queues and client
play_q: queue.Queue[bytes] = queue.Queue()  # Server -> Speaker
send_q: queue.Queue[bytes] = queue.Queue()  # Microphone -> Server
sio = socketio.Client(logger=False, engineio_logger=False, request_timeout=60)

# Store connection error details
connection_error_details: Dict[str, Union[str, int, float]] = {}


@sio.event
def connect() -> None:
    """Handle connection to server."""
    logger.info("🔗 Connected to server")


@sio.event
def disconnect() -> None:
    """Handle disconnection from server."""
    logger.info("🔌 Disconnected from server")


@sio.on("audio_chunk")
def on_audio_chunk(data: bytes) -> None:
    """Handle audio chunk received from server.

    Args:
        data: Audio chunk data as bytes.
    """
    logger.debug("📦 Received audio packet (size: %d bytes)", len(data))
    play_q.put(data)


@sio.on("connection_error")
def on_connection_error(data: SocketIOConnectionError) -> None:
    """Handle connection error details from server.

    Args:
        data: Error details from server.
    """
    global connection_error_details
    connection_error_details = data


@sio.on("conversion_error")
def on_conversion_error(data: RealtimeConversionError) -> None:
    """Report a realtime conversion failure from the server.

    The matching audio chunk contains silence so the original microphone
    signal is never played back as a fallback.

    Args:
        data: Structured conversion error details from the server.
    """
    logger.error(
        "Realtime conversion failed: %s (action=%s)",
        data.get("message", "unknown error"),
        data.get("action", "unknown"),
    )


@sio.on("connect_error")
def on_connect_error(data: SocketIOConnectionError) -> None:
    """Handle connection refusal error from server.

    Args:
        data: Error details from server when connection is refused.
    """
    global connection_error_details
    connection_error_details = data


def audio_callback(
    indata: np.ndarray,
    frames: int,
    time: Any,
    status: sd.CallbackFlags,
) -> None:
    """Callback for microphone input.

    Args:
        indata: Input audio data.
        frames: Number of frames.
        time: Timing information.
        status: Status flags.
    """
    if status:
        logger.warning("⚠️ Input status: %s", status)
    # Copy buffer data to avoid reuse issues
    send_q.put(bytes(indata))


def send_loop() -> None:
    """Background thread to send microphone data to server."""
    while True:
        data = send_q.get()
        if send_q.qsize() > MAX_QUEUE_SIZE:
            skipped = 0
            while not send_q.empty():
                data = send_q.get()
                skipped += 1
            logger.warning("⚠️ Skipped %d stale send chunks to reduce latency", skipped)
        if sio.connected:
            try:
                logger.debug("📤 Sending audio packet (size: %d bytes)", len(data))
                sio.emit("audio_chunk", data)
            except Exception as e:
                logger.error("❌ Send error: %s", e)


def parse_audio_device(value: str) -> AudioDevice:
    """Parse a sounddevice index or a device name from a CLI value."""
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("audio device must not be empty")
    try:
        return int(value)
    except ValueError:
        return value


def validate_audio_devices(
    input_device: Optional[AudioDevice],
    output_device: Optional[AudioDevice],
    sample_rate: int,
) -> None:
    """Validate explicitly selected audio devices before connecting.

    Args:
        input_device: Input device index or name, or None for the OS default.
        output_device: Output device index or name, or None for the OS default.
        sample_rate: Server-compatible stream sampling rate.
    """
    if input_device is not None:
        sd.check_input_settings(
            device=input_device,
            channels=CHANNELS,
            dtype=DTYPE,
            samplerate=sample_rate,
        )
    if output_device is not None:
        sd.check_output_settings(
            device=output_device,
            channels=CHANNELS,
            dtype=DTYPE,
            samplerate=sample_rate,
        )


def main() -> None:
    """Start the audio streaming client."""
    parser = argparse.ArgumentParser(description="Real-time audio streaming client")
    parser.add_argument(
        "--host",
        default="localhost",
        help="Server hostname or IP address",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=(
            "Size of audio chunks to send (default: 7938 frames) "
            "This must be the same as the server's chunk size."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Server port number",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit",
    )
    parser.add_argument(
        "--input-device",
        type=parse_audio_device,
        default=None,
        help="Input device index or name (default: OS default)",
    )
    parser.add_argument(
        "--output-device",
        type=parse_audio_device,
        default=None,
        help="Output device index or name (default: OS default)",
    )
    parser.add_argument(
        "--operator-id",
        default=None,
        help="Operator ID used for deterministic voice preset assignment",
    )
    parser.add_argument(
        "--gender",
        default=None,
        choices=["male", "female"],
        help="Optional preset gender filter",
    )
    parser.add_argument(
        "--preset-id",
        default=None,
        help="Explicit voice preset ID to use",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return

    # Update log level if specified via command line
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logger.setLevel(numeric_level)

    url = f"http://{args.host}:{args.port}"
    logger.info("🔗 Connecting to %s", url)

    try:
        validate_audio_devices(args.input_device, args.output_device, SAMPLE_RATE)
    except Exception as e:
        logger.error("Invalid audio device configuration: %s", e)
        return

    # Send chunk size and sample rate for validation
    auth_data: ClientAudioConfig = {"chunk_size": args.chunk_size, "sample_rate": SAMPLE_RATE}
    if args.operator_id is not None:
        auth_data["operator_id"] = args.operator_id
    if args.gender is not None:
        auth_data["gender"] = args.gender
    if args.preset_id is not None:
        auth_data["preset_id"] = args.preset_id

    try:
        sio.connect(url, auth=auth_data)
    except socketio.exceptions.ConnectionError as e:
        logger.error("❌ Failed to connect to server: %s", e)

        # Check if we received specific error details from server
        error_type = connection_error_details.get("error")
        if error_type == ConnectionErrorType.CHUNK_SIZE_MISMATCH.value:
            logger.error(
                "❌ Chunk size mismatch: client=%d, expected=%d (block_time=%.2fs, sample_rate=%d)",
                connection_error_details["client_chunk_size"],
                connection_error_details["expected_chunk_size"],
                connection_error_details["block_time"],
                connection_error_details["sample_rate"],
            )
            logger.error(
                "To fix this, use --chunk-size %d or adjust the server's block_time",
                connection_error_details["expected_chunk_size"],
            )
        elif error_type == ConnectionErrorType.MAX_CLIENTS_REACHED.value:
            logger.error(
                "❌ Connection rejected: %s",
                connection_error_details.get(
                    "message", "Maximum number of clients already connected"
                ),
            )
            logger.error("Please wait for another client to disconnect, or try connecting later.")
        elif error_type == ConnectionErrorType.INVALID_OPERATOR_INFO.value:
            logger.error(
                "❌ Invalid operator voice request: %s",
                connection_error_details.get("message", "invalid operator information"),
            )
        else:
            # No specific error details from server
            if not connection_error_details:
                logger.error(
                    "No response from server. Possible causes:\n"
                    "  1. Server is not running at %s\n"
                    "  2. Network connectivity issues\n"
                    "  3. Firewall blocking the connection\n"
                    "  4. Server crashed during handshake",
                    url,
                )
                logger.info(
                    "Try: python -m seed_vc.socketio.server --host %s --port %d",
                    args.host,
                    args.port,
                )
            else:
                # Other server errors
                logger.error("Server error: %s", connection_error_details)
        return

    # Start sending thread
    threading.Thread(target=send_loop, daemon=True).start()

    # Input stream (microphone)
    with sd.RawInputStream(
        device=args.input_device,
        samplerate=SAMPLE_RATE,
        blocksize=args.chunk_size,
        dtype=DTYPE,
        channels=CHANNELS,
        callback=audio_callback,
    ):
        # Output stream (speaker)
        with sd.RawOutputStream(
            device=args.output_device,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            dtype=DTYPE,
            channels=CHANNELS,
        ) as outstream:
            logger.info("🎧 Streaming... (Ctrl+C to stop)")
            try:
                while True:
                    chunk = play_q.get()
                    if play_q.qsize() > MAX_QUEUE_SIZE:
                        skipped = 0
                        while not play_q.empty():
                            chunk = play_q.get()
                            skipped += 1
                        logger.warning(
                            "⚠️ Skipped %d stale playback chunks to reduce latency",
                            skipped,
                        )
                    outstream.write(chunk)
            except KeyboardInterrupt:
                logger.info("⏹️ Stopped by user")
    sio.disconnect()


if __name__ == "__main__":
    main()
