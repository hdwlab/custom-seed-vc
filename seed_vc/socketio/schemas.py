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

"""Type definitions for Socket.IO communication and API requests."""

from enum import Enum
from typing import Literal, Optional, TypedDict, Union

from pydantic import BaseModel, Field


class ConnectionErrorType(Enum):
    """Enumeration of connection error types."""

    CHUNK_SIZE_MISMATCH = "chunk_size_mismatch"
    MAX_CLIENTS_REACHED = "max_clients_reached"
    OFFLINE_BUSY = "offline_busy"


OFFLINE_BUSY_MESSAGE = "Offline conversion is already in progress. Please try again later."


class ChunkSizeMismatchError(TypedDict):
    """Error details for chunk size mismatch."""

    error: str  # Should be ConnectionErrorType.CHUNK_SIZE_MISMATCH.value
    client_chunk_size: int
    expected_chunk_size: int
    block_time: float
    sample_rate: int


class MaxClientsReachedError(TypedDict):
    """Error details for maximum clients reached."""

    error: str  # Should be ConnectionErrorType.MAX_CLIENTS_REACHED.value
    message: str


class OfflineBusyError(TypedDict):
    """Error details for an offline conversion job in progress."""

    error: str  # Should be ConnectionErrorType.OFFLINE_BUSY.value
    message: str


class ClientAudioConfig(TypedDict):
    """Audio configuration sent from client to server during connection."""

    chunk_size: int
    sample_rate: int


# Union type for all connection error types
ConnectionError = Union[ChunkSizeMismatchError, MaxClientsReachedError, OfflineBusyError]


# API Request Models


class ReferenceAudioRequest(BaseModel):
    """Request model for updating reference audio."""

    file_path: str = Field(
        ...,
        description="Path to the reference audio file. Must be within allowed directories.",
        example="assets/examples/reference/sample.wav",
    )


class FileConversionRequest(BaseModel):
    """Request model for file-path-based voice conversion."""

    input_path: str = Field(
        ...,
        description="Path to the input audio file. Must be within allowed directories.",
        example="assets/examples/reference/sample.wav",
    )
    output_path: str = Field(
        ...,
        description="Path to save the converted audio file. Must be within allowed directories.",
        example="assets/examples/reference/converted.wav",
    )


class ConversionModeRequest(BaseModel):
    """Request model for changing conversion mode."""

    mode: Literal["convert", "passthrough", "silence"] = Field(
        ...,
        description="Voice conversion mode: 'convert' for voice conversion, "
        "'passthrough' for original audio, 'silence' for muted output",
        example="convert",
    )


class ModelParametersRequest(BaseModel):
    """Request model for updating model parameters.

    All fields are optional - only specified fields will be updated.
    """

    # Audio processing parameters
    block_time: Optional[float] = Field(
        None,
        description="the time length of each audio chunk for inference, the higher the value, "
        "the higher the latency, note this value must be greater than the inference time per block,"
        "set according to your hardware condition (in seconds)",
        example=0.18,
        gt=0,
    )
    crossfade_time: Optional[float] = Field(
        None,
        description="the time length of crossfade between audio chunks,"
        "normally not needed to change (in seconds)",
        example=0.04,
        ge=0,
    )
    extra_time_ce: Optional[float] = Field(
        None,
        description="the time length of extra history context for inference, the higher the value,"
        "the higher the inference time, but can increase stability (in seconds)",
        example=2.5,
        ge=0,
    )
    extra_time: Optional[float] = Field(
        None, description="Extra context time for DiT (in seconds)", example=0.5, ge=0
    )
    extra_time_right: Optional[float] = Field(
        None,
        description="the time length of extra future context for inference, the higher the value"
        "the higher the inference time and latency, but can increase stability (in seconds)",
        example=0.02,
        ge=0,
    )

    # Inference parameters
    diffusion_steps: Optional[int] = Field(
        None,
        description="the number of diffusion steps to use "
        "in real-time case usually set to 4~10 for fastest inference",
        example=10,
        gt=0,
    )
    max_prompt_length: Optional[float] = Field(
        None,
        description="Maximum duration of reference audio prompt (in seconds)",
        example=3.0,
        gt=0,
    )
    inference_cfg_rate: Optional[float] = Field(
        None,
        description="Classifier-free guidance rate for inference (0-1)",
        example=0.7,
        ge=0,
        le=1,
    )

    # VAD parameter
    use_vad: Optional[bool] = Field(
        None,
        description="Enable Voice Activity Detection for processing optimization",
        example=True,
    )


class ModelReloadRequest(BaseModel):
    """Request model for reloading the model with new checkpoint and config."""

    checkpoint_path: Optional[str] = Field(
        None,
        description="Path to the model checkpoint file (.pth). "
        "If None or empty, will load default model from HuggingFace",
        example="path/to/DiT_checkpoint.pth",
    )
    config_path: Optional[str] = Field(
        None,
        description="Path to the model configuration file (.yml). "
        "Required if checkpoint_path is provided",
        example="path/to/config.yml",
    )
