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

"""API for VoiceConverter model with FastAPI integration."""

import logging
import tempfile
import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable, Iterator, NoReturn, Optional

import librosa
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from seed_vc.socketio.model import ConversionMode, VoiceConverter
from seed_vc.socketio.runtime import ServerRuntimeCoordinator
from seed_vc.socketio.schemas import (
    OFFLINE_BUSY_MESSAGE,
    ConversionModeRequest,
    FileConversionRequest,
    ModelParametersRequest,
    ModelReloadRequest,
    ReferenceAudioRequest,
)

logger = logging.getLogger(__name__)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [API] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


def _get_connected_clients_count() -> int:
    """Get the number of connected clients from server module.

    This function is kept for backward compatibility but is not actively used
    in the current implementation. The preferred method is to use the
    client_count_checker provided to APIRouterVCModel constructor.

    Returns:
        Number of currently connected clients.
    """
    try:
        # Import here to avoid circular imports
        from seed_vc.socketio.server import client_converters, converter_lock

        with converter_lock:
            return len(client_converters)
    except ImportError as e:
        logger.warning("⚠️ API: Could not import server module: %s", e)
        # If server module is not available, assume no clients
        return 0


def _connected_clients_http_exception(connected_count: int) -> HTTPException:
    """Return the standard 409 response for active realtime clients.

    Args:
        connected_count: Number of currently connected clients.
    """
    return HTTPException(
        status_code=409,  # Conflict
        detail=(
            f"Cannot perform this operation while {connected_count} "
            "client(s) are connected. Please disconnect all clients first "
            "and try again."
        ),
    )


def require_no_connected_clients_method(func: Callable) -> Callable:
    """Decorator to ensure no clients are connected before executing API endpoint method.

    This decorator is designed for APIRouterVCModel methods that have access to
    self.client_count_checker.

    Args:
        func: The API endpoint method to wrap.

    Returns:
        Wrapped method that checks for connected clients.

    Raises:
        HTTPException: If clients are connected.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.client_count_checker is not None:
            connected_count = self.client_count_checker()
            logger.info(
                "🔍 API: Checking connected clients before %s: %d clients",
                func.__name__,
                connected_count,
            )
            if connected_count > 0:
                logger.error(
                    "❌ API: Blocking %s - %d client(s) connected", func.__name__, connected_count
                )
                raise _connected_clients_http_exception(connected_count)
            logger.info("✅ API: Allowing %s - no clients connected", func.__name__)
        else:
            logger.warning("⚠️ API: No client count checker provided - allowing %s", func.__name__)

        return func(self, *args, **kwargs)

    return wrapper


class APIRouterVCModel:
    """API router for VoiceConverter model."""

    def __init__(
        self,
        model: VoiceConverter,
        allowed_audio_dirs: Optional[list[str]] = None,
        log_level: str = "INFO",
        client_count_checker: Optional[Callable[[], int]] = None,
        runtime: Optional[ServerRuntimeCoordinator] = None,
    ) -> None:
        """Initialize the API router with the given VoiceConverter model.

        Args:
            model (VoiceConverter): The VoiceConverter model instance.
            allowed_audio_dirs: List of allowed directories for audio files.
                If None, defaults to ["assets/examples/reference"]
            log_level (str): Logging level for the API. Defaults to "INFO".
            client_count_checker: Function to get current client count. If None,
                falls back to the runtime's client count when a runtime is given,
                otherwise no client connection checking will be performed.
            runtime: Runtime state shared with the Socket.IO event handlers.
                If None, a private coordinator is created.
        """
        # Configure logger level for this instance
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        logger.setLevel(numeric_level)
        self.logger = logger

        self.api_router = APIRouter()
        self._init_routes()
        self.model = model
        self.runtime = runtime or ServerRuntimeCoordinator()
        if client_count_checker is None and runtime is not None:
            client_count_checker = runtime.client_count
        self.client_count_checker = client_count_checker

        # Set allowed directories for audio files
        if allowed_audio_dirs is None:
            self.allowed_audio_dirs = ["assets/examples/reference"]
        else:
            self.allowed_audio_dirs = allowed_audio_dirs

    @property
    def model_lock(self) -> threading.Lock:
        """Lock guarding model access, shared with the Socket.IO handlers."""
        return self.runtime.model_lock

    def _init_routes(self) -> None:
        """Set up API routes. Can be overridden in subclasses."""
        # Define your API routes here
        self.api_router.add_api_route(
            "/config",
            self.get_config,
            methods=["GET"],
        )
        self.api_router.add_api_route(
            "/reference",
            self.update_reference_audio,
            methods=["POST"],
        )
        self.api_router.add_api_route(
            "/mode",
            self.change_conversion_mode,
            methods=["POST"],
        )
        self.api_router.add_api_route(
            "/parameters",
            self.update_model_parameters,
            methods=["POST"],
        )
        self.api_router.add_api_route(
            "/reload",
            self.reload_model,
            methods=["POST"],
        )
        self.api_router.add_api_route(
            "/convert",
            self.convert_file,
            methods=["POST"],
        )
        self.api_router.add_api_route(
            "/convert/upload",
            self.convert_file_upload,
            methods=["POST"],
        )

    def _is_in_allowed_dirs(self, resolved_path: Path) -> bool:
        """Check whether the resolved path is within allowed directories.

        Args:
            resolved_path: The resolved absolute path to check.

        Returns:
            True if the path is within one of the allowed directories.
        """
        for allowed_dir in self.allowed_audio_dirs:
            allowed_path = Path(allowed_dir).resolve()
            try:
                # Check if the file is within the allowed directory
                resolved_path.relative_to(allowed_path)
                return True
            except ValueError:
                # Not within this allowed directory, continue checking
                continue
        return False

    def _validate_file_path(self, file_path: str) -> str:
        """Validate and resolve file path to prevent directory traversal attacks.

        Args:
            file_path: The file path to validate.

        Returns:
            The resolved absolute path.

        Raises:
            HTTPException: If the path is invalid or not in allowed directories.
        """
        try:
            # Resolve the path to handle any relative components
            resolved_path = Path(file_path).resolve()

            # Check if the file exists
            if not resolved_path.exists():
                raise HTTPException(status_code=404, detail=f"Audio file not found: {file_path}")

            if not self._is_in_allowed_dirs(resolved_path):
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Access denied: File must be in one of the allowed directories: "
                        f"{self.allowed_audio_dirs}"
                    ),
                )

            return str(resolved_path)

        except (OSError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid file path: {str(e)}") from e

    def _validate_output_path(self, file_path: str) -> str:
        """Validate and resolve output file path for writing.

        Args:
            file_path: The output file path to validate.

        Returns:
            The resolved absolute path.

        Raises:
            HTTPException: If the parent directory does not exist or the path is not allowed.
        """
        try:
            resolved_path = Path(file_path).resolve()

            if not resolved_path.parent.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Output directory does not exist: {resolved_path.parent}",
                )

            if not self._is_in_allowed_dirs(resolved_path):
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Access denied: File must be in one of the allowed directories: "
                        f"{self.allowed_audio_dirs}"
                    ),
                )

            return str(resolved_path)

        except (OSError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid output path: {str(e)}") from e

    def get_config(self) -> JSONResponse:
        """Return server audio configuration for client auto-setup.

        Returns:
            JSONResponse with sampling rate, block time, and chunk size.
            The chunk size is the zc-aligned block_frame used in connection validation.
        """
        return JSONResponse(
            content={
                "input_sampling_rate": self.model.input_sampling_rate,
                "block_time": self.model.block_time,
                "chunk_size": self.model.block_frame,
            },
            status_code=200,
        )

    def update_reference_audio(self, request: ReferenceAudioRequest) -> JSONResponse:
        """Update the reference audio for voice conversion.

        Args:
            request: Request containing the file path to the new reference audio.

        Returns:
            JSONResponse containing the updated reference audio information.

        Raises:
            HTTPException: If the file does not exist, is not allowed, or cannot be loaded.
        """
        self.logger.info("🎵 API: Update reference audio - %s", request.file_path)

        # Validate and resolve file path to prevent directory traversal
        validated_file_path = self._validate_file_path(request.file_path)

        try:
            # Load outside the lock so realtime audio processing is not stalled
            reference_wav, _ = librosa.load(
                validated_file_path,
                sr=self.model.model_set[-1]["sampling_rate"],
            )

            # Thread-safe update of model's reference audio and cache
            with self.model_lock:
                self._apply_reference_audio(validated_file_path, reference_wav)

            duration = len(reference_wav) / self.model.model_set[-1]["sampling_rate"]
            self.logger.info("✅ API: Reference audio updated successfully (%.2fs)", duration)

            return JSONResponse(
                content={
                    "message": "Reference audio updated successfully",
                    "reference_path": validated_file_path,
                    "sampling_rate": self.model.model_set[-1]["sampling_rate"],
                    "audio_duration": duration,
                },
                status_code=200,
            )

        except Exception as e:
            self.logger.error("❌ API: Failed to load audio file: %s", str(e))
            raise HTTPException(
                status_code=500, detail=f"Failed to load audio file: {str(e)}"
            ) from e

    def _clear_reference_cache(self) -> None:
        """Clear the model's cached reference values to force regeneration.

        Must be called while holding self.model_lock.
        """
        self.model.prompt_condition = None
        self.model.mel2 = None
        self.model.style2 = None
        self.model.reference_wav_name = ""

    def _apply_reference_audio(self, file_path, reference_wav) -> None:
        """Update the model's reference state and clear dependent caches.

        Must be called while holding self.model_lock.

        Args:
            file_path: Path to the reference audio file.
            reference_wav: Reference waveform loaded at the model's sampling rate.
        """
        self.model.reference_wav_path = file_path
        self.model.reference_wav = reference_wav
        self._clear_reference_cache()

    def _set_reference_audio(self, file_path: str) -> None:
        """Load reference audio and update the model's reference state and caches.

        Must be called while holding self.model_lock.

        Args:
            file_path: Path to the reference audio file.
        """
        reference_wav, _ = librosa.load(
            file_path,
            sr=self.model.model_set[-1]["sampling_rate"],
        )
        self._apply_reference_audio(file_path, reference_wav)

    @staticmethod
    def _raise_conversion_error(error: Exception) -> NoReturn:
        """Translate known conversion errors into appropriate HTTP responses.

        Args:
            error: The exception raised during conversion.

        Raises:
            HTTPException: With a status code matching the error type.
        """
        detail = str(error)
        if isinstance(error, ValueError):
            raise HTTPException(status_code=400, detail=detail) from error
        if isinstance(error, RuntimeError) and "Reference audio not set" in detail:
            raise HTTPException(status_code=400, detail=detail) from error
        raise HTTPException(status_code=500, detail=f"File conversion failed: {detail}") from error

    def _raise_offline_exclusion_error(self) -> NoReturn:
        """Raise the appropriate 409 error for offline conversion conflicts.

        Raises:
            HTTPException: With a message matching the conflicting state.
        """
        connected_count = (
            self.client_count_checker() if self.client_count_checker is not None else 0
        )
        if connected_count > 0:
            raise _connected_clients_http_exception(connected_count)
        raise HTTPException(status_code=409, detail=OFFLINE_BUSY_MESSAGE)

    @contextmanager
    def _offline_model_session(self) -> Iterator[None]:
        """Reserve exclusive offline access and lock the model for a conversion.

        While the reservation is held, new realtime client connections and
        other offline conversion jobs are rejected.

        Raises:
            HTTPException: If realtime clients are connected or another
                offline job is in progress.
        """
        # Reject first based on the external client count checker, which is
        # the only exclusion source when the router is used without a shared
        # runtime (the runtime check below covers the attached case atomically)
        connected_count = (
            self.client_count_checker() if self.client_count_checker is not None else 0
        )
        if connected_count > 0:
            raise _connected_clients_http_exception(connected_count)
        if not self.runtime.try_begin_offline_job():
            self._raise_offline_exclusion_error()
        try:
            with self.model_lock:
                yield
        finally:
            self.runtime.finish_offline_job()

    def convert_file(self, request: FileConversionRequest) -> JSONResponse:
        """Convert an audio file offline using the voice conversion model.

        Args:
            request: Request containing input and output file paths.

        Returns:
            JSONResponse with conversion metadata.

        Raises:
            HTTPException: If validation fails or the conversion fails.
        """
        self.logger.info("🎵 API: Convert file - %s", request.input_path)
        validated_input = self._validate_file_path(request.input_path)
        validated_output = self._validate_output_path(request.output_path)

        try:
            with self._offline_model_session():
                result = self.model.convert_file(validated_input, validated_output)
            self.logger.info("✅ API: File conversion completed - %s", validated_output)
            return JSONResponse(content=result, status_code=200)
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error("❌ API: File conversion failed: %s", str(e))
            self._raise_conversion_error(e)

    def convert_file_upload(
        self,
        input_file: UploadFile,
        reference_file: Optional[UploadFile] = None,
    ) -> Response:
        """Convert an uploaded audio file and return the converted WAV binary.

        Args:
            input_file: Uploaded input audio file.
            reference_file: Optional uploaded reference audio file. If provided,
                it is used only for this conversion and the previous reference
                is restored afterwards.

        Returns:
            Response with converted WAV audio binary.

        Raises:
            HTTPException: If the upload is too large or the conversion fails.
        """
        self.logger.info("🎵 API: Convert uploaded file - %s", input_file.filename)
        tmp_paths: list[str] = []
        try:
            input_tmp_path = self._save_upload_file(input_file, "input", tmp_paths)
            ref_tmp_path = (
                self._save_upload_file(reference_file, "reference", tmp_paths)
                if reference_file is not None
                else None
            )
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_tmp_path = f.name
            tmp_paths.append(output_tmp_path)

            with self._offline_model_session():
                if ref_tmp_path is None:
                    self.model.convert_file(input_tmp_path, output_tmp_path)
                else:
                    previous_path = self.model.reference_wav_path
                    previous_wav = self.model.reference_wav
                    self._set_reference_audio(ref_tmp_path)
                    try:
                        self.model.convert_file(input_tmp_path, output_tmp_path)
                    finally:
                        # Restore the previous reference audio and clear caches
                        self._apply_reference_audio(previous_path, previous_wav)

            wav_bytes = Path(output_tmp_path).read_bytes()
            self.logger.info("✅ API: Upload conversion completed (%d bytes)", len(wav_bytes))
            return Response(content=wav_bytes, media_type="audio/wav")
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error("❌ API: Upload conversion failed: %s", str(e))
            self._raise_conversion_error(e)
        finally:
            for path in tmp_paths:
                Path(path).unlink(missing_ok=True)

    def _save_upload_file(self, upload: UploadFile, label: str, tmp_paths: list[str]) -> str:
        """Read an uploaded file, validate its size, and write it to a temp file.

        Args:
            upload: The uploaded file.
            label: Label used in error messages (e.g. "input", "reference").
            tmp_paths: List to append the temp file path for cleanup.

        Returns:
            Path to the written temp file.

        Raises:
            HTTPException: If the uploaded file is too large.
        """
        # Stream to a temp file in chunks so oversized uploads are rejected
        # without buffering the whole payload in memory
        size = 0
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            tmp_paths.append(tmp_path)
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"{label.capitalize()} file too large "
                            f"(max {MAX_UPLOAD_SIZE_BYTES // 1024 // 1024} MB)"
                        ),
                    )
                f.write(chunk)
        return tmp_path

    def change_conversion_mode(self, request: ConversionModeRequest) -> JSONResponse:
        """Change the conversion mode.

        Args:
            request: Request containing the new conversion mode.

        Returns:
            JSONResponse containing the updated mode information.

        Raises:
            HTTPException: If the mode is invalid.
        """
        self.logger.info("🔄 API: Change conversion mode - %s", request.mode)
        mode_str = request.mode.lower()

        # Validate mode
        valid_modes = {mode.value for mode in ConversionMode}
        if mode_str not in valid_modes:
            self.logger.warning("⚠️ API: Invalid mode requested: %s", mode_str)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {mode_str}. Valid modes are: {', '.join(valid_modes)}",
            )

        # Update conversion mode
        for mode in ConversionMode:
            if mode.value == mode_str:
                self.model.conversion_mode = mode
                break

        self.logger.info("✅ API: Conversion mode updated to %s", self.model.conversion_mode.value)

        return JSONResponse(
            content={
                "message": "Conversion mode updated successfully",
                "mode": self.model.conversion_mode.value,
                "available_modes": list(valid_modes),
            },
            status_code=200,
        )

    @require_no_connected_clients_method
    def update_model_parameters(self, request: ModelParametersRequest) -> JSONResponse:
        """Update model parameters dynamically.

        Only the parameters specified in the request will be updated.
        All others will remain unchanged.

        Args:
            request: Request containing the parameters to update.

        Returns:
            JSONResponse containing the updated parameters.

        Raises:
            HTTPException: If parameter validation fails.
        """
        self.logger.info("⚙️ API: Update model parameters")
        updated_params = {}

        # Update audio processing parameters
        if request.block_time is not None:
            if request.block_time <= 0:
                raise HTTPException(status_code=400, detail="block_time must be positive")
            self.model.block_time = request.block_time
            updated_params["block_time"] = request.block_time

        if request.crossfade_time is not None:
            if request.crossfade_time < 0:
                raise HTTPException(status_code=400, detail="crossfade_time must be non-negative")
            self.model.crossfade_time = request.crossfade_time
            updated_params["crossfade_time"] = request.crossfade_time

        if request.extra_time_ce is not None:
            if request.extra_time_ce < 0:
                raise HTTPException(status_code=400, detail="extra_time_ce must be non-negative")
            self.model.extra_time_ce = request.extra_time_ce
            updated_params["extra_time_ce"] = request.extra_time_ce

        if request.extra_time is not None:
            if request.extra_time < 0:
                raise HTTPException(status_code=400, detail="extra_time must be non-negative")
            self.model.extra_time = request.extra_time
            updated_params["extra_time"] = request.extra_time

        if request.extra_time_right is not None:
            if request.extra_time_right < 0:
                raise HTTPException(status_code=400, detail="extra_time_right must be non-negative")
            self.model.extra_time_right = request.extra_time_right
            updated_params["extra_time_right"] = request.extra_time_right

        # Update inference parameters
        if request.diffusion_steps is not None:
            if request.diffusion_steps <= 0:
                raise HTTPException(status_code=400, detail="diffusion_steps must be positive")
            self.model.diffusion_steps = request.diffusion_steps
            updated_params["diffusion_steps"] = request.diffusion_steps

        if request.max_prompt_length is not None:
            if request.max_prompt_length <= 0:
                raise HTTPException(status_code=400, detail="max_prompt_length must be positive")
            self.model.max_prompt_length = request.max_prompt_length
            updated_params["max_prompt_length"] = request.max_prompt_length

        if request.inference_cfg_rate is not None:
            if not 0 <= request.inference_cfg_rate <= 1:
                raise HTTPException(
                    status_code=400, detail="inference_cfg_rate must be between 0 and 1"
                )
            self.model.inference_cfg_rate = request.inference_cfg_rate
            updated_params["inference_cfg_rate"] = request.inference_cfg_rate

        # Update VAD parameter
        if request.use_vad is not None:
            self.model.use_vad = request.use_vad
            updated_params["use_vad"] = request.use_vad

        # Reinitialize buffers if timing parameters changed
        buffer_reinit_needed = any(
            param in updated_params
            for param in [
                "block_time",
                "crossfade_time",
                "extra_time_ce",
                "extra_time",
                "extra_time_right",
            ]
        )

        if buffer_reinit_needed:
            self.logger.info("🔄 API: Reinitializing buffers due to timing parameter changes")
            self.model._init_buffers()

        self.logger.info("✅ API: Model parameters updated: %s", list(updated_params.keys()))

        return JSONResponse(
            content={
                "message": "Model parameters updated successfully",
                "updated_parameters": updated_params,
                "current_parameters": {
                    "block_time": self.model.block_time,
                    "crossfade_time": self.model.crossfade_time,
                    "extra_time_ce": self.model.extra_time_ce,
                    "extra_time": self.model.extra_time,
                    "extra_time_right": self.model.extra_time_right,
                    "diffusion_steps": self.model.diffusion_steps,
                    "max_prompt_length": self.model.max_prompt_length,
                    "inference_cfg_rate": self.model.inference_cfg_rate,
                    "use_vad": self.model.use_vad,
                },
            },
            status_code=200,
        )

    @require_no_connected_clients_method
    def reload_model(self, request: ModelReloadRequest) -> JSONResponse:
        """Reload the model with new checkpoint and config files.

        This method updates the model in-place by reloading the internal components
        without creating a new VoiceConverter instance. This ensures that the
        global_converter reference in server.py remains valid.

        Args:
            request: Model reload request containing checkpoint and config paths.

        Returns:
            JSON response with reload status.

        Raises:
            HTTPException: If model reload fails.
        """
        try:
            with self.model_lock:
                self.logger.info("🔄 API: Reload model")

                # Validate checkpoint and config path combination
                if request.checkpoint_path and not request.config_path:
                    raise HTTPException(
                        status_code=400,
                        detail="config_path is required when checkpoint_path is provided",
                    )

                # Update model's checkpoint and config paths
                self.model.checkpoint_path = request.checkpoint_path
                self.model.config_path = request.config_path

                # Log the paths being used
                if request.checkpoint_path:
                    self.logger.info(
                        "🔄 API: Loading custom model from %s", request.checkpoint_path
                    )
                else:
                    self.logger.info("🔄 API: Loading default model from HuggingFace")

                # Clear cached prompt condition to force recalculation
                self._clear_reference_cache()

                # Reload models in-place
                self.logger.info("🔄 API: Reloading model components...")
                self.model.model_set = self.model._load_models()

                # Reinitialize buffers if needed
                self.model._init_buffers()

                self.logger.info("✅ API: Model reloaded successfully")

                return JSONResponse(
                    content={
                        "status": "success",
                        "message": "Model reloaded successfully",
                        "checkpoint_path": request.checkpoint_path or "default (HuggingFace)",
                        "config_path": request.config_path or "default",
                    }
                )

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error("❌ API: Failed to reload model: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}") from e
