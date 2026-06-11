"""Tests for offline file conversion API endpoints."""

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from seed_vc.socketio.api import APIRouterVCModel
from seed_vc.socketio.model import VoiceConverter

SAMPLE_RATE = 44100


class VoiceConverterMock(VoiceConverter):
    """Mock VoiceConverter that skips model loading for faster tests."""

    def _load_models(self):
        """Override to skip actual model loading."""
        model = {"dummy": "model"}
        semantic_fn = lambda x: x
        vocoder_fn = lambda x: x
        campplus_model = None
        to_mel = lambda x: x
        mel_fn_args = {}

        return model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args

    def _init_buffers(self):
        """Override to skip buffer initialization."""
        self.model_set = [{"sampling_rate": 16000}]
        self.zc = self.input_sampling_rate // 50
        self.block_frame = (
            int(round(self.block_time * self.input_sampling_rate / self.zc)) * self.zc
        )

    def convert_file(self, input_path: str, output_path: str) -> dict:
        """Override to skip actual conversion and write a dummy output file."""
        input_wav, _ = sf.read(input_path)
        if len(input_wav) == 0:
            raise ValueError(f"Input audio file is empty: {input_path}")
        sf.write(output_path, np.zeros(len(input_wav), dtype=np.float32), SAMPLE_RATE)
        duration = len(input_wav) / SAMPLE_RATE
        return {
            "message": "File conversion completed successfully",
            "input_path": input_path,
            "output_path": output_path,
            "input_duration": duration,
            "output_duration": duration,
            "sampling_rate": SAMPLE_RATE,
        }


@pytest.fixture
def voice_converter():
    """Create a VoiceConverter instance for testing."""
    return VoiceConverterMock(input_sampling_rate=SAMPLE_RATE, block_time=0.18)


@pytest.fixture
def allowed_dir(tmp_path):
    """Create a temporary allowed directory containing an input audio file."""
    sf.write(tmp_path / "input.wav", np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    return tmp_path


@pytest.fixture
def api_router(voice_converter, allowed_dir):
    """Create APIRouterVCModel instance with a temporary allowed directory."""
    return APIRouterVCModel(model=voice_converter, allowed_audio_dirs=[str(allowed_dir)])


def build_test_client(api_router) -> TestClient:
    """Create a FastAPI test client for the given API router."""
    app = FastAPI()
    app.include_router(api_router.api_router, prefix="/api/v1")
    return TestClient(app)


@pytest.fixture
def test_client(api_router):
    """Create FastAPI test client."""
    return build_test_client(api_router)


def make_wav_bytes(duration_sec: float = 0.1) -> bytes:
    """Create WAV binary data for upload tests."""
    import io

    buf = io.BytesIO()
    sf.write(
        buf, np.zeros(int(SAMPLE_RATE * duration_sec), dtype=np.float32), SAMPLE_RATE, format="WAV"
    )
    return buf.getvalue()


class TestConvertEndpoint:
    """Test the file-path-based conversion endpoint."""

    def test_convert_success(self, test_client, allowed_dir):
        """Test successful file conversion with valid paths."""
        input_path = str(allowed_dir / "input.wav")
        output_path = str(allowed_dir / "output.wav")
        response = test_client.post(
            "/api/v1/convert",
            json={"input_path": input_path, "output_path": output_path},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "File conversion completed successfully"
        assert data["sampling_rate"] == SAMPLE_RATE
        assert (allowed_dir / "output.wav").exists()

    def test_convert_input_not_found(self, test_client, allowed_dir):
        """Test conversion with a non-existent input file."""
        response = test_client.post(
            "/api/v1/convert",
            json={
                "input_path": str(allowed_dir / "missing.wav"),
                "output_path": str(allowed_dir / "output.wav"),
            },
        )

        assert response.status_code == 404

    def test_convert_input_outside_allowed_directory(
        self, test_client, allowed_dir, tmp_path_factory
    ):
        """Test conversion with an input file outside allowed directories."""
        outside_dir = tmp_path_factory.mktemp("outside")
        sf.write(outside_dir / "input.wav", np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        response = test_client.post(
            "/api/v1/convert",
            json={
                "input_path": str(outside_dir / "input.wav"),
                "output_path": str(allowed_dir / "output.wav"),
            },
        )

        assert response.status_code == 403

    def test_convert_output_outside_allowed_directory(
        self, test_client, allowed_dir, tmp_path_factory
    ):
        """Test conversion with an output path outside allowed directories."""
        outside_dir = tmp_path_factory.mktemp("outside")
        response = test_client.post(
            "/api/v1/convert",
            json={
                "input_path": str(allowed_dir / "input.wav"),
                "output_path": str(outside_dir / "output.wav"),
            },
        )

        assert response.status_code == 403

    def test_convert_output_directory_not_found(self, test_client, allowed_dir):
        """Test conversion with an output path whose parent directory does not exist."""
        response = test_client.post(
            "/api/v1/convert",
            json={
                "input_path": str(allowed_dir / "input.wav"),
                "output_path": str(allowed_dir / "missing_dir" / "output.wav"),
            },
        )

        assert response.status_code == 400

    def test_convert_blocked_while_clients_connected(self, voice_converter, allowed_dir):
        """Test that conversion is blocked while realtime clients are connected."""
        api_router = APIRouterVCModel(
            model=voice_converter,
            allowed_audio_dirs=[str(allowed_dir)],
            client_count_checker=lambda: 1,
        )
        client = build_test_client(api_router)

        response = client.post(
            "/api/v1/convert",
            json={
                "input_path": str(allowed_dir / "input.wav"),
                "output_path": str(allowed_dir / "output.wav"),
            },
        )

        assert response.status_code == 409

    def test_convert_reference_not_set(
        self, test_client, voice_converter, allowed_dir, monkeypatch
    ):
        """Test conversion when reference audio is not set."""

        def raise_runtime_error(input_path, output_path):
            raise RuntimeError("Reference audio not set. Call update_reference_audio() first.")

        monkeypatch.setattr(voice_converter, "convert_file", raise_runtime_error)
        response = test_client.post(
            "/api/v1/convert",
            json={
                "input_path": str(allowed_dir / "input.wav"),
                "output_path": str(allowed_dir / "output.wav"),
            },
        )

        assert response.status_code == 400


class TestConvertUploadEndpoint:
    """Test the upload-based conversion endpoint."""

    def test_upload_success(self, test_client):
        """Test successful conversion of an uploaded file."""
        response = test_client.post(
            "/api/v1/convert/upload",
            files={"input_file": ("input.wav", make_wav_bytes(), "audio/wav")},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert len(response.content) > 0

    def test_upload_with_reference_restores_previous_reference(self, test_client, voice_converter):
        """Test that uploading a temporary reference restores the previous one."""
        previous_path = voice_converter.reference_wav_path
        previous_wav = voice_converter.reference_wav

        response = test_client.post(
            "/api/v1/convert/upload",
            files={
                "input_file": ("input.wav", make_wav_bytes(), "audio/wav"),
                "reference_file": ("reference.wav", make_wav_bytes(), "audio/wav"),
            },
        )

        assert response.status_code == 200
        assert voice_converter.reference_wav_path == previous_path
        assert voice_converter.reference_wav is previous_wav

    def test_upload_too_large(self, test_client):
        """Test that an oversized upload is rejected."""
        oversized = b"\x00" * (25 * 1024 * 1024 + 1)
        response = test_client.post(
            "/api/v1/convert/upload",
            files={"input_file": ("input.wav", oversized, "audio/wav")},
        )

        assert response.status_code == 413

    def test_upload_blocked_while_clients_connected(self, voice_converter, allowed_dir):
        """Test that upload conversion is blocked while realtime clients are connected."""
        api_router = APIRouterVCModel(
            model=voice_converter,
            allowed_audio_dirs=[str(allowed_dir)],
            client_count_checker=lambda: 1,
        )
        client = build_test_client(api_router)

        response = client.post(
            "/api/v1/convert/upload",
            files={"input_file": ("input.wav", make_wav_bytes(), "audio/wav")},
        )

        assert response.status_code == 409


class TestOfflineJobExclusion:
    """Test exclusion between concurrent offline conversion jobs."""

    def test_convert_blocked_while_offline_job_active(self, test_client, api_router, allowed_dir):
        """Test that /convert returns 409 while another offline job is active."""
        assert api_router.runtime.try_begin_offline_job()
        try:
            response = test_client.post(
                "/api/v1/convert",
                json={
                    "input_path": str(allowed_dir / "input.wav"),
                    "output_path": str(allowed_dir / "output.wav"),
                },
            )
        finally:
            api_router.runtime.finish_offline_job()

        assert response.status_code == 409

    def test_upload_blocked_while_offline_job_active(self, test_client, api_router):
        """Test that /convert/upload returns 409 while another offline job is active."""
        assert api_router.runtime.try_begin_offline_job()
        try:
            response = test_client.post(
                "/api/v1/convert/upload",
                files={"input_file": ("input.wav", make_wav_bytes(), "audio/wav")},
            )
        finally:
            api_router.runtime.finish_offline_job()

        assert response.status_code == 409

    def test_offline_job_released_after_conversion(self, test_client, api_router, allowed_dir):
        """Test that the offline job reservation is released after each conversion."""
        request_json = {
            "input_path": str(allowed_dir / "input.wav"),
            "output_path": str(allowed_dir / "output.wav"),
        }

        first = test_client.post("/api/v1/convert", json=request_json)
        second = test_client.post("/api/v1/convert", json=request_json)

        assert first.status_code == 200
        assert second.status_code == 200
        assert not api_router.runtime.is_offline_job_active()

    def test_offline_job_released_after_conversion_error(
        self, test_client, api_router, allowed_dir
    ):
        """Test that the offline job reservation is released after a failed conversion."""
        response = test_client.post(
            "/api/v1/convert",
            json={
                "input_path": str(allowed_dir / "missing.wav"),
                "output_path": str(allowed_dir / "output.wav"),
            },
        )

        assert response.status_code == 404
        assert not api_router.runtime.is_offline_job_active()
