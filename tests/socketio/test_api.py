"""Tests for API router using real VoiceConverter model."""

import os
import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from seed_vc.socketio.api import APIRouterVCModel
from seed_vc.socketio.model import ConversionMode, VoiceConverter


class VoiceConverterMock(VoiceConverter):
    """Mock VoiceConverter that skips model loading for faster tests."""

    def _load_models(self):
        """Override to skip actual model loading."""
        # Return dummy values that won't be used in API tests
        model = {"dummy": "model"}
        semantic_fn = lambda x: x
        vocoder_fn = lambda x: x
        campplus_model = None
        to_mel = lambda x: x
        mel_fn_args = {}

        return model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args

    def _init_buffers(self):
        """Override to skip buffer initialization."""
        # Set minimal required attributes
        self.model_set = [{"sampling_rate": 16000}]
        # Set zc-aligned block_frame using the same formula as the real _init_buffers()
        self.zc = self.input_sampling_rate // 50
        self.block_frame = (
            int(round(self.block_time * self.input_sampling_rate / self.zc)) * self.zc
        )


@pytest.fixture
def voice_converter():
    """Create a VoiceConverter instance for testing."""
    # Use the mock converter for faster tests
    converter = VoiceConverterMock(
        input_sampling_rate=44100,
        block_time=0.18,
        crossfade_time=0.04,
        extra_time_ce=2.5,
        extra_time=0.5,
        extra_time_right=0.02,
        diffusion_steps=10,
        max_prompt_length=3.0,
        inference_cfg_rate=0.7,
        use_vad=False,
    )
    return converter


@pytest.fixture
def api_router(voice_converter):
    """Create APIRouterVCModel instance."""
    return APIRouterVCModel(model=voice_converter)


@pytest.fixture
def test_client(api_router):
    """Create FastAPI test client."""
    app = FastAPI()
    app.include_router(api_router.api_router, prefix="/api/v1")
    return TestClient(app)


class TestConfigEndpoint:
    """Test the server audio config endpoint."""

    def test_get_config_returns_audio_configuration(self, test_client):
        """Test that /config returns the server audio configuration for client auto-setup."""
        response = test_client.get("/api/v1/config")

        assert response.status_code == 200
        data = response.json()
        assert data["input_sampling_rate"] == 44100
        assert data["block_time"] == 0.18
        # chunk_size must match the zc-aligned block_frame used in connection validation
        assert data["chunk_size"] == 7938


class TestConversionModeEndpoint:
    """Test the conversion mode endpoint."""

    def test_change_mode_to_passthrough(self, test_client, voice_converter):
        """Test changing conversion mode to passthrough."""
        response = test_client.post("/api/v1/mode", json={"mode": "passthrough"})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Conversion mode updated successfully"
        assert data["mode"] == "passthrough"
        assert voice_converter.conversion_mode == ConversionMode.PASSTHROUGH

    def test_change_mode_to_silence(self, test_client, voice_converter):
        """Test changing conversion mode to silence."""
        response = test_client.post("/api/v1/mode", json={"mode": "silence"})
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "silence"
        assert voice_converter.conversion_mode == ConversionMode.SILENCE

    def test_change_mode_to_convert(self, test_client, voice_converter):
        """Test changing conversion mode back to convert."""
        # First change to another mode
        test_client.post("/api/v1/mode", json={"mode": "silence"})

        # Then change back to convert
        response = test_client.post("/api/v1/mode", json={"mode": "convert"})
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "convert"
        assert voice_converter.conversion_mode == ConversionMode.CONVERT

    def test_invalid_mode(self, test_client):
        """Test that invalid mode returns error."""
        response = test_client.post("/api/v1/mode", json={"mode": "invalid_mode"})
        assert response.status_code == 422  # Pydantic validation error for Literal type
        assert "detail" in response.json()
        # Check that the error message indicates invalid value
        details = response.json()["detail"]
        assert any("invalid_mode" in str(detail) for detail in details)

    def test_available_modes_listed(self, test_client):
        """Test that response includes available modes."""
        response = test_client.post("/api/v1/mode", json={"mode": "convert"})
        assert response.status_code == 200
        data = response.json()
        assert "available_modes" in data
        assert set(data["available_modes"]) == {"convert", "passthrough", "silence"}


class TestReferenceAudioEndpoint:
    """Test the reference audio endpoint."""

    def test_update_reference_audio(self, test_client, voice_converter):
        """Test updating reference audio with valid file."""
        # Use existing test audio file from assets
        test_audio_path = "assets/examples/reference/teio_0.wav"

        # Check if file exists, if not skip test
        if not os.path.exists(test_audio_path):
            pytest.skip(f"Test audio file not found: {test_audio_path}")

        response = test_client.post("/api/v1/reference", json={"file_path": test_audio_path})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Reference audio updated successfully"
        # Since path validation resolves to absolute path, check if it ends with our test path
        assert data["reference_path"].endswith("assets/examples/reference/teio_0.wav")
        assert voice_converter.reference_wav_path.endswith("assets/examples/reference/teio_0.wav")

    def test_update_reference_audio_file_not_found(self, test_client):
        """Test updating reference audio with non-existent file."""
        response = test_client.post(
            "/api/v1/reference", json={"file_path": "/path/to/nonexistent/file.wav"}
        )
        assert response.status_code == 404
        assert "Audio file not found" in response.json()["detail"]

    def test_directory_traversal_attack_prevention(self, test_client):
        """Test that directory traversal attacks are prevented."""
        # Test various directory traversal attempts
        traversal_attempts = [
            "../../../etc/passwd",
            "../../../../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "/var/log/syslog",
            "assets/examples/reference/../../../etc/passwd",
        ]

        for malicious_path in traversal_attempts:
            response = test_client.post("/api/v1/reference", json={"file_path": malicious_path})
            # Should get either 404 (file not found) or 403 (access denied) or 400 (invalid path)
            assert response.status_code in [400, 403, 404]
            detail = response.json()["detail"]
            assert any(
                keyword in detail.lower()
                for keyword in ["access denied", "invalid file path", "audio file not found"]
            )

    def test_file_outside_allowed_directory(self, test_client):
        """Test that files outside allowed directories are rejected."""
        # Create a temporary file outside the allowed directory
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            # Write minimal WAV header (44 bytes) + some data
            tmp_file.write(b"RIFF" + b"\x00" * 40)

        try:
            response = test_client.post("/api/v1/reference", json={"file_path": tmp_path})
            # Should be denied access since it's not in allowed directories
            assert response.status_code == 403
            assert "Access denied" in response.json()["detail"]
            assert "allowed directories" in response.json()["detail"]
        finally:
            os.unlink(tmp_path)

    def test_custom_allowed_directories(self, voice_converter):
        """Test APIRouterVCModel with custom allowed directories."""
        # Create temporary directory and file
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.wav")
            # Create a proper minimal WAV file that librosa can load
            import wave

            with wave.open(test_file, "wb") as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16 kHz
                wav_file.writeframes(b"\x00" * 3200)  # 0.1 seconds of silence

            # Create API router with custom allowed directory
            api_router = APIRouterVCModel(model=voice_converter, allowed_audio_dirs=[temp_dir])
            app = FastAPI()
            app.include_router(api_router.api_router, prefix="/api/v1")
            test_client = TestClient(app)

            # This should work since the file is in allowed directory
            response = test_client.post("/api/v1/reference", json={"file_path": test_file})
            assert response.status_code == 200


class TestModelParametersEndpoint:
    """Test the model parameters endpoint."""

    def test_update_single_parameter(self, test_client, voice_converter):
        """Test updating a single parameter."""
        response = test_client.post("/api/v1/parameters", json={"diffusion_steps": 20})
        assert response.status_code == 200
        data = response.json()
        assert data["updated_parameters"] == {"diffusion_steps": 20}
        assert voice_converter.diffusion_steps == 20
        # Other parameters should remain unchanged
        assert voice_converter.block_time == 0.18

    def test_update_multiple_parameters(self, test_client, voice_converter):
        """Test updating multiple parameters at once."""
        response = test_client.post(
            "/api/v1/parameters",
            json={
                "block_time": 0.2,
                "diffusion_steps": 15,
                "inference_cfg_rate": 0.8,
                "use_vad": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updated_parameters"] == {
            "block_time": 0.2,
            "diffusion_steps": 15,
            "inference_cfg_rate": 0.8,
            "use_vad": True,
        }
        assert voice_converter.block_time == 0.2
        assert voice_converter.diffusion_steps == 15
        assert voice_converter.inference_cfg_rate == 0.8
        assert voice_converter.use_vad is True

    def test_current_parameters_returned(self, test_client):
        """Test that current parameters are returned in response."""
        response = test_client.post("/api/v1/parameters", json={"max_prompt_length": 4.0})
        assert response.status_code == 200
        data = response.json()
        current = data["current_parameters"]
        assert "block_time" in current
        assert "crossfade_time" in current
        assert "diffusion_steps" in current
        assert current["max_prompt_length"] == 4.0

    def test_invalid_parameter_values(self, test_client):
        """Test validation of parameter values."""
        # Test negative block_time
        response = test_client.post("/api/v1/parameters", json={"block_time": -0.1})
        assert response.status_code == 422
        assert "detail" in response.json()

        # Test out of range inference_cfg_rate
        response = test_client.post("/api/v1/parameters", json={"inference_cfg_rate": 1.5})
        assert response.status_code == 422
        assert "detail" in response.json()

        # Test zero diffusion_steps
        response = test_client.post("/api/v1/parameters", json={"diffusion_steps": 0})
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_empty_request(self, test_client, voice_converter):
        """Test that empty request doesn't change anything."""
        original_block_time = voice_converter.block_time
        original_steps = voice_converter.diffusion_steps

        response = test_client.post("/api/v1/parameters", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["updated_parameters"] == {}
        assert voice_converter.block_time == original_block_time
        assert voice_converter.diffusion_steps == original_steps


class TestModelReloadEndpoint:
    """Test the model reload endpoint."""

    def test_reload_with_default_model(self, test_client, voice_converter):
        """Test reloading with default model (HuggingFace)."""
        # Clear any checkpoint/config paths
        voice_converter.checkpoint_path = None
        voice_converter.config_path = None

        response = test_client.post(
            "/api/v1/reload", json={"checkpoint_path": None, "config_path": None}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["message"] == "Model reloaded successfully"
        assert data["checkpoint_path"] == "default (HuggingFace)"
        assert data["config_path"] == "default"

    def test_reload_with_custom_model(self, test_client, voice_converter):
        """Test reloading with custom checkpoint and config paths."""
        # Mock paths (these don't need to exist for the mock converter)
        checkpoint_path = "path/to/custom_model.pth"
        config_path = "path/to/custom_config.yml"

        response = test_client.post(
            "/api/v1/reload", json={"checkpoint_path": checkpoint_path, "config_path": config_path}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["checkpoint_path"] == checkpoint_path
        assert data["config_path"] == config_path
        assert voice_converter.checkpoint_path == checkpoint_path
        assert voice_converter.config_path == config_path

    def test_reload_checkpoint_without_config(self, test_client):
        """Test that checkpoint without config returns error."""
        response = test_client.post(
            "/api/v1/reload", json={"checkpoint_path": "path/to/model.pth", "config_path": None}
        )
        assert response.status_code == 400
        assert "config_path is required" in response.json()["detail"]

    def test_reload_clears_cached_data(self, test_client, voice_converter):
        """Test that reload clears cached prompt condition."""
        # Set some cached values
        voice_converter.prompt_condition = "dummy_condition"
        voice_converter.mel2 = "dummy_mel"
        voice_converter.style2 = "dummy_style"
        voice_converter.reference_wav_name = "dummy_name"

        response = test_client.post(
            "/api/v1/reload", json={"checkpoint_path": None, "config_path": None}
        )
        assert response.status_code == 200

        # Check that cached values were cleared
        assert voice_converter.prompt_condition is None
        assert voice_converter.mel2 is None
        assert voice_converter.style2 is None
        assert voice_converter.reference_wav_name == ""

    def test_reload_preserves_other_parameters(self, test_client, voice_converter):
        """Test that reload preserves other model parameters."""
        # Set some custom parameters first
        voice_converter.block_time = 0.25
        voice_converter.diffusion_steps = 20
        voice_converter.inference_cfg_rate = 0.8

        response = test_client.post(
            "/api/v1/reload", json={"checkpoint_path": None, "config_path": None}
        )
        assert response.status_code == 200

        # Check that parameters were preserved
        assert voice_converter.block_time == 0.25
        assert voice_converter.diffusion_steps == 20
        assert voice_converter.inference_cfg_rate == 0.8

    def test_reload_with_empty_request(self, test_client):
        """Test reload with empty request body."""
        response = test_client.post("/api/v1/reload", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["checkpoint_path"] == "default (HuggingFace)"
        assert data["config_path"] == "default"
