"""Tests for VoiceConverter model."""

import sys
from pathlib import Path

import librosa
import numpy as np
import pytest
import torch

from seed_vc.socketio.model import VoiceConverter

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class TestVoiceConverter:
    """Test cases for VoiceConverter class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.test_audio_path = project_root / "assets/examples/reference/teio_0.wav"
        self.test_source_path = project_root / "assets/examples/source/jay_0.wav"

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_voice_converter_can_be_loaded(self):
        """Test that VoiceConverter can be instantiated with real models."""
        from seed_vc.socketio.model import VoiceConverter

        # Test that VoiceConverter can be instantiated with real models
        # This will download models from HuggingFace if not cached
        converter = VoiceConverter(
            input_sampling_rate=44100,
            block_time=0.25,
            diffusion_steps=5,  # Use fewer steps for faster testing
        )

        # Basic checks
        assert converter is not None
        assert converter.input_sampling_rate == 44100
        assert converter.block_time == 0.25
        assert hasattr(converter, "model_set")
        assert len(converter.model_set) == 6  # Should have 6 components

        # Check that models are properly loaded
        model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args = converter.model_set
        assert model is not None
        assert semantic_fn is not None
        assert vocoder_fn is not None
        assert campplus_model is not None
        assert to_mel is not None
        assert isinstance(mel_fn_args, dict)
        assert "sampling_rate" in mel_fn_args

    @pytest.mark.skipif(
        not Path(__file__)
        .resolve()
        .parents[2]
        .joinpath("assets/examples/reference/teio_0.wav")
        .exists(),
        reason="Test audio files not available",
    )
    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_audio_processing_with_real_files(self):
        """Test audio processing with real audio files and real models."""
        # Load test audio
        source_audio, _sr = librosa.load(self.test_source_path, sr=44100, mono=True)

        # Create converter with real models and test reference audio
        converter = VoiceConverter(
            reference_wav_path=str(self.test_audio_path),
            input_sampling_rate=44100,
            block_time=0.25,  # 250ms blocks
            diffusion_steps=5,  # Use fewer steps for faster testing
        )

        # Test that converter has necessary attributes for audio processing
        assert hasattr(converter, "block_frame")
        assert hasattr(converter, "input_sampling_rate")
        assert hasattr(converter, "custom_infer")
        assert hasattr(converter, "audio_callback")

        # Test basic chunk processing setup
        chunk_size = int(0.25 * 44100)  # 250ms at 44.1kHz
        assert converter.block_frame > 0
        assert converter.input_sampling_rate == 44100

        # Verify that audio can be divided into chunks
        num_chunks = (len(source_audio) + chunk_size - 1) // chunk_size
        assert num_chunks > 0

        # Test that we can create properly shaped audio chunks
        for i in range(min(2, num_chunks)):  # Test only first 2 chunks
            chunk = source_audio[i * chunk_size : (i + 1) * chunk_size]

            # Pad last chunk if necessary
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            # Verify chunk shape
            indata = chunk.reshape(-1, 1).astype(np.float32)
            assert indata.shape[0] == chunk_size
            assert indata.shape[1] == 1
            assert indata.dtype == np.float32

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_audio_callback_with_real_models(self):
        """Test audio_callback method with real models."""
        from seed_vc.socketio.model import VoiceConverter

        # Create converter with minimal configuration for testing
        converter = VoiceConverter(
            input_sampling_rate=44100,
            block_time=0.18,  # Match server configuration
            diffusion_steps=3,  # Use very few steps for speed
        )

        # Create test audio data matching block_frame size
        frames = converter.block_frame
        indata = np.random.normal(0, 0.1, (frames, 1)).astype(np.float32)
        outdata = np.zeros_like(indata)

        # Test audio_callback doesn't crash
        try:
            converter.audio_callback(
                indata=indata, outdata=outdata, frames=frames, time=None, status=None
            )
            # If we get here, the callback executed successfully
            assert True
        except Exception as e:
            pytest.fail(f"audio_callback failed with real models: {e}")

        # Verify outdata was modified (should not be all zeros)
        assert not np.allclose(outdata, 0)
        assert outdata.shape == indata.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_buffer_size_handling(self):
        """Test that VoiceConverter handles different buffer sizes correctly."""
        from seed_vc.socketio.model import VoiceConverter

        # Test with a single block_time to verify basic functionality
        converter = VoiceConverter(
            input_sampling_rate=44100,
            block_time=0.25,
            diffusion_steps=1,  # Minimal for speed
        )

        # Verify buffer sizes are calculated correctly
        expected_block_frame = int(np.round(0.25 * 44100 / converter.zc)) * converter.zc
        assert converter.block_frame == expected_block_frame

        # Test that buffers are properly sized
        assert converter.input_wav.shape[0] > converter.block_frame
        assert converter.input_wav_res.shape[0] > 0

        # Test with the exact block_frame size (should work correctly)
        test_size = converter.block_frame
        indata = np.random.normal(0, 0.1, (test_size, 1)).astype(np.float32)
        outdata = np.zeros_like(indata)

        # Should not crash with matching size
        try:
            converter.audio_callback(
                indata=indata, outdata=outdata, frames=test_size, time=None, status=None
            )
        except Exception as e:
            pytest.fail(f"audio_callback failed with matching block size: {e}")

        # Verify outdata was modified (should not be all zeros)
        assert not np.allclose(outdata, 0)

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_init_buffers_resets_vad_state(self):
        """Test that _init_buffers resets VAD state when block_time changes."""
        converter = VoiceConverter(
            input_sampling_rate=44100,
            block_time=0.18,
            diffusion_steps=1,
        )

        # Simulate stale VAD state from a previous session
        converter.vad_cache = {"some_key": "stale_value"}
        converter.vad_speech_detected = True
        converter.set_speech_detected_false_at_end_flag = True
        old_vad_chunk_size = converter.vad_chunk_size

        # Change block_time and reinitialize (mimics API parameter update)
        converter.block_time = 0.12
        converter._init_buffers()

        # VAD state should be fully reset
        assert converter.vad_cache == {}
        assert converter.vad_speech_detected is False
        assert converter.set_speech_detected_false_at_end_flag is False
        # vad_chunk_size should be recalculated for new block_time
        expected_vad_chunk_size = min(500, 1000 * 0.12)  # = 120
        assert converter.vad_chunk_size == expected_vad_chunk_size
        assert converter.vad_chunk_size != old_vad_chunk_size

    def test_voice_converter_basic_initialization(self):
        """Test basic VoiceConverter initialization without GPU requirements."""
        # Test basic initialization
        converter = VoiceConverter(
            input_sampling_rate=44100,
            block_time=0.25,
            diffusion_steps=1,  # Minimal for speed
            gpu=-1,  # Force CPU mode
        )

        # Basic checks that don't require model loading
        assert converter is not None
        assert converter.input_sampling_rate == 44100
        assert converter.block_time == 0.25
        assert converter.device.type == "cpu"

        # Check buffer calculations
        assert converter.block_frame > 0
        assert converter.block_frame_16k > 0
        assert converter.input_wav.shape[0] > 0
        assert converter.input_wav_res.shape[0] > 0
