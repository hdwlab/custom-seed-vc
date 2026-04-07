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

"""Voice conversion model for real-time audio processing."""

import logging
import os
import sys
import warnings
from contextlib import contextmanager
from enum import Enum
from functools import partialmethod
from typing import Any, Callable, Dict, Optional, Tuple

import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as tat
import yaml
from dotenv import load_dotenv
from funasr import AutoModel
from tqdm import tqdm

from seed_vc.hf_utils import load_custom_model_from_hf
from seed_vc.modules.campplus.DTDNN import CAMPPlus
from seed_vc.modules.commons import build_model, load_checkpoint, recursive_munch

load_dotenv()

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)
logger.propagate = False  # Prevent duplicate logs to root logger
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [MODEL] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@contextmanager
def suppress_print_and_logging(log_level=logging.CRITICAL + 1):
    """Suppress both print output and logging temporarily."""
    # Save original values
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Save all logger levels
    loggers = [logging.getLogger()]  # Root logger
    # Add all existing loggers
    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        loggers.append(logger)

    original_levels = {logger: logger.level for logger in loggers}

    try:
        # Redirect stdout/stderr to devnull
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

        # Set all loggers to higher than CRITICAL
        for logger in loggers:
            logger.setLevel(log_level)

        yield

    finally:
        # Restore stdout/stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Restore logger levels
        for logger, level in original_levels.items():
            logger.setLevel(level)


class ConversionMode(Enum):
    """Voice conversion mode enumeration."""

    CONVERT = "convert"  # Apply voice conversion
    PASSTHROUGH = "passthrough"  # Pass input audio without modification
    SILENCE = "silence"  # Return silence


class VoiceConverter:
    """Voice converter class that encapsulates all voice conversion functionality."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        fp16: bool = False,
        gpu: int = 0,
        reference_wav_path: Optional[str] = None,
        input_sampling_rate: int = 44100,
        block_time: float = 0.18,
        crossfade_time: float = 0.04,
        extra_time_ce: float = 2.5,
        extra_time: float = 0.5,
        extra_time_right: float = 0.02,
        diffusion_steps: int = 10,
        max_prompt_length: float = 3.0,
        inference_cfg_rate: float = 0.7,
        use_vad: bool = False,
        log_level: str = "INFO",
    ):
        """Initialize VoiceConverter with configuration parameters.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to model configuration
            fp16: Whether to use float16 precision
            gpu: GPU device index (-1 for CPU)
            reference_wav_path: Path to reference voice audio
            input_sampling_rate: Input audio sampling rate
            block_time: Audio block processing time in seconds
            crossfade_time: Crossfade duration in seconds
            extra_time_ce: Extra context time for content encoder
            extra_time: Extra context time for DiT
            extra_time_right: Right-side extra context time
            diffusion_steps: Number of diffusion steps
            max_prompt_length: Maximum prompt length in seconds
            inference_cfg_rate: Classifier-free guidance rate
            use_vad: Whether to use a VAD model (not implemented yet)
            log_level: Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR")
        """
        # Configure logger level for this instance
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        logger.setLevel(numeric_level)
        self.logger = logger

        # Disable tqdm
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

        # Model configuration
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.fp16 = fp16
        self.gpu = gpu

        # Audio configuration
        self.input_sampling_rate = input_sampling_rate
        self.block_time = block_time
        self.crossfade_time = crossfade_time
        self.extra_time_ce = extra_time_ce
        self.extra_time = extra_time
        self.extra_time_right = extra_time_right

        # Inference configuration
        self.diffusion_steps = diffusion_steps
        self.max_prompt_length = max_prompt_length
        self.inference_cfg_rate = inference_cfg_rate
        self.channels = 1

        # Initialize device
        self.device = self._init_device()

        # Initialize models
        self.model_set = self._load_models()

        # Initialize buffers and parameters
        self._init_buffers()

        # Load reference audio if provided
        self.reference_wav_path = reference_wav_path
        if reference_wav_path:
            self.reference_wav, _ = librosa.load(
                reference_wav_path,
                sr=self.model_set[-1]["sampling_rate"],
            )
        else:
            # Default reference path
            self.reference_wav_path = "assets/examples/reference/trump_0.wav"
            self.reference_wav, _ = librosa.load(
                self.reference_wav_path,
                sr=self.model_set[-1]["sampling_rate"],
            )

        # Cache for voice conversion
        self.prompt_condition: Optional[torch.Tensor] = None
        self.mel2: Optional[torch.Tensor] = None
        self.style2: Optional[torch.Tensor] = None
        self.reference_wav_name: str = ""
        self.prompt_len: float = 3.0
        self.ce_dit_difference: float = 2.0

        # VAD configuration
        self.vad_cache = {}
        self.vad_chunk_size = min(500, 1000 * self.block_time)
        self.vad_speech_detected = False
        self.set_speech_detected_false_at_end_flag = False
        self.use_vad = use_vad
        self.vad_model: Optional[AutoModel] = None
        if self.use_vad:
            with suppress_print_and_logging():
                self.vad_model = AutoModel(
                    model="fsmn-vad", model_revision="v2.0.4", disable_update=True
                )

        # Output buffer
        self.output_wav = []

        # Conversion mode
        self.conversion_mode = ConversionMode.CONVERT

    def _init_device(self) -> torch.device:
        """Initialize torch device based on availability and configuration."""
        return torch.device(
            f"cuda:{self.gpu}"
            if torch.cuda.is_available() and self.gpu >= 0
            else "mps"
            if sys.platform == "darwin"
            else "cpu",
        )

    def _create_timing_events(self) -> Tuple[Any, Any]:
        """Create timing events for performance measurement based on device type.

        Returns:
            Tuple of (start_event, end_event).
        """
        if self.device.type == "mps":
            start_event = torch.mps.event.Event(enable_timing=True)
            end_event = torch.mps.event.Event(enable_timing=True)
            torch.mps.synchronize()
        else:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
        return start_event, end_event

    def _synchronize_device(self) -> None:
        """Synchronize device based on device type."""
        if self.device.type == "mps":
            torch.mps.synchronize()
        else:
            torch.cuda.synchronize()

    def _calculate_elapsed_time(self, start_event: Any, end_event: Any) -> float:
        """Calculate elapsed time between two events.

        Args:
            start_event: Start timing event.
            end_event: End timing event.

        Returns:
            Elapsed time in milliseconds.
        """
        self._synchronize_device()
        return start_event.elapsed_time(end_event)

    def _init_buffers(self):
        """Initialize audio processing buffers."""
        # Frame calculations
        self.zc = self.input_sampling_rate // 50
        self.block_frame = (
            int(np.round(self.block_time * self.input_sampling_rate / self.zc)) * self.zc
        )
        self.block_frame_16k = 320 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(np.round(self.crossfade_time * self.input_sampling_rate / self.zc)) * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(np.round(self.extra_time_ce * self.input_sampling_rate / self.zc)) * self.zc
        )
        self.extra_frame_right = (
            int(np.round(self.extra_time_right * self.input_sampling_rate / self.zc)) * self.zc
        )

        # Audio buffers
        self.input_wav = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
            + self.extra_frame_right,
            device=self.device,
            dtype=torch.float32,
        )

        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            320 * self.input_wav.shape[0] // self.zc,
            device=self.device,
            dtype=torch.float32,
        )

        self.rms_buffer = np.zeros(4 * self.zc, dtype=np.float32)
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame,
            device=self.device,
            dtype=torch.float32,
        )
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()

        # Processing parameters
        self.skip_head = self.extra_frame // self.zc
        self.skip_tail = self.extra_frame_right // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc

        # Window functions
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    self.sola_buffer_frame,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window

        # Resamplers
        self.resampler = tat.Resample(
            orig_freq=self.input_sampling_rate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.device)

        if self.model_set[-1]["sampling_rate"] != self.input_sampling_rate:
            self.resampler2 = tat.Resample(
                orig_freq=self.model_set[-1]["sampling_rate"],
                new_freq=self.input_sampling_rate,
                dtype=torch.float32,
            ).to(self.device)
        else:
            self.resampler2 = None

        # Reset VAD state for new buffer configuration
        self.vad_cache = {}
        self.vad_chunk_size = min(500, 1000 * self.block_time)
        self.vad_speech_detected = False
        self.set_speech_detected_false_at_end_flag = False

    def _load_models(
        self,
    ) -> Tuple[
        Dict[str, Any],
        Callable[[torch.Tensor], torch.Tensor],
        Callable[[torch.Tensor], torch.Tensor],
        CAMPPlus,
        Callable[[torch.Tensor], torch.Tensor],
        Dict[str, Any],
    ]:
        """Load all models required for voice conversion.

        Returns:
            Tuple containing:
                - model: Dictionary of model components
                - semantic_fn: Function to extract semantic features
                - vocoder_fn: Vocoder function for waveform generation
                - campplus_model: CAMPPlus speaker embedding model
                - to_mel: Function to convert waveform to mel spectrogram
                - mel_fn_args: Arguments for mel spectrogram computation
        """
        if self.fp16:
            self.logger.info("ℹ️ Using fp16 precision for model inference.")
        if self.checkpoint_path is None or self.checkpoint_path == "":
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                "Plachta/Seed-VC",
                "DiT_uvit_tat_xlsr_ema.pth",
                "config_dit_mel_seed_uvit_xlsr_tiny.yml",
            )
        else:
            dit_checkpoint_path = self.checkpoint_path
            dit_config_path = self.config_path

        self.logger.debug("ℹ️ DiT checkpoint path: %s", dit_checkpoint_path)
        self.logger.debug("ℹ️ DiT config path: %s", dit_config_path)

        config = yaml.safe_load(open(dit_config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = "DiT"
        model = build_model(model_params, stage="DiT")
        sr = config["preprocess_params"]["sr"]

        # Load checkpoints
        with suppress_print_and_logging():
            model, _, _, _ = load_checkpoint(
                model,
                None,
                dit_checkpoint_path,
                load_only_params=True,
                ignore_modules=[],
                is_distributed=False,
            )
        for key in model:
            model[key].eval()
            model[key].to(self.device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # Load additional modules
        with suppress_print_and_logging():
            campplus_ckpt_path = load_custom_model_from_hf(
                "funasr/campplus",
                "campplus_cn_common.bin",
                config_filename=None,
            )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        campplus_model.eval()
        campplus_model.to(self.device)

        # Load vocoder
        vocoder_fn = self._load_vocoder(model_params)

        # Load speech tokenizer
        semantic_fn = self._load_speech_tokenizer(config, model_params)

        # Generate mel spectrograms
        mel_fn_args = {
            "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": config["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
            "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": sr,
            "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0),
            "fmax": None
            if config["preprocess_params"]["spect_params"].get("fmax", "None") == "None"
            else 8000,
            "center": False,
        }
        from seed_vc.modules.audio import mel_spectrogram

        to_mel: Callable[[torch.Tensor], torch.Tensor] = lambda x: mel_spectrogram(x, **mel_fn_args)

        return (
            model,
            semantic_fn,
            vocoder_fn,
            campplus_model,
            to_mel,
            mel_fn_args,
        )

    def _load_vocoder(self, model_params) -> Callable[[torch.Tensor], torch.Tensor]:
        """Load vocoder model based on configuration."""
        vocoder_type = model_params.vocoder.type

        if vocoder_type == "bigvgan":
            from seed_vc.modules.bigvgan import bigvgan

            bigvgan_name = model_params.vocoder.name
            bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
            # remove weight norm in the model and set to eval mode
            bigvgan_model.remove_weight_norm()
            bigvgan_model = bigvgan_model.eval().to(self.device)
            vocoder_fn = bigvgan_model
        elif vocoder_type == "hifigan":
            from seed_vc.modules.hifigan.f0_predictor import ConvRNNF0Predictor
            from seed_vc.modules.hifigan.generator import HiFTGenerator

            hift_config = yaml.safe_load(open("configs/hifigan.yml", "r"))
            hift_gen = HiFTGenerator(
                **hift_config["hift"],
                f0_predictor=ConvRNNF0Predictor(**hift_config["f0_predictor"]),
            )
            hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", "hift.pt", None)
            hift_gen.load_state_dict(torch.load(hift_path, map_location="cpu"))
            hift_gen.eval()
            hift_gen.to(self.device)
            vocoder_fn = hift_gen
        elif vocoder_type == "vocos":
            vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, "r"))
            vocos_path = model_params.vocoder.vocos.path
            vocos_model_params = recursive_munch(vocos_config["model_params"])
            vocos = build_model(vocos_model_params, stage="mel_vocos")
            vocos_checkpoint_path = vocos_path
            vocos, _, _, _ = load_checkpoint(
                vocos,
                None,
                vocos_checkpoint_path,
                load_only_params=True,
                ignore_modules=[],
                is_distributed=False,
            )
            _ = [vocos[key].eval().to(self.device) for key in vocos]
            _ = [vocos[key].to(self.device) for key in vocos]
            total_params = sum(
                sum(p.numel() for p in vocos[key].parameters() if p.requires_grad)
                for key in vocos.keys()
            )
            self.logger.debug("ℹ️ Vocoder model total parameters: %.2fM", total_params / 1_000_000)
            vocoder_fn = vocos.decoder
        else:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")

        return vocoder_fn

    def _load_speech_tokenizer(
        self, config: Dict[str, Any], model_params
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Load speech tokenizer model based on configuration."""
        speech_tokenizer_type = model_params.speech_tokenizer.type

        if speech_tokenizer_type == "whisper":
            # whisper
            from transformers import AutoFeatureExtractor, WhisperModel

            whisper_name = model_params.speech_tokenizer.name
            whisper_model = WhisperModel.from_pretrained(
                whisper_name, torch_dtype=torch.float16
            ).to(
                self.device,
            )
            del whisper_model.decoder
            whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

            def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
                """Extract semantic features using Whisper model."""
                ori_inputs = whisper_feature_extractor(
                    [waves_16k.squeeze(0).cpu().numpy()],
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                ori_input_features = whisper_model._mask_input_features(
                    ori_inputs.input_features,
                    attention_mask=ori_inputs.attention_mask,
                ).to(self.device)
                with torch.no_grad():
                    ori_outputs = whisper_model.encoder(
                        ori_input_features.to(whisper_model.encoder.dtype),
                        head_mask=None,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                S_ori = ori_outputs.last_hidden_state.to(torch.float32)
                S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
                return S_ori
        elif speech_tokenizer_type == "cnhubert":
            from transformers import HubertModel, Wav2Vec2FeatureExtractor

            hubert_model_name = config["model_params"]["speech_tokenizer"]["name"]
            hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
            hubert_model = HubertModel.from_pretrained(hubert_model_name)
            hubert_model = hubert_model.to(self.device)
            hubert_model = hubert_model.eval()
            hubert_model = hubert_model.half()

            def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
                """Extract semantic features using CNHubert model."""
                ori_waves_16k_input_list = [
                    waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))
                ]
                ori_inputs = hubert_feature_extractor(
                    ori_waves_16k_input_list,
                    return_tensors="pt",
                    return_attention_mask=True,
                    padding=True,
                    sampling_rate=16000,
                ).to(self.device)
                with torch.no_grad():
                    ori_outputs = hubert_model(
                        ori_inputs.input_values.half(),
                    )
                S_ori = ori_outputs.last_hidden_state.float()
                return S_ori
        elif speech_tokenizer_type == "xlsr":
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

            model_name = config["model_params"]["speech_tokenizer"]["name"]
            output_layer = config["model_params"]["speech_tokenizer"]["output_layer"]
            wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
            wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
            wav2vec_model = wav2vec_model.to(self.device)
            wav2vec_model = wav2vec_model.eval()
            wav2vec_model = wav2vec_model.half()

            def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
                """Extract semantic features using XLSR model."""
                ori_waves_16k_input_list = [
                    waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))
                ]
                ori_inputs = wav2vec_feature_extractor(
                    ori_waves_16k_input_list,
                    return_tensors="pt",
                    return_attention_mask=True,
                    padding=True,
                    sampling_rate=16000,
                ).to(self.device)
                with torch.no_grad():
                    ori_outputs = wav2vec_model(
                        ori_inputs.input_values.half(),
                    )
                S_ori = ori_outputs.last_hidden_state.float()
                return S_ori
        else:
            raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")

        return semantic_fn

    @torch.no_grad()
    def custom_infer(
        self,
        reference_wav: np.ndarray,
        new_reference_wav_name: str,
        input_wav_res: torch.Tensor,
        block_frame_16k: int,
        skip_head: int,
        skip_tail: int,
        return_length: int,
        diffusion_steps: int,
        inference_cfg_rate: float,
        max_prompt_length: float,
        cd_difference: float = 2.0,
    ) -> torch.Tensor:
        """Perform custom voice conversion inference.

        Args:
            reference_wav: Reference audio waveform as numpy array.
            new_reference_wav_name: Name of the new reference audio file.
            input_wav_res: Input waveform resampled to 16kHz.
            block_frame_16k: Number of frames in a block at 16kHz.
            skip_head: Number of frames to skip at the beginning.
            skip_tail: Number of frames to skip at the end.
            return_length: Expected length of output in frames.
            diffusion_steps: Number of diffusion steps for inference.
            inference_cfg_rate: Classifier-free guidance rate.
            max_prompt_length: Maximum prompt length in seconds.
            cd_difference: Content encoder-DiT time difference in seconds.

        Returns:
            Converted audio waveform as torch tensor.
        """
        (
            model,
            semantic_fn,
            vocoder_fn,
            campplus_model,
            to_mel,
            mel_fn_args,
        ) = self.model_set
        sr = mel_fn_args["sampling_rate"]
        hop_length = mel_fn_args["hop_size"]

        if self.ce_dit_difference != cd_difference:
            self.ce_dit_difference = cd_difference
            self.logger.info("ℹ️ Setting ce_dit_difference to %s seconds", cd_difference)

        if (
            self.prompt_condition is None
            or self.reference_wav_name != new_reference_wav_name
            or self.prompt_len != max_prompt_length
        ):
            self.prompt_len = max_prompt_length
            self.logger.info("ℹ️ Setting max prompt length to %s seconds", max_prompt_length)
            reference_wav = reference_wav[: int(sr * self.prompt_len)]
            reference_wav_tensor = torch.from_numpy(reference_wav).to(self.device)

            ori_waves_16k = torchaudio.functional.resample(reference_wav_tensor, sr, 16000)
            S_ori = semantic_fn(ori_waves_16k.unsqueeze(0))
            feat2 = torchaudio.compliance.kaldi.fbank(
                ori_waves_16k.unsqueeze(0),
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000,
            )
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            self.style2 = campplus_model(feat2.unsqueeze(0))

            self.mel2 = to_mel(reference_wav_tensor.unsqueeze(0))
            target2_lengths = torch.LongTensor([self.mel2.size(2)]).to(self.mel2.device)
            self.prompt_condition = model.length_regulator(
                S_ori,
                ylens=target2_lengths,
                n_quantizers=3,
                f0=None,
            )[0]

            self.reference_wav_name = new_reference_wav_name

        converted_waves_16k = input_wav_res
        start_event, end_event = self._create_timing_events()
        start_event.record()
        S_alt = semantic_fn(converted_waves_16k.unsqueeze(0))
        end_event.record()
        elapsed_time_ms = self._calculate_elapsed_time(start_event, end_event)
        self.logger.debug("ℹ️ Time taken for semantic_fn: %.3fms", elapsed_time_ms)

        ce_dit_frame_difference = int(self.ce_dit_difference * 50)
        S_alt = S_alt[:, ce_dit_frame_difference:]
        target_lengths = torch.LongTensor(
            [
                (skip_head + return_length + skip_tail - ce_dit_frame_difference)
                / 50
                * sr
                // hop_length
            ],
        ).to(S_alt.device)
        cond = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        cat_condition = torch.cat([self.prompt_condition, cond], dim=1)

        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16 if self.fp16 else torch.float32
        ):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(self.mel2.device),
                self.mel2,
                self.style2,
                None,
                n_timesteps=diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, self.mel2.size(-1) :]
            vc_wave = vocoder_fn(vc_target).squeeze()

        output_len = return_length * sr // 50
        tail_len = skip_tail * sr // 50
        output = vc_wave[-output_len - tail_len : -tail_len]

        return output

    def audio_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Audio block callback function for real-time processing.

        Args:
            indata: Input audio data array.
            outdata: Output audio data array to fill.
            frames: Number of frames.
            time: Timing information.
            status: Callback status flags.
        """
        import time as time_module

        # Start timing for RTF calculation
        start_time = time_module.time()

        indata = librosa.to_mono(indata.T)

        # Calculate audio duration for RTF
        audio_duration_sec = len(indata) / self.input_sampling_rate

        # Check if indata size matches expected block_frame
        if indata.shape[0] != self.block_frame:
            self.logger.warning(
                "Expected block_frame=%s, got indata.shape[0]=%s", self.block_frame, indata.shape[0]
            )

        # VAD first
        start_event, end_event = self._create_timing_events()
        start_event.record()

        if self.use_vad:
            indata_16k = librosa.resample(
                indata,
                orig_sr=self.input_sampling_rate,
                target_sr=16000,
            )
            res = self.vad_model.generate(
                input=indata_16k,
                cache=self.vad_cache,
                is_final=False,
                chunk_size=self.vad_chunk_size,
            )
            res_value = res[0]["value"]
            if len(res_value) % 2 == 1 and not self.vad_speech_detected:
                self.vad_speech_detected = True
            elif len(res_value) % 2 == 1 and self.vad_speech_detected:
                self.set_speech_detected_false_at_end_flag = True
        else:
            # If VAD is not used, assume speech is always detected
            self.vad_speech_detected = True
            self.set_speech_detected_false_at_end_flag = True
        end_event.record()
        elapsed_time_ms = self._calculate_elapsed_time(start_event, end_event)
        self.logger.debug("ℹ️ Time taken for VAD: %.3fms", elapsed_time_ms)

        # Update input buffers
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()

        # Handle case where indata size doesn't match block_frame
        if indata.shape[0] <= self.block_frame:
            # If indata is smaller than or equal to block_frame, pad with zeros if needed
            self.input_wav[-self.block_frame :] = torch.zeros(self.block_frame, device=self.device)
            self.input_wav[-self.block_frame :][: indata.shape[0]] = torch.from_numpy(indata).to(
                self.device
            )
        else:
            # If indata is larger than block_frame, truncate it
            self.logger.warning(
                "Truncating indata from %s to %s", indata.shape[0], self.block_frame
            )
            self.input_wav[-self.block_frame :] = torch.from_numpy(indata[: self.block_frame]).to(
                self.device
            )

        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()

        # Calculate the actual size to use (either indata size or block_frame)
        actual_indata_size = min(indata.shape[0], self.block_frame)

        # Safe resampling approach
        try:
            # Extract audio segment for resampling, ensuring we have enough context
            extract_size = min(actual_indata_size + 2 * self.zc, self.input_wav.shape[0])
            audio_segment = self.input_wav[-extract_size:].cpu().numpy()

            # Resample to 16kHz
            resampled = librosa.resample(
                audio_segment,
                orig_sr=self.input_sampling_rate,
                target_sr=16000,
            )

            # Skip first 320 samples and take what we need
            if len(resampled) > 320:
                resampled_chunk = resampled[320:]
                target_size = 320 * (actual_indata_size // self.zc + 1)

                # Ensure we don't exceed buffer size
                target_size = min(target_size, self.block_frame_16k)

                if len(resampled_chunk) >= target_size:
                    self.input_wav_res[-target_size:] = torch.from_numpy(
                        resampled_chunk[:target_size]
                    )
                else:
                    # Pad with zeros if resampled chunk is too short
                    self.input_wav_res[-target_size:] = 0
                    self.input_wav_res[-len(resampled_chunk) :] = torch.from_numpy(resampled_chunk)
            else:
                # If resampled audio is too short, just zero out the target area
                target_size = min(320 * (actual_indata_size // self.zc + 1), self.block_frame_16k)
                self.input_wav_res[-target_size:] = 0

        except Exception as e:
            self.logger.warning("Error in resampling, using zeros: %s", e)
            # Fallback: zero out the resampled buffer area
            target_size = min(320 * (actual_indata_size // self.zc + 1), self.block_frame_16k)
            self.input_wav_res[-target_size:] = 0

        # Process based on conversion mode
        if self.conversion_mode == ConversionMode.SILENCE:
            # Return silence
            infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])
        elif self.conversion_mode == ConversionMode.PASSTHROUGH:
            # Return original input audio
            infer_wav = self.input_wav[self.extra_frame :]
        else:  # ConversionMode.CONVERT
            # Voice conversion inference
            if self.extra_time_ce - self.extra_time < 0:
                raise ValueError(
                    "Content encoder extra context must be greater than DiT extra context!",
                )

            start_event, end_event = self._create_timing_events()
            start_event.record()
            infer_wav = self.custom_infer(
                self.reference_wav,
                self.reference_wav_path,
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.skip_tail,
                self.return_length,
                int(self.diffusion_steps),
                self.inference_cfg_rate,
                self.max_prompt_length,
                self.extra_time_ce - self.extra_time,
            )

            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)

            end_event.record()
            elapsed_time_ms = self._calculate_elapsed_time(start_event, end_event)
            self.logger.debug("ℹ️ Time taken for VC: %.3fms", elapsed_time_ms)

            if not self.vad_speech_detected:
                infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])

        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]

        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.device),
            )
            + 1e-8,
        )

        tensor = cor_nom[0, 0] / cor_den[0, 0]
        if tensor.numel() > 1:  # If tensor has multiple elements
            if sys.platform == "darwin":
                _, sola_offset = torch.max(tensor, dim=0)
                sola_offset = sola_offset.item()
            else:
                sola_offset = torch.argmax(tensor, dim=0).item()
        else:
            sola_offset = tensor.item()

        # Post-process
        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]
        outdata_tensor = infer_wav[: self.block_frame].repeat(self.channels, 1).t().cpu().numpy()

        self.output_wav.append(outdata_tensor.copy())

        # Fill outdata if provided
        if outdata is not None:
            # Handle size mismatch between outdata and outdata_tensor
            if outdata.shape[0] == outdata_tensor.shape[0]:
                outdata[:] = outdata_tensor
            elif outdata.shape[0] > outdata_tensor.shape[0]:
                # outdata is larger, pad with zeros
                outdata[: outdata_tensor.shape[0]] = outdata_tensor
                outdata[outdata_tensor.shape[0] :] = 0
            else:
                # outdata is smaller, truncate
                outdata[:] = outdata_tensor[: outdata.shape[0]]
        if self.set_speech_detected_false_at_end_flag:
            self.vad_speech_detected = False
            self.set_speech_detected_false_at_end_flag = False

        # Calculate and log RTF
        end_time = time_module.time()
        processing_time_sec = end_time - start_time
        rtf = processing_time_sec / audio_duration_sec
        self.logger.debug(
            "ℹ️ RTF: %.3f (Processing: %.3fs, Audio: %.3fs)",
            rtf,
            processing_time_sec,
            audio_duration_sec,
        )
