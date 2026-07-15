# Copyright (C) 2026 Human Dataware Lab.
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

"""Voice preset loading and deterministic SeedVC vector sampling."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_VALID_GENDERS = ("male", "female")
_DEFAULT_SAMPLING_K = 8
_DEFAULT_SAMPLING_EPS = 0.2
_ENGINE_NAME = "seed-vc"


class VoicePresetError(ValueError):
    """Raised when preset data is invalid or no preset can be resolved."""


@dataclass(frozen=True)
class VoicePreset:
    """A single voice preset entry."""

    preset_id: str
    gender: Optional[str]
    audio_path: str


def vector_space_path(presets_dir: str | Path, engine_name: str = _ENGINE_NAME) -> Path:
    """Return the conventional vector space path for an engine."""
    return Path(presets_dir) / "spaces" / f"{engine_name.replace('-', '_')}.npz"


def apply_preset_voice(
    engine,
    store: "VoicePresetStore",
    preset: VoicePreset,
    operator_id: str,
    eps: Optional[float] = None,
    farthest: bool = False,
) -> None:
    """Apply a preset voice and optional operator-specific sampled vector."""
    engine.update_reference_audio(preset.audio_path)
    if store.has_vector_space:
        try:
            anchor = engine.get_speaker_vector()
            engine.update_speaker_vector(
                store.sample_vector(anchor, operator_id, eps=eps, farthest=farthest)
            )
        except NotImplementedError:
            logger.warning("Engine does not support speaker vectors; using reference audio only")


class VoicePresetStore:
    """Load voice presets and resolve operator-specific voices."""

    def __init__(self, presets_dir: str, engine_name: str = _ENGINE_NAME) -> None:
        """Load and validate a presets directory."""
        self._presets_dir = Path(presets_dir)
        manifest_path = self._presets_dir / "presets.yaml"
        if not manifest_path.is_file():
            raise VoicePresetError(f"Manifest not found: {manifest_path} (presets.yaml)")
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f) or {}

        sampling = manifest.get("sampling") or {}
        try:
            self._k = int(sampling.get("k", _DEFAULT_SAMPLING_K))
            self._eps = float(sampling.get("eps", _DEFAULT_SAMPLING_EPS))
        except (TypeError, ValueError) as exc:
            raise VoicePresetError(f"Invalid sampling config: {sampling}") from exc
        if self._k < 1:
            raise VoicePresetError(f"sampling.k must be >= 1: {self._k}")
        if not 0.0 <= self._eps <= 1.0:
            raise VoicePresetError(f"sampling.eps must be within [0, 1]: {self._eps}")

        self._presets = self._load_presets(manifest)
        self._vectors = self._load_vector_space(engine_name)
        self._vector_norms = (
            np.linalg.norm(self._vectors, axis=1)
            if self._vectors is not None
            else np.zeros(0, dtype=np.float32)
        )

    def _load_presets(self, manifest: dict) -> list[VoicePreset]:
        entries = manifest.get("presets") or []
        if not entries:
            raise VoicePresetError("Manifest contains no presets")
        presets = []
        seen_ids: set[str] = set()
        for entry in entries:
            preset_id = str(entry.get("id", ""))
            gender = _optional_manifest_gender(entry.get("gender"), preset_id)
            audio = str(entry.get("audio", ""))
            if not preset_id or not audio:
                raise VoicePresetError(f"Preset entry must have id and audio: {entry}")
            if preset_id in seen_ids:
                raise VoicePresetError(f"Duplicate preset id: {preset_id}")
            seen_ids.add(preset_id)
            audio_path = (self._presets_dir / audio).resolve()
            try:
                audio_path.relative_to(self._presets_dir.resolve())
            except ValueError:
                raise VoicePresetError(
                    f"Audio path for preset {preset_id} must stay inside the presets "
                    f"directory: {audio}"
                ) from None
            if not audio_path.is_file():
                raise VoicePresetError(f"Audio file for preset {preset_id} not found: {audio_path}")
            presets.append(
                VoicePreset(preset_id=preset_id, gender=gender, audio_path=str(audio_path))
            )
        return presets

    def _load_vector_space(self, engine_name: str) -> Optional[np.ndarray]:
        space_path = vector_space_path(self._presets_dir, engine_name)
        if not space_path.is_file():
            return None
        with np.load(space_path) as data:
            if "vectors" not in data:
                raise VoicePresetError(f"Vector space file has no 'vectors' key: {space_path}")
            vectors = np.asarray(data["vectors"], dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) == 0:
            raise VoicePresetError(f"Vector space must be a non-empty 2-D array: {space_path}")
        return vectors

    @property
    def has_vector_space(self) -> bool:
        """Whether a SeedVC vector space is loaded."""
        return self._vectors is not None

    @property
    def vector_dim(self) -> Optional[int]:
        """Dimension of the loaded vector space, or None when absent."""
        return None if self._vectors is None else int(self._vectors.shape[1])

    def list_presets(self) -> list[VoicePreset]:
        """Return all presets in manifest order."""
        return list(self._presets)

    def resolve_preset(
        self,
        operator_id: str,
        gender: Optional[str],
        preset_id: Optional[str] = None,
    ) -> VoicePreset:
        """Resolve a deterministic preset for an operator."""
        if preset_id is not None:
            for preset in self._presets:
                if preset.preset_id == preset_id:
                    return preset
            raise VoicePresetError(f"Unknown preset_id: {preset_id}")
        if gender is None:
            pool = self._presets
        elif gender not in _VALID_GENDERS:
            raise VoicePresetError(f"gender must be one of {_VALID_GENDERS} when given: {gender!r}")
        else:
            pool = [p for p in self._presets if p.gender == gender]
        if not pool:
            raise VoicePresetError(f"No preset available for gender: {gender}")
        return max(
            pool,
            key=lambda p: hashlib.sha256(f"{operator_id}:{p.preset_id}".encode()).digest(),
        )

    def cosine_similarities(self, anchor: np.ndarray) -> np.ndarray:
        """Return cosine similarity between anchor and every loaded vector."""
        if self._vectors is None:
            raise VoicePresetError("No speaker vector space is loaded for this engine")
        anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
        return (self._vectors @ anchor) / (self._vector_norms * np.linalg.norm(anchor) + 1e-8)

    def sample_vector(
        self,
        anchor: np.ndarray,
        operator_id: str,
        eps: Optional[float] = None,
        farthest: bool = False,
    ) -> np.ndarray:
        """Sample a deterministic vector near the anchor within the vector space."""
        if self._vectors is None:
            raise VoicePresetError("No speaker vector space is loaded for this engine")
        if eps is None:
            eps = self._eps
        anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
        seed = int.from_bytes(hashlib.sha256(operator_id.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        vectors = self._vectors
        k = min(self._k, len(vectors))
        ordered = np.argsort(self.cosine_similarities(anchor))
        neighbor_idx = ordered[:k] if farthest else ordered[-k:]
        weights = rng.dirichlet(np.ones(k)).astype(np.float32)
        mix = weights @ vectors[neighbor_idx]
        sampled = (1.0 - eps) * anchor + eps * mix
        scale = np.linalg.norm(anchor) / (np.linalg.norm(sampled) + 1e-8)
        return (sampled * scale).astype(np.float32)


def _optional_manifest_gender(value: object, preset_id: str) -> Optional[str]:
    if value is None:
        return None
    gender = str(value).strip()
    if not gender:
        return None
    if gender not in _VALID_GENDERS:
        raise VoicePresetError(
            f"Preset {preset_id} has invalid gender: {gender!r} "
            f"(must be omitted or one of {_VALID_GENDERS})"
        )
    return gender
