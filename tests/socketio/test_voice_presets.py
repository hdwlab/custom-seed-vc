"""Tests for SeedVC voice preset loading and vector sampling."""

import numpy as np
import pytest

from seed_vc.socketio.voice_presets import VoicePresetError, VoicePresetStore, vector_space_path


def make_presets_dir(tmp_path, with_space=True):
    """Create a minimal presets directory."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "p1.wav").write_bytes(b"wav")
    (audio_dir / "p2.wav").write_bytes(b"wav")
    (tmp_path / "presets.yaml").write_text(
        """
sampling:
  k: 2
  eps: 0.5
presets:
  - id: p1
    gender: female
    audio: audio/p1.wav
  - id: p2
    audio: audio/p2.wav
"""
    )
    if with_space:
        space_path = vector_space_path(tmp_path)
        space_path.parent.mkdir()
        np.savez(space_path, vectors=np.eye(2, 3, dtype=np.float32))
    return tmp_path


def test_store_loads_presets_and_seed_vc_vector_space(tmp_path):
    """VoicePresetStore loads presets.yaml and spaces/seed_vc.npz."""
    presets_dir = make_presets_dir(tmp_path)

    store = VoicePresetStore(str(presets_dir))

    assert [p.preset_id for p in store.list_presets()] == ["p1", "p2"]
    assert store.has_vector_space is True
    assert store.vector_dim == 3


def test_store_without_vector_space_is_allowed(tmp_path):
    """A presets directory can be used without vector sampling."""
    presets_dir = make_presets_dir(tmp_path, with_space=False)

    store = VoicePresetStore(str(presets_dir))

    assert store.has_vector_space is False
    assert store.vector_dim is None


def test_resolve_preset_filters_gender(tmp_path):
    """Gender filtering uses only matching preset entries."""
    store = VoicePresetStore(str(make_presets_dir(tmp_path, with_space=False)))

    preset = store.resolve_preset("alice", "female")

    assert preset.preset_id == "p1"


def test_invalid_gender_raises(tmp_path):
    """Invalid gender requests fail clearly."""
    store = VoicePresetStore(str(make_presets_dir(tmp_path, with_space=False)))

    with pytest.raises(VoicePresetError, match="gender"):
        store.resolve_preset("alice", "unknown")


def test_sample_vector_is_deterministic(tmp_path):
    """Vector sampling is deterministic for the same operator ID."""
    store = VoicePresetStore(str(make_presets_dir(tmp_path)))
    anchor = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    first = store.sample_vector(anchor, "alice")
    second = store.sample_vector(anchor, "alice")

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
