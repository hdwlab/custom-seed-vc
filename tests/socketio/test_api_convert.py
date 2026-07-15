"""Tests for offline file conversion API endpoints."""

import io
import json
import threading

import numpy as np
import pytest
import soundfile as sf
import yaml
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient
from pydantic import ValidationError

from seed_vc.socketio.api import APIRouterVCModel
from seed_vc.socketio.model import VoiceConverter
from seed_vc.socketio.offline_jobs import OfflineJobManager
from seed_vc.socketio.runtime import ServerRuntimeCoordinator
from seed_vc.socketio.schemas import BatchFileConversionRequest, FileConversionRequest
from seed_vc.socketio.voice_presets import VoicePresetStore

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
    buf = io.BytesIO()
    sf.write(
        buf, np.zeros(int(SAMPLE_RATE * duration_sec), dtype=np.float32), SAMPLE_RATE, format="WAV"
    )
    return buf.getvalue()


def make_voice_store(tmp_path, bundle_id=None) -> VoicePresetStore:
    """Create a valid reference-only preset store for offline API tests."""
    presets_dir = tmp_path / "presets"
    audio_dir = presets_dir / "audio"
    audio_dir.mkdir(parents=True)
    sf.write(audio_dir / "female_001.wav", np.zeros(16000, dtype=np.float32), 16000)
    manifest = {
        "presets": [
            {
                "id": "female_001",
                "gender": "female",
                "audio": "audio/female_001.wav",
            }
        ]
    }
    if bundle_id is not None:
        manifest["schema_version"] = 1
        manifest["provenance"] = {"bundle_id": bundle_id}
    with open(presets_dir / "presets.yaml", "w") as f:
        yaml.safe_dump(manifest, f)
    return VoicePresetStore(str(presets_dir))


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

    def test_operator_voice_is_applied_and_restored(
        self, voice_converter, allowed_dir, monkeypatch
    ):
        """JSON conversion uses the realtime preset selection behavior."""
        previous_path = voice_converter.reference_wav_path
        previous_wav = voice_converter.reference_wav
        reference_during_conversion = None
        original_convert = voice_converter.convert_file

        def tracking_convert(input_path, output_path):
            nonlocal reference_during_conversion
            reference_during_conversion = voice_converter.reference_wav_path
            return original_convert(input_path, output_path)

        monkeypatch.setattr(voice_converter, "convert_file", tracking_convert)
        api_router = APIRouterVCModel(
            model=voice_converter,
            allowed_audio_dirs=[str(allowed_dir)],
            voice_store=make_voice_store(allowed_dir, bundle_id="bundle-123"),
        )
        response = api_router.convert_file(
            FileConversionRequest(
                input_path=str(allowed_dir / "input.wav"),
                output_path=str(allowed_dir / "output.wav"),
                operator_id="alice",
                gender="female",
            )
        )

        assert response.status_code == 200
        assert json.loads(bytes(response.body))["preset_id"] == "female_001"
        assert json.loads(bytes(response.body))["preset_bundle_id"] == "bundle-123"
        assert reference_during_conversion is not None
        assert reference_during_conversion.endswith("female_001.wav")
        assert voice_converter.reference_wav_path == previous_path
        assert voice_converter.reference_wav is previous_wav

    def test_operator_voice_requires_configured_presets(self, api_router, allowed_dir):
        """Operator selection returns 400 when the server has no preset store."""
        request = FileConversionRequest(
            input_path=str(allowed_dir / "input.wav"),
            output_path=str(allowed_dir / "output.wav"),
            operator_id="alice",
        )

        with pytest.raises(HTTPException) as exc_info:
            api_router.convert_file(request)

        assert exc_info.value.status_code == 400
        assert "Voice presets are not configured" in str(exc_info.value.detail)

    @pytest.mark.parametrize("field", ["gender", "preset_id"])
    def test_operator_id_is_required_for_voice_fields(self, allowed_dir, field):
        """Partial JSON operator voice requests are rejected by the schema."""
        with pytest.raises(ValidationError, match="operator_id is required"):
            FileConversionRequest(
                input_path=str(allowed_dir / "input.wav"),
                output_path=str(allowed_dir / "output.wav"),
                **{field: "female" if field == "gender" else "female_001"},
            )


class TestBatchFileConversionRequest:
    """Tests for asynchronous file conversion batch validation."""

    def test_accepts_non_empty_batch(self):
        """A batch with one item is accepted."""
        request = BatchFileConversionRequest(
            items=[FileConversionRequest(input_path="/in.wav", output_path="/out.wav")]
        )

        assert len(request.items) == 1

    @pytest.mark.parametrize("item_count", [0, 101])
    def test_rejects_out_of_bounds_batch_size(self, item_count):
        """One job must contain between one and one hundred items."""
        with pytest.raises(ValidationError):
            BatchFileConversionRequest(
                items=[
                    FileConversionRequest(
                        input_path=f"/in-{index}.wav",
                        output_path=f"/out-{index}.wav",
                    )
                    for index in range(item_count)
                ]
            )

    def test_rejects_duplicate_literal_output_paths(self):
        """Two items cannot name the same output path."""
        with pytest.raises(ValidationError, match="output_path values must be unique"):
            BatchFileConversionRequest(
                items=[
                    FileConversionRequest(input_path="/in-a.wav", output_path="/out.wav"),
                    FileConversionRequest(input_path="/in-b.wav", output_path="/out.wav"),
                ]
            )


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

    def test_operator_voice_returns_selected_preset_header(
        self, voice_converter, allowed_dir, monkeypatch
    ):
        """Multipart conversion applies operator voice and reports the selected preset."""
        previous_path = voice_converter.reference_wav_path
        previous_wav = voice_converter.reference_wav
        reference_during_conversion = None
        original_convert = voice_converter.convert_file

        def tracking_convert(input_path, output_path):
            nonlocal reference_during_conversion
            reference_during_conversion = voice_converter.reference_wav_path
            return original_convert(input_path, output_path)

        monkeypatch.setattr(voice_converter, "convert_file", tracking_convert)
        api_router = APIRouterVCModel(
            model=voice_converter,
            allowed_audio_dirs=[str(allowed_dir)],
            voice_store=make_voice_store(allowed_dir, bundle_id="bundle-123"),
        )
        response = api_router.convert_file_upload(
            input_file=UploadFile(filename="input.wav", file=io.BytesIO(make_wav_bytes())),
            operator_id="alice",
            gender="female",
        )

        assert response.status_code == 200
        assert response.headers["X-Voice-Preset-ID"] == "female_001"
        assert response.headers["X-Voice-Preset-Bundle-ID"] == "bundle-123"
        assert reference_during_conversion is not None
        assert reference_during_conversion.endswith("female_001.wav")
        assert voice_converter.reference_wav_path == previous_path
        assert voice_converter.reference_wav is previous_wav

    def test_reference_upload_and_operator_voice_are_mutually_exclusive(self, api_router):
        """A multipart request cannot select two temporary target voices."""
        with pytest.raises(HTTPException) as exc_info:
            api_router.convert_file_upload(
                input_file=UploadFile(filename="input.wav", file=io.BytesIO(make_wav_bytes())),
                reference_file=UploadFile(
                    filename="reference.wav", file=io.BytesIO(make_wav_bytes())
                ),
                operator_id="alice",
            )

        assert exc_info.value.status_code == 400
        assert "cannot be used together" in str(exc_info.value.detail)


class TestReloadVoicePresets:
    """Tests for validated in-place preset bundle reloads."""

    def test_reloads_shared_store(self, voice_converter, allowed_dir):
        """A valid replacement is exposed without replacing the shared store object."""
        store = make_voice_store(allowed_dir, bundle_id="bundle-old")
        router = APIRouterVCModel(model=voice_converter, voice_store=store)
        manifest_path = allowed_dir / "presets" / "presets.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        manifest["provenance"]["bundle_id"] = "bundle-new"
        manifest_path.write_text(yaml.safe_dump(manifest))

        response = router.reload_voice_presets()
        body = json.loads(bytes(response.body))

        assert body["previous_bundle_id"] == "bundle-old"
        assert body["bundle_id"] == "bundle-new"
        assert body["preset_count"] == 1
        assert store.bundle_id == "bundle-new"

    def test_invalid_reload_returns_400_and_keeps_old_store(self, voice_converter, allowed_dir):
        """Invalid replacement data is rejected atomically."""
        store = make_voice_store(allowed_dir, bundle_id="bundle-old")
        router = APIRouterVCModel(model=voice_converter, voice_store=store)
        manifest_path = allowed_dir / "presets" / "presets.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        manifest["presets"][0]["sha256"] = "0" * 64
        manifest_path.write_text(yaml.safe_dump(manifest))

        with pytest.raises(HTTPException) as exc_info:
            router.reload_voice_presets()

        assert exc_info.value.status_code == 400
        assert store.bundle_id == "bundle-old"
        assert len(store.list_presets()) == 1

    def test_reload_rejects_while_realtime_client_is_connected(self, voice_converter, allowed_dir):
        """Reload uses runtime exclusion so no client observes a partial update."""
        store = make_voice_store(allowed_dir)
        runtime = ServerRuntimeCoordinator()
        assert runtime.try_register_client("sid-1", 1) is None
        router = APIRouterVCModel(
            model=voice_converter,
            runtime=runtime,
            voice_store=store,
        )

        with pytest.raises(HTTPException) as exc_info:
            router.reload_voice_presets()

        assert exc_info.value.status_code == 409


class TestConversionJobs:
    """Tests for asynchronous file-path conversion batches."""

    def test_submit_and_get_successful_batch(self, voice_converter, allowed_dir):
        """A submitted batch runs sequentially and exposes its completed result."""
        runtime = ServerRuntimeCoordinator()
        router = APIRouterVCModel(
            model=voice_converter,
            runtime=runtime,
            allowed_audio_dirs=[str(allowed_dir)],
        )
        manager = OfflineJobManager(runtime, voice_converter)
        router.attach_offline_job_manager(manager)
        request = BatchFileConversionRequest(
            items=[
                FileConversionRequest(
                    input_path=str(allowed_dir / "input.wav"),
                    output_path=str(allowed_dir / f"output-{index}.wav"),
                )
                for index in range(2)
            ]
        )

        try:
            response = router.submit_conversion_job(request)
            submitted = json.loads(bytes(response.body))
            job = manager.wait(submitted["id"], timeout=5)
            status = json.loads(bytes(router.get_conversion_job(submitted["id"]).body))

            assert response.status_code == 202
            assert submitted["status"] == "queued"
            assert submitted["item_count"] == 2
            assert submitted["status_url"].endswith(submitted["id"])
            assert job is not None
            assert job["status"] == "succeeded"
            assert status["result"]["succeeded"] == 2
            assert status["result"]["failed"] == 0
            assert [item["index"] for item in status["result"]["items"]] == [0, 1]
            assert (allowed_dir / "output-0.wav").is_file()
            assert (allowed_dir / "output-1.wav").is_file()
        finally:
            manager.shutdown()

    def test_batch_continues_after_an_item_fails(self, voice_converter, allowed_dir, monkeypatch):
        """One conversion error is retained per item without aborting later work."""
        original_convert = voice_converter.convert_file

        def partially_failing_convert(input_path, output_path):
            if output_path.endswith("failed-output.wav"):
                raise RuntimeError("bad input audio")
            return original_convert(input_path, output_path)

        monkeypatch.setattr(voice_converter, "convert_file", partially_failing_convert)
        runtime = ServerRuntimeCoordinator()
        router = APIRouterVCModel(
            model=voice_converter,
            runtime=runtime,
            allowed_audio_dirs=[str(allowed_dir)],
        )
        manager = OfflineJobManager(runtime, voice_converter)
        router.attach_offline_job_manager(manager)
        request = BatchFileConversionRequest(
            items=[
                FileConversionRequest(
                    input_path=str(allowed_dir / "input.wav"),
                    output_path=str(allowed_dir / "failed-output.wav"),
                ),
                FileConversionRequest(
                    input_path=str(allowed_dir / "input.wav"),
                    output_path=str(allowed_dir / "successful-output.wav"),
                ),
            ]
        )

        try:
            submitted = json.loads(bytes(router.submit_conversion_job(request).body))
            job = manager.wait(submitted["id"], timeout=5)

            assert job is not None
            assert job["status"] == "succeeded"
            assert job["result"]["succeeded"] == 1
            assert job["result"]["failed"] == 1
            assert job["result"]["items"][0]["error"] == "bad input audio"
            assert job["result"]["items"][1]["status"] == "succeeded"
        finally:
            manager.shutdown()

    def test_batch_applies_operator_voice_and_restores_model(
        self, voice_converter, allowed_dir, monkeypatch
    ):
        """Queued items retain synchronous conversion's temporary preset behavior."""
        previous_path = voice_converter.reference_wav_path
        previous_wav = voice_converter.reference_wav
        reference_during_conversion = None
        original_convert = voice_converter.convert_file

        def tracking_convert(input_path, output_path):
            nonlocal reference_during_conversion
            reference_during_conversion = voice_converter.reference_wav_path
            return original_convert(input_path, output_path)

        monkeypatch.setattr(voice_converter, "convert_file", tracking_convert)
        runtime = ServerRuntimeCoordinator()
        router = APIRouterVCModel(
            model=voice_converter,
            runtime=runtime,
            allowed_audio_dirs=[str(allowed_dir)],
            voice_store=make_voice_store(allowed_dir, bundle_id="bundle-123"),
        )
        manager = OfflineJobManager(runtime, voice_converter)
        router.attach_offline_job_manager(manager)
        request = BatchFileConversionRequest(
            items=[
                FileConversionRequest(
                    input_path=str(allowed_dir / "input.wav"),
                    output_path=str(allowed_dir / "output.wav"),
                    operator_id="alice",
                    gender="female",
                )
            ]
        )

        try:
            submitted = json.loads(bytes(router.submit_conversion_job(request).body))
            job = manager.wait(submitted["id"], timeout=5)

            assert job is not None
            item = job["result"]["items"][0]
            assert item["preset_id"] == "female_001"
            assert item["preset_bundle_id"] == "bundle-123"
            assert reference_during_conversion is not None
            assert reference_during_conversion.endswith("female_001.wav")
            assert voice_converter.reference_wav_path == previous_path
            assert voice_converter.reference_wav is previous_wav
        finally:
            manager.shutdown()

    def test_rejects_resolved_duplicate_output_paths(self, voice_converter, allowed_dir):
        """Path aliases cannot bypass the batch output uniqueness rule."""
        (allowed_dir / "sub").mkdir()
        runtime = ServerRuntimeCoordinator()
        router = APIRouterVCModel(
            model=voice_converter,
            runtime=runtime,
            allowed_audio_dirs=[str(allowed_dir)],
        )
        manager = OfflineJobManager(runtime, voice_converter)
        router.attach_offline_job_manager(manager)
        request = BatchFileConversionRequest(
            items=[
                FileConversionRequest(
                    input_path=str(allowed_dir / "input.wav"),
                    output_path=str(allowed_dir / "output.wav"),
                ),
                FileConversionRequest(
                    input_path=str(allowed_dir / "input.wav"),
                    output_path=str(allowed_dir / "sub" / ".." / "output.wav"),
                ),
            ]
        )

        try:
            with pytest.raises(HTTPException) as exc_info:
                router.submit_conversion_job(request)
            assert exc_info.value.status_code == 400
            assert "must be unique" in str(exc_info.value.detail)
        finally:
            manager.shutdown()

    def test_rejects_submission_while_realtime_client_is_connected(
        self, voice_converter, allowed_dir
    ):
        """The queue uses the same realtime/offline exclusion as synchronous conversion."""
        runtime = ServerRuntimeCoordinator()
        assert runtime.try_register_client("sid-1", max_clients=1) is None
        router = APIRouterVCModel(
            model=voice_converter,
            runtime=runtime,
            allowed_audio_dirs=[str(allowed_dir)],
        )
        manager = OfflineJobManager(runtime, voice_converter)
        router.attach_offline_job_manager(manager)
        request = BatchFileConversionRequest(
            items=[
                FileConversionRequest(
                    input_path=str(allowed_dir / "input.wav"),
                    output_path=str(allowed_dir / "output.wav"),
                )
            ]
        )

        try:
            with pytest.raises(HTTPException) as exc_info:
                router.submit_conversion_job(request)
            assert exc_info.value.status_code == 409
        finally:
            manager.shutdown()

    def test_rejects_submission_when_pending_queue_is_full(self, voice_converter, allowed_dir):
        """A full bounded queue is reported as HTTP 429."""
        runtime = ServerRuntimeCoordinator()
        router = APIRouterVCModel(
            model=voice_converter,
            runtime=runtime,
            allowed_audio_dirs=[str(allowed_dir)],
        )
        manager = OfflineJobManager(runtime, voice_converter, max_pending_jobs=1)
        router.attach_offline_job_manager(manager)
        started = threading.Event()
        release = threading.Event()

        def blocking_operation():
            started.set()
            assert release.wait(timeout=5)
            return {"message": "ok"}

        request = BatchFileConversionRequest(
            items=[
                FileConversionRequest(
                    input_path=str(allowed_dir / "input.wav"),
                    output_path=str(allowed_dir / "output.wav"),
                )
            ]
        )

        try:
            existing_job_id = manager.submit(blocking_operation)
            assert started.wait(timeout=5)
            with pytest.raises(HTTPException) as exc_info:
                router.submit_conversion_job(request)
            assert exc_info.value.status_code == 429
            release.set()
            manager.wait(existing_job_id, timeout=5)
        finally:
            release.set()
            manager.shutdown()

    def test_missing_queue_and_unknown_job_return_clear_errors(self, voice_converter):
        """Unavailable queue state is distinguished from an unknown retained job."""
        router = APIRouterVCModel(model=voice_converter)
        with pytest.raises(HTTPException) as exc_info:
            router.get_conversion_job("missing")
        assert exc_info.value.status_code == 503

        runtime = ServerRuntimeCoordinator()
        manager = OfflineJobManager(runtime, voice_converter)
        router.attach_offline_job_manager(manager)
        try:
            with pytest.raises(HTTPException) as exc_info:
                router.get_conversion_job("missing")
            assert exc_info.value.status_code == 404
        finally:
            manager.shutdown()


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
