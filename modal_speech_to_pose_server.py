import base64
import math
import os
import struct
import tempfile
import time
from typing import Optional

import modal


APP_NAME = "speech-to-pose-server"
SIGN_MT_ENDPOINT = "https://us-central1-sign-mt.cloudfunctions.net/spoken_text_to_signed_pose"

def parse_bool_env(value: str) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


ASR_MODEL = os.environ.get("ASR_MODEL", "tiny.en")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "en")
ASR_BEAM_SIZE = int(os.environ.get("ASR_BEAM_SIZE", "1"))
ASR_CPU_THREADS = int(os.environ.get("ASR_CPU_THREADS", "0"))
ASR_VAD_FILTER = parse_bool_env(os.environ.get("ASR_VAD_FILTER", "0"))
ASR_VAD_MIN_SILENCE_MS = int(os.environ.get("ASR_VAD_MIN_SILENCE_MS", "250"))
ASR_GPU = os.environ.get("ASR_GPU", "").strip()
ASR_DEVICE = "cuda" if ASR_GPU else "cpu"
ASR_COMPUTE_TYPE = os.environ.get(
    "ASR_COMPUTE_TYPE",
    "float16" if ASR_GPU else "int8",
)
SIGNED_LANGUAGE = os.environ.get("SIGNED_LANGUAGE", "ase")
SPOKEN_LANGUAGE = os.environ.get("SPOKEN_LANGUAGE", "en")
SIGN_MT_TIMEOUT = float(os.environ.get("SIGN_MT_TIMEOUT", "45"))
POSE_TARGET_FPS = float(os.environ.get("POSE_TARGET_FPS", "12"))
MODAL_CPU = float(os.environ.get("MODAL_CPU", "4"))
MODAL_MEMORY = int(os.environ.get("MODAL_MEMORY", "8192"))
MODAL_MIN_CONTAINERS = int(os.environ.get("MODAL_MIN_CONTAINERS", "0"))
MODAL_BUFFER_CONTAINERS = int(os.environ.get("MODAL_BUFFER_CONTAINERS", "0"))
MODAL_SCALEDOWN_WINDOW = int(os.environ.get("MODAL_SCALEDOWN_WINDOW", "900"))
WARM_ASR_MODEL = parse_bool_env(os.environ.get("WARM_ASR_MODEL", "1"))

runtime_env = {
    "ASR_MODEL": ASR_MODEL,
    "ASR_LANGUAGE": ASR_LANGUAGE,
    "ASR_BEAM_SIZE": str(ASR_BEAM_SIZE),
    "ASR_CPU_THREADS": str(ASR_CPU_THREADS),
    "ASR_VAD_FILTER": "1" if ASR_VAD_FILTER else "0",
    "ASR_VAD_MIN_SILENCE_MS": str(ASR_VAD_MIN_SILENCE_MS),
    "ASR_GPU": ASR_GPU,
    "ASR_COMPUTE_TYPE": ASR_COMPUTE_TYPE,
    "SIGNED_LANGUAGE": SIGNED_LANGUAGE,
    "SPOKEN_LANGUAGE": SPOKEN_LANGUAGE,
    "SIGN_MT_TIMEOUT": str(SIGN_MT_TIMEOUT),
    "POSE_TARGET_FPS": str(POSE_TARGET_FPS),
    "WARM_ASR_MODEL": "1" if WARM_ASR_MODEL else "0",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
}


def _read_uint16(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<H", data, offset)[0], offset + 2


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = _read_uint16(data, offset)
    value = data[offset : offset + length].decode("utf-8", errors="replace")
    return value, offset + length


def _pose_header_summary(data: bytes) -> tuple[float, int, int, int]:
    version = struct.unpack_from("<f", data, 0)[0]
    offset = 4 + 2 + 2 + 2
    component_count, offset = _read_uint16(data, offset)
    total_points = 0
    max_fmt_len = 0

    for _ in range(component_count):
        _name, offset = _read_string(data, offset)
        fmt, offset = _read_string(data, offset)
        max_fmt_len = max(max_fmt_len, len(fmt))
        point_count, offset = _read_uint16(data, offset)
        limb_count, offset = _read_uint16(data, offset)
        color_count, offset = _read_uint16(data, offset)
        total_points += point_count
        for _point in range(point_count):
            _point_name, offset = _read_string(data, offset)
        offset += limb_count * 4
        offset += color_count * 6

    dims = max(0, max_fmt_len - 1)
    return version, offset, total_points, dims


def downsample_pose_binary(data: bytes, target_fps: float) -> tuple[bytes, dict[str, object]]:
    info: dict[str, object] = {
        "enabled": False,
        "reason": "unchanged",
        "original_bytes": len(data),
        "output_bytes": len(data),
    }
    if target_fps <= 0:
        info["reason"] = "disabled"
        return data, info

    try:
        version, header_length, total_points, dims = _pose_header_summary(data)
        version_key = round(version, 3)
        reader_offset = header_length
        if version_key == 0.1:
            fps = float(struct.unpack_from("<H", data, reader_offset)[0])
            frame_count = int(struct.unpack_from("<H", data, reader_offset + 2)[0])
            people_count = int(struct.unpack_from("<H", data, reader_offset + 4)[0])
            body_info_size = 6
        elif version_key == 0.2:
            fps = float(struct.unpack_from("<f", data, reader_offset)[0])
            frame_count = int(struct.unpack_from("<I", data, reader_offset + 4)[0])
            people_count = int(struct.unpack_from("<H", data, reader_offset + 8)[0])
            body_info_size = 10
        else:
            info["reason"] = f"unsupported-version-{version:.3f}"
            return data, info

        info.update(
            {
                "original_fps": fps,
                "original_frames": frame_count,
                "people_count": people_count,
            }
        )
        if fps <= 0 or frame_count <= 2 or people_count <= 0 or total_points <= 0 or dims <= 0:
            info["reason"] = "invalid-body"
            return data, info
        if fps <= target_fps:
            info["reason"] = "already-at-or-under-target-fps"
            return data, info

        duration = frame_count / fps
        new_frame_count = max(2, min(frame_count, int(math.ceil(duration * target_fps))))
        if new_frame_count >= frame_count:
            info["reason"] = "no-frame-reduction"
            return data, info

        indices: list[int] = []
        last_index = -1
        for output_index in range(new_frame_count):
            if new_frame_count == 1:
                frame_index = 0
            else:
                frame_index = int(round(output_index * (frame_count - 1) / (new_frame_count - 1)))
            frame_index = max(0, min(frame_count - 1, frame_index))
            if frame_index <= last_index:
                frame_index = min(frame_count - 1, last_index + 1)
            indices.append(frame_index)
            last_index = frame_index

        frame_values = people_count * total_points * dims
        confidence_values = people_count * total_points
        data_frame_bytes = frame_values * 4
        confidence_frame_bytes = confidence_values * 4
        data_offset = header_length + body_info_size
        confidence_offset = data_offset + frame_count * data_frame_bytes
        expected_length = confidence_offset + frame_count * confidence_frame_bytes
        if expected_length > len(data):
            info["reason"] = "truncated-body"
            return data, info

        downsampled = bytearray(data[:header_length])
        new_fps = new_frame_count / duration
        if version_key == 0.1:
            downsampled.extend(
                struct.pack("<HHH", max(1, int(round(new_fps))), new_frame_count, people_count)
            )
        else:
            downsampled.extend(struct.pack("<fIH", float(new_fps), new_frame_count, people_count))

        for frame_index in indices:
            start = data_offset + frame_index * data_frame_bytes
            downsampled.extend(data[start : start + data_frame_bytes])
        for frame_index in indices:
            start = confidence_offset + frame_index * confidence_frame_bytes
            downsampled.extend(data[start : start + confidence_frame_bytes])

        output = bytes(downsampled)
        info.update(
            {
                "enabled": True,
                "reason": "downsampled",
                "target_fps": target_fps,
                "output_fps": new_fps,
                "output_frames": new_frame_count,
                "output_bytes": len(output),
                "ratio": len(output) / max(1, len(data)),
            }
        )
        return output, info
    except Exception as exc:
        info["reason"] = f"error-{type(exc).__name__}"
        return data, info


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi",
        "python-multipart",
        "requests",
        "faster-whisper",
        "hf-transfer",
    )
    .env(runtime_env)
)

app = modal.App(APP_NAME, image=image)


@app.function(
    image=image,
    gpu=ASR_GPU or None,
    cpu=MODAL_CPU,
    memory=MODAL_MEMORY,
    timeout=180,
    min_containers=MODAL_MIN_CONTAINERS,
    buffer_containers=MODAL_BUFFER_CONTAINERS,
    scaledown_window=MODAL_SCALEDOWN_WINDOW,
)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
    from fastapi.responses import JSONResponse, Response
    import requests
    from faster_whisper import WhisperModel
    from starlette.middleware.gzip import GZipMiddleware

    web = FastAPI(title="Speech to ASL Pose Server")
    web.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
    state: dict[str, object] = {
        "model": None,
        "pose_cache": {},
        "session": requests.Session(),
    }

    def normalize_text(text: str) -> str:
        return " ".join(str(text or "").strip().split())

    def parse_timing_header(value: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @web.middleware("http")
    async def add_request_timing(request: Request, call_next):
        request_started_at = time.perf_counter()
        response = await call_next(request)
        asgi_ms = (time.perf_counter() - request_started_at) * 1000.0
        handler_ms = parse_timing_header(response.headers.get("X-Total-MS", "0"))
        upload_parse_ms = max(0.0, asgi_ms - handler_ms)

        response.headers["X-ASGI-MS"] = f"{asgi_ms:.1f}"
        response.headers["X-Upload-Parse-MS"] = f"{upload_parse_ms:.1f}"
        server_timing = response.headers.get("Server-Timing", "")
        extra_timing = f"uploadparse;dur={upload_parse_ms:.1f}, asgi;dur={asgi_ms:.1f}"
        response.headers["Server-Timing"] = (
            f"{server_timing}, {extra_timing}" if server_timing else extra_timing
        )

        if request.method == "POST":
            print(
                "[request-network] "
                f"path={request.url.path} asgi_ms={asgi_ms:.1f} "
                f"handler_ms={handler_ms:.1f} upload_parse_ms={upload_parse_ms:.1f}"
            )
        return response

    def get_model() -> WhisperModel:
        model = state.get("model")
        if model is None:
            started_at = time.perf_counter()
            print(
                "[asr] loading faster-whisper "
                f"model={ASR_MODEL} device={ASR_DEVICE} compute_type={ASR_COMPUTE_TYPE}"
            )
            model_kwargs = {
                "device": ASR_DEVICE,
                "compute_type": ASR_COMPUTE_TYPE,
            }
            if ASR_CPU_THREADS > 0:
                model_kwargs["cpu_threads"] = ASR_CPU_THREADS
            model = WhisperModel(ASR_MODEL, **model_kwargs)
            state["model"] = model
            print(f"[asr] model loaded in {time.perf_counter() - started_at:.2f}s")
        return model

    def transcribe_audio(audio_bytes: bytes, filename: str, language: str) -> str:
        suffix = os.path.splitext(filename or "speech.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            model = get_model()
            segments, _info = model.transcribe(
                tmp_path,
                language=language or ASR_LANGUAGE,
                beam_size=max(1, ASR_BEAM_SIZE),
                vad_filter=ASR_VAD_FILTER,
                vad_parameters={"min_silence_duration_ms": max(100, ASR_VAD_MIN_SILENCE_MS)},
                condition_on_previous_text=False,
            )
            return normalize_text(" ".join(segment.text.strip() for segment in segments))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def fetch_pose(text: str, spoken: str, signed: str) -> tuple[bytes, str, bool]:
        cache_key = f"{spoken}:{signed}:{text.lower()}"
        pose_cache = state["pose_cache"]
        if isinstance(pose_cache, dict) and cache_key in pose_cache:
            pose_bytes, content_type = pose_cache[cache_key]
            return pose_bytes, content_type, True  # type: ignore[misc]

        session = state["session"]
        if not isinstance(session, requests.Session):
            session = requests.Session()
            state["session"] = session
        response = session.get(
            SIGN_MT_ENDPOINT,
            params={
                "text": text,
                "spoken": spoken,
                "signed": signed,
            },
            headers={
                "Origin": "http://localhost",
                "User-Agent": "modal-speech-to-pose/1.0",
            },
            timeout=SIGN_MT_TIMEOUT,
        )
        if not response.ok:
            raise HTTPException(
                status_code=502,
                detail=f"sign.mt failed HTTP {response.status_code}: {response.text[:240]}",
            )

        content_type = response.headers.get("content-type", "application/pose")
        result = (response.content, content_type)
        if isinstance(pose_cache, dict):
            if len(pose_cache) > 256:
                pose_cache.clear()
            pose_cache[cache_key] = result
        return response.content, content_type, False

    def build_pose_response(
        *,
        transcript: str,
        audio_bytes: Optional[bytes],
        audio_filename: str,
        language: str,
        spoken: str,
        signed: str,
        response_format: str,
        payload_format: str,
    ):
        total_started_at = time.perf_counter()
        asr_ms = 0.0
        audio_bytes = audio_bytes or b""
        transcript = normalize_text(transcript)
        if not transcript:
            if not audio_bytes:
                raise HTTPException(status_code=400, detail="audio file or text is required")
            asr_started_at = time.perf_counter()
            transcript = transcribe_audio(audio_bytes, audio_filename or "speech.wav", language)
            asr_ms = (time.perf_counter() - asr_started_at) * 1000.0

        transcript = normalize_text(transcript)
        if not transcript:
            total_ms = (time.perf_counter() - total_started_at) * 1000.0
            print(
                "[request] no_speech "
                f"payload={payload_format} audio_kb={len(audio_bytes) / 1024.0:.1f} "
                f"asr_ms={asr_ms:.1f} total_ms={total_ms:.1f}"
            )
            return Response(
                status_code=204,
                headers={
                    "X-No-Speech": "1",
                    "X-Audio-Bytes": str(len(audio_bytes)),
                    "X-Payload-Format": payload_format,
                    "X-ASR-MS": f"{asr_ms:.1f}",
                    "X-Total-MS": f"{total_ms:.1f}",
                    "Server-Timing": f"asr;dur={asr_ms:.1f}, total;dur={total_ms:.1f}",
                    "Cache-Control": "no-store",
                },
            )

        pose_started_at = time.perf_counter()
        pose_bytes, content_type, pose_cache_hit = fetch_pose(
            transcript,
            spoken or SPOKEN_LANGUAGE,
            signed or SIGNED_LANGUAGE,
        )
        pose_ms = (time.perf_counter() - pose_started_at) * 1000.0
        original_pose_bytes = pose_bytes
        pose_bytes, downsample_info = downsample_pose_binary(pose_bytes, POSE_TARGET_FPS)
        total_ms = (time.perf_counter() - total_started_at) * 1000.0
        print(
            "[request] "
            f"text={transcript!r} payload={payload_format} "
            f"audio_kb={len(audio_bytes) / 1024.0:.1f} "
            f"pose_kb={len(pose_bytes) / 1024.0:.1f} "
            f"original_pose_kb={len(original_pose_bytes) / 1024.0:.1f} "
            f"downsample={downsample_info.get('reason')} "
            f"frames={downsample_info.get('original_frames', '?')}->{downsample_info.get('output_frames', '?')} "
            f"fps={float(downsample_info.get('original_fps', 0.0) or 0.0):.1f}->{float(downsample_info.get('output_fps', 0.0) or 0.0):.1f} "
            f"asr_ms={asr_ms:.1f} pose_ms={pose_ms:.1f} total_ms={total_ms:.1f} "
            f"pose_cache={'hit' if pose_cache_hit else 'miss'}"
        )

        headers = {
            "X-Transcript": transcript,
            "X-Audio-Bytes": str(len(audio_bytes)),
            "X-Pose-Bytes": str(len(pose_bytes)),
            "X-Pose-Original-Bytes": str(len(original_pose_bytes)),
            "X-Pose-Downsample": str(downsample_info.get("reason", "unknown")),
            "X-Pose-Original-FPS": f"{float(downsample_info.get('original_fps', 0.0) or 0.0):.3f}",
            "X-Pose-FPS": f"{float(downsample_info.get('output_fps', downsample_info.get('original_fps', 0.0)) or 0.0):.3f}",
            "X-Pose-Original-Frames": str(downsample_info.get("original_frames", "")),
            "X-Pose-Frames": str(downsample_info.get("output_frames", downsample_info.get("original_frames", ""))),
            "X-Pose-Size-Ratio": f"{float(downsample_info.get('ratio', len(pose_bytes) / max(1, len(original_pose_bytes))) or 1.0):.3f}",
            "X-Pose-Target-FPS": f"{POSE_TARGET_FPS:.3f}",
            "X-Payload-Format": payload_format,
            "X-ASR-MS": f"{asr_ms:.1f}",
            "X-Sign-MT-MS": f"{pose_ms:.1f}",
            "X-Total-MS": f"{total_ms:.1f}",
            "X-Pose-Cache": "hit" if pose_cache_hit else "miss",
            "Server-Timing": f"asr;dur={asr_ms:.1f}, signmt;dur={pose_ms:.1f}, total;dur={total_ms:.1f}",
            "Cache-Control": "no-store",
        }
        if response_format == "json":
            return JSONResponse(
                {
                    "text": transcript,
                    "content_type": content_type,
                    "pose_base64": base64.b64encode(pose_bytes).decode("ascii"),
                },
                headers=headers,
            )

        return Response(
            content=pose_bytes,
            media_type=content_type or "application/pose",
            headers=headers,
        )

    @web.get("/health")
    async def health():
        return {
            "ok": True,
            "model": ASR_MODEL,
            "device": ASR_DEVICE,
            "compute_type": ASR_COMPUTE_TYPE,
            "beam_size": ASR_BEAM_SIZE,
            "vad_filter": ASR_VAD_FILTER,
            "warm_model": WARM_ASR_MODEL,
        }

    @web.get("/")
    async def root():
        return await health()

    @web.post("/")
    @web.post("/speech-to-pose")
    async def speech_to_pose(
        audio: Optional[UploadFile] = File(default=None),
        language: str = Form(default=ASR_LANGUAGE),
        spoken: str = Form(default=SPOKEN_LANGUAGE),
        signed: str = Form(default=SIGNED_LANGUAGE),
        text: str = Form(default=""),
        response_format: str = Form(default="binary"),
    ):
        audio_bytes = await audio.read() if audio is not None else b""
        return build_pose_response(
            transcript=text,
            audio_bytes=audio_bytes,
            audio_filename=audio.filename if audio is not None else "speech.wav",
            language=language,
            spoken=spoken,
            signed=signed,
            response_format=response_format,
            payload_format="multipart",
        )

    @web.post("/speech-to-pose-raw")
    async def speech_to_pose_raw(
        request: Request,
        language: str = Query(default=ASR_LANGUAGE),
        spoken: str = Query(default=SPOKEN_LANGUAGE),
        signed: str = Query(default=SIGNED_LANGUAGE),
        text: str = Query(default=""),
        response_format: str = Query(default="binary"),
    ):
        return build_pose_response(
            transcript=text,
            audio_bytes=await request.body(),
            audio_filename="speech.wav",
            language=language,
            spoken=spoken,
            signed=signed,
            response_format=response_format,
            payload_format="raw",
        )

    if WARM_ASR_MODEL:
        try:
            get_model()
        except Exception as exc:
            print(f"[asr] warmup skipped: {exc}")

    return web
