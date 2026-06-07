from __future__ import annotations

import argparse
import base64
import json
import http.server
import importlib.util
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional

import tkinter as tk


def ensure_runtime_dependencies() -> None:
    required = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "requests": "requests",
    }
    missing = [
        package_name
        for import_name, package_name in required.items()
        if importlib.util.find_spec(import_name) is None
    ]
    if not missing:
        return

    auto_install = os.environ.get("DUAL_CLIENT_AUTO_INSTALL", "1").strip().lower()
    if auto_install in {"0", "false", "no", "off"}:
        raise RuntimeError(
            "Missing Python packages: "
            + ", ".join(missing)
            + ". Install them or unset DUAL_CLIENT_AUTO_INSTALL=0."
        )

    print(f"Installing missing Python packages: {' '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_runtime_dependencies()

import cv2
import numpy as np

import client as sign_client
import khoa as pose_client


START_CLIENT_DEFAULT_SERVER_URL = (
    "https://thanhhoang12032005--openslt-realtime-realtime-app.modal.run"
)


def parse_bool_env(value: str) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def make_translucent_overlay(width: int, height: int, *, alpha: int = 176) -> tk.PhotoImage:
    width = max(1, int(width))
    height = max(1, int(height))
    alpha = max(0, min(255, int(alpha)))
    # OpenCV encodes 4-channel PNG arrays as BGRA. Tk PhotoImage keeps the
    # alpha channel, which gives a real translucent video subtitle plate.
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[:, :, 0] = 32
    pixels[:, :, 1] = 18
    pixels[:, :, 2] = 11
    pixels[:, :, 3] = alpha
    ok, png = cv2.imencode(".png", pixels)
    if not ok:
        raise RuntimeError("failed to build translucent overlay image")
    encoded = base64.b64encode(png.tobytes()).decode("ascii")
    return tk.PhotoImage(data=encoded, format="PNG")


def option_provided(argv: list[str], *options: str) -> bool:
    for option in options:
        if option in argv:
            return True
        prefix = option + "="
        if any(arg.startswith(prefix) for arg in argv):
            return True
    return False


def apply_start_client_env_defaults(args: argparse.Namespace, argv: list[str]) -> None:
    if (
        not option_provided(argv, "--base-url")
        and not os.environ.get("OPENSLT_REALTIME_BASE_URL")
    ):
        args.base_url = os.environ.get("SERVER_URL", START_CLIENT_DEFAULT_SERVER_URL)

    env_defaults: list[tuple[str, str, Callable[[str], object], str]] = [
        ("CAMERA_INDEX", "camera_index", int, "--camera-index"),
        ("CAMERA_FPS", "camera_fps", int, "--camera-fps"),
        ("CAMERA_WARMUP_SECONDS", "camera_warmup_seconds", float, "--camera-warmup-seconds"),
        ("SAMPLE_FPS", "sample_fps", int, "--sample-fps"),
        ("CHUNK_FRAMES", "chunk_frames", int, "--chunk-frames"),
        ("FRAME_WIDTH", "width", int, "--width"),
        ("FRAME_HEIGHT", "height", int, "--height"),
        ("JPEG_QUALITY", "jpeg_quality", int, "--jpeg-quality"),
        ("UPLOAD_WORKERS", "upload_workers", int, "--upload-workers"),
        ("MAX_PENDING_UPLOADS", "max_pending_uploads", int, "--max-pending-uploads"),
        ("CHUNK_ENDPOINT", "chunk_endpoint", str, "--chunk-endpoint"),
        ("REQUEST_TIMEOUT", "request_timeout", float, "--request-timeout"),
        ("DURATION_SECONDS", "duration_seconds", float, "--duration-seconds"),
        ("SERVER_STATS_INTERVAL", "server_stats_interval", int, "--server-stats-interval"),
        ("SPEECH_MODE", "speech_mode", str, "--speech-mode"),
        ("SPEECH_BACKEND", "speech_backend", str, "--speech-backend"),
        ("SPEECH_APLAY_DEVICE", "speech_aplay_device", str, "--speech-aplay-device"),
        ("SPEECH_VOICE", "speech_voice", str, "--speech-voice"),
        ("SPEECH_RATE", "speech_rate", int, "--speech-rate"),
        ("SPEECH_COMMAND_TIMEOUT", "speech_command_timeout", float, "--speech-command-timeout"),
        ("AUDIO_CUE_BACKEND", "audio_cue_backend", str, "--audio-cue-backend"),
        ("AUDIO_CUE_DEVICE", "audio_cue_device", str, "--audio-cue-device"),
        ("AUDIO_CUE_VOLUME", "audio_cue_volume", float, "--audio-cue-volume"),
        ("AUDIO_HIGH_LATENCY_MS", "audio_high_latency_ms", float, "--audio-high-latency-ms"),
        (
            "AUDIO_HIGH_LATENCY_INTERVAL_SECONDS",
            "audio_high_latency_interval_seconds",
            float,
            "--audio-high-latency-interval-seconds",
        ),
    ]
    for env_name, attr, caster, option in env_defaults:
        if env_name in os.environ and not option_provided(argv, option):
            setattr(args, attr, caster(os.environ[env_name]))

    bool_defaults: list[tuple[str, str, str, str]] = [
        ("DEBUG_TIMING", "debug_timing", "--debug-timing", "--no-debug-timing"),
        ("SPEAK", "speak", "--speak", "--no-speak"),
        ("SPEECH_DEBUG", "speech_debug", "--speech-debug", "--no-speech-debug"),
        ("AUDIO_CUES", "audio_cues", "--audio-cues", "--no-audio-cues"),
        ("AUDIO_CUE_DEBUG", "audio_cue_debug", "--audio-cue-debug", "--no-audio-cue-debug"),
        (
            "AUDIO_UTTERANCE_STARTED_CUE",
            "audio_utterance_started_cue",
            "--audio-utterance-started-cue",
            "--no-audio-utterance-started-cue",
        ),
    ]
    for env_name, attr, enabled_option, disabled_option in bool_defaults:
        if env_name in os.environ and not option_provided(argv, enabled_option, disabled_option):
            setattr(args, attr, parse_bool_env(os.environ[env_name]))


def build_pose_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--endpoint", default=pose_client.SIGN_MT_ENDPOINT)
    parser.add_argument("--spoken", default=pose_client.DEFAULT_SPOKEN_LANGUAGE)
    parser.add_argument("--signed", default=pose_client.DEFAULT_SIGNED_LANGUAGE)
    parser.add_argument(
        "--asr-backend",
        choices=["faster-whisper", "browser", "whisper", "vosk"],
        default=os.environ.get("ASR_BACKEND", "faster-whisper").strip().lower(),
        help="Speech-to-text backend for the English -> ASL side.",
    )
    parser.add_argument(
        "--faster-whisper-model",
        default=os.environ.get("FASTER_WHISPER_MODEL", "tiny.en"),
        help="Faster-Whisper model name/path, e.g. base.en, small.en, distil-large-v3, large-v3-turbo.",
    )
    parser.add_argument(
        "--faster-whisper-device",
        default=os.environ.get("FASTER_WHISPER_DEVICE", "cpu"),
        help="Faster-Whisper device: cpu or cuda.",
    )
    parser.add_argument(
        "--faster-whisper-compute-type",
        default=os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "int8"),
        help="Faster-Whisper compute type: int8 for CPU, float16 for CUDA.",
    )
    parser.add_argument(
        "--faster-whisper-language",
        default=os.environ.get("FASTER_WHISPER_LANGUAGE", "en"),
        help="Language hint passed to Faster-Whisper.",
    )
    parser.add_argument(
        "--faster-whisper-beam-size",
        type=int,
        default=int(os.environ.get("FASTER_WHISPER_BEAM_SIZE", "5")),
        help="Beam size for decoding; lower is faster, higher may be more accurate.",
    )
    parser.add_argument(
        "--faster-whisper-cpu-threads",
        type=int,
        default=int(os.environ.get("FASTER_WHISPER_CPU_THREADS", "0")),
        help="CPU threads for CTranslate2; 0 lets the runtime choose.",
    )
    parser.add_argument(
        "--faster-whisper-vad-filter",
        action=argparse.BooleanOptionalAction,
        default=parse_bool_env(os.environ.get("FASTER_WHISPER_VAD_FILTER", "1")),
        help="Enable Silero VAD inside Faster-Whisper in addition to microphone endpointing.",
    )
    parser.add_argument(
        "--browser-asr-host",
        default=os.environ.get("BROWSER_ASR_HOST", "127.0.0.1"),
        help="Local host used by the browser SpeechRecognition bridge.",
    )
    parser.add_argument(
        "--browser-asr-port",
        type=int,
        default=int(os.environ.get("BROWSER_ASR_PORT", "8765")),
        help="Local port used by the browser SpeechRecognition bridge; 0 chooses a free port.",
    )
    parser.add_argument(
        "--browser-asr-language",
        default=os.environ.get("BROWSER_ASR_LANGUAGE", "en-US"),
        help="Language tag used by Web Speech API, e.g. en-US.",
    )
    parser.add_argument(
        "--browser-asr-open",
        action=argparse.BooleanOptionalAction,
        default=parse_bool_env(os.environ.get("BROWSER_ASR_OPEN", "1")),
        help="Open the local browser speech bridge tab automatically.",
    )
    parser.add_argument(
        "--whisper-model",
        default=os.environ.get("WHISPER_MODEL", "tiny.en"),
        help="Local Whisper model name/path, e.g. tiny.en, base.en, small.en.",
    )
    parser.add_argument(
        "--whisper-device",
        default=os.environ.get("WHISPER_DEVICE", "cpu"),
        help="Whisper device: cpu or cuda.",
    )
    parser.add_argument(
        "--whisper-language",
        default=os.environ.get("WHISPER_LANGUAGE", "en"),
        help="Language hint passed to Whisper.",
    )
    parser.add_argument(
        "--whisper-vad-threshold",
        type=float,
        default=float(os.environ.get("WHISPER_VAD_THRESHOLD", "0.012")),
        help="RMS threshold that marks microphone audio as speech.",
    )
    parser.add_argument(
        "--whisper-silence-seconds",
        type=float,
        default=float(os.environ.get("WHISPER_SILENCE_SECONDS", "0.9")),
        help="Silence duration used to finalize one utterance.",
    )
    parser.add_argument(
        "--whisper-min-utterance-seconds",
        type=float,
        default=float(os.environ.get("WHISPER_MIN_UTTERANCE_SECONDS", "0.45")),
        help="Ignore utterances shorter than this duration.",
    )
    parser.add_argument(
        "--whisper-max-utterance-seconds",
        type=float,
        default=float(os.environ.get("WHISPER_MAX_UTTERANCE_SECONDS", "12.0")),
        help="Force-finalize a long utterance after this duration.",
    )
    parser.add_argument("--concurrency", type=int, default=pose_client.DEFAULT_CONCURRENCY)
    parser.add_argument("--playback-rate", type=float, default=pose_client.DEFAULT_PLAYBACK_RATE)
    parser.add_argument(
        "--pose-render-fps",
        type=float,
        default=float(os.environ.get("POSE_RENDER_FPS", "30")),
        help="Target UI redraw FPS for the skeleton pose renderer.",
    )
    parser.add_argument("--sample-rate", type=int, default=pose_client.ASR_SAMPLE_RATE)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--interpolate",
        action=argparse.BooleanOptionalAction,
        default=parse_bool_env(os.environ.get("POSE_INTERPOLATE", "1")),
        help="Interpolate between pose frames for smoother skeleton motion.",
    )
    parser.add_argument("--demo-text", nargs="*", default=None)
    parser.add_argument("--auto-close-seconds", type=float, default=0.0)
    parser.add_argument("--self-test", default=None)
    return parser


def build_dashboard_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dashboard-geometry", default="1366x768")
    parser.add_argument("--fullscreen", action="store_true", default=False)
    parser.add_argument("--ui-fps", type=float, default=15.0)
    parser.add_argument("--mirror-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-auto-start", action="store_true", default=False)
    return parser


def print_combined_help() -> None:
    print("usage: dual_client.py [client options] [speech-pose options] [dashboard options]")
    print()
    print("Runs the ASL -> English camera pipeline and the English -> ASL pose pipeline")
    print("side by side in one Tkinter dashboard.")
    print()
    print("Most client options are inherited from client.py.")
    print()
    print("speech-pose options:")
    print(build_pose_arg_parser().format_help())
    print("dashboard options:")
    print(build_dashboard_arg_parser().format_help())
    print("client.py options:")
    sign_client.build_arg_parser().print_help()


def parse_dual_args(argv: list[str]) -> tuple[argparse.Namespace, argparse.Namespace, argparse.Namespace]:
    if option_provided(argv, "-h", "--help"):
        print_combined_help()
        raise SystemExit(0)

    client_parser = sign_client.build_arg_parser()
    client_args, remaining = client_parser.parse_known_args(argv)
    apply_start_client_env_defaults(client_args, argv)

    pose_parser = build_pose_arg_parser()
    pose_args, remaining = pose_parser.parse_known_args(remaining)
    pose_args.asr_backend = str(pose_args.asr_backend).strip().lower()
    if pose_args.asr_backend not in {"faster-whisper", "browser", "whisper", "vosk"}:
        pose_parser.error("--asr-backend must be faster-whisper, browser, whisper, or vosk")
    pose_args.concurrency = max(1, int(pose_args.concurrency))
    pose_args.playback_rate = max(0.1, float(pose_args.playback_rate))
    pose_args.pose_render_fps = max(1.0, float(pose_args.pose_render_fps))
    pose_args.faster_whisper_beam_size = max(1, int(pose_args.faster_whisper_beam_size))
    pose_args.faster_whisper_cpu_threads = max(0, int(pose_args.faster_whisper_cpu_threads))

    dashboard_parser = build_dashboard_arg_parser()
    dashboard_args, remaining = dashboard_parser.parse_known_args(remaining)
    dashboard_args.ui_fps = max(1.0, float(dashboard_args.ui_fps))

    if remaining:
        dashboard_parser.error(f"unrecognized arguments: {' '.join(remaining)}")

    # The dashboard itself is the preview. Keeping OpenCV preview enabled would
    # open a third window and break the single-screen operator workflow.
    client_args.show_preview = False
    return client_args, pose_args, dashboard_args


class TkEventBridge:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.events: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[tuple[np.ndarray, dict[str, object]]] = None
        self.closed = False
        self.camera_handler: Optional[Callable[[np.ndarray, dict[str, object]], None]] = None

    def set_camera_handler(self, handler: Callable[[np.ndarray, dict[str, object]], None]) -> None:
        self.camera_handler = handler

    def post(self, callback: Callable[[], None]) -> None:
        if self.closed:
            return
        self.events.put(callback)

    def post_camera_frame(self, frame: np.ndarray, snapshot: dict[str, object]) -> None:
        if self.closed:
            return
        with self.frame_lock:
            self.latest_frame = (frame, snapshot)

    def start(self) -> None:
        self.root.after(16, self.drain)

    def close(self) -> None:
        self.closed = True
        with self.frame_lock:
            self.latest_frame = None

    def drain(self) -> None:
        if self.closed:
            return

        frame_payload: Optional[tuple[np.ndarray, dict[str, object]]] = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame_payload = self.latest_frame
                self.latest_frame = None

        if frame_payload is not None and self.camera_handler is not None:
            frame, snapshot = frame_payload
            self.camera_handler(frame, snapshot)

        for _ in range(80):
            try:
                callback = self.events.get_nowait()
            except queue.Empty:
                break
            try:
                callback()
            except tk.TclError:
                self.close()
                return

        self.root.after(16, self.drain)


class SignToSpeechPanel:
    def __init__(self, parent: tk.Widget, *, mirror_camera: bool):
        self.mirror_camera = mirror_camera
        self.photo: Optional[tk.PhotoImage] = None
        self.overlay_photo: Optional[tk.PhotoImage] = None
        self.overlay_photo_size: tuple[int, int] = (0, 0)
        self.last_frame: Optional[np.ndarray] = None
        self.last_snapshot: dict[str, object] = {}
        self.subtitle_text = "Waiting for ASL input..."
        self.status_text = "Starting"

        self.container = tk.Frame(parent, bg="#121826", highlightthickness=0, bd=0)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.container.rowconfigure(0, weight=1)
        self.container.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.container,
            bg="#121826",
            highlightthickness=0,
            bd=0,
        )
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.canvas.bind("<Configure>", lambda _event: self.render_frame())

    def update_state(self, snapshot: dict[str, object]) -> None:
        self.last_snapshot = snapshot
        status = str(snapshot.get("status") or "Streaming")
        detail = str(snapshot.get("detail") or "")
        draft = str(snapshot.get("draft") or "")
        final = str(snapshot.get("final") or "")
        subtitle = final or draft or detail or "Waiting for ASL input..."

        status_bits = [
            f"Status: {status}",
            f"Session: {snapshot.get('session_short') or 'none'}",
            f"Chunks: {snapshot.get('chunks_sent') or 0}",
            f"Pending: {snapshot.get('pending_uploads') or 0}",
            f"Inflight: {snapshot.get('inflight_uploads') or 0}",
        ]
        self.status_text = "  |  ".join(status_bits)
        self.subtitle_text = subtitle
        self.render_frame()

    def update_frame(self, frame_bgr: np.ndarray, snapshot: dict[str, object]) -> None:
        self.last_frame = frame_bgr
        self.update_state(snapshot)
        self.render_frame()

    def render_frame(self) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, width, height, fill="#121826", outline="")

        if self.last_frame is None or width <= 8 or height <= 8:
            self.canvas.create_text(
                width / 2,
                height / 2,
                text="Camera starting",
                fill="#cbd5e1",
                font=("Segoe UI", 13, "bold"),
            )
            if width > 8 and height > 8:
                self._draw_overlay(width, height)
            return

        frame = self.last_frame
        if self.mirror_camera:
            frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        scale = min(width / max(1, frame_w), height / max(1, frame_h))
        target_w = max(1, int(frame_w * scale))
        target_h = max(1, int(frame_h * scale))
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", resized)
        if not ok:
            self.canvas.create_text(
                width / 2,
                height / 2,
                text="Camera frame render error",
                fill="#fecaca",
                font=("Segoe UI", 13, "bold"),
            )
            return
        encoded = base64.b64encode(png.tobytes()).decode("ascii")
        self.photo = tk.PhotoImage(data=encoded, format="PNG")
        self.canvas.create_image(width / 2, height / 2, image=self.photo, anchor=tk.CENTER)
        self._draw_overlay(width, height)

    def _draw_overlay(self, width: int, height: int) -> None:
        overlay_width = max(260, int(width * 0.92))
        overlay_height = min(max(112, int(height * 0.18)), 160)
        x = int((width - overlay_width) / 2)
        y = max(12, height - overlay_height - 22)

        if self.overlay_photo is None or self.overlay_photo_size != (overlay_width, overlay_height):
            self.overlay_photo = make_translucent_overlay(overlay_width, overlay_height, alpha=178)
            self.overlay_photo_size = (overlay_width, overlay_height)

        self.canvas.create_image(x, y, image=self.overlay_photo, anchor=tk.NW)
        self.canvas.create_rectangle(
            x,
            y,
            x + overlay_width,
            y + overlay_height,
            outline="#334155",
            width=1,
        )
        self.canvas.create_text(
            x + 16,
            y + 13,
            text="ASL TO ENGLISH",
            fill="#93c5fd",
            font=("Segoe UI", 8, "bold"),
            anchor=tk.NW,
        )
        self.canvas.create_text(
            x + 16,
            y + 35,
            text=self.status_text,
            fill="#cbd5e1",
            font=("Segoe UI", 9, "bold"),
            anchor=tk.NW,
            width=overlay_width - 32,
        )
        self.canvas.create_text(
            x + overlay_width / 2,
            y + overlay_height - 38,
            text=self.subtitle_text,
            fill="#ffffff",
            font=("Segoe UI", 18, "bold"),
            anchor=tk.CENTER,
            justify=tk.CENTER,
            width=overlay_width - 32,
        )


class BrowserSpeechAsr:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        language: str,
        open_browser: bool,
        on_final: Callable[[str], None],
        on_interim: Callable[[str], None],
        on_status: Callable[[str], None],
    ):
        self.host = str(host or "127.0.0.1").strip() or "127.0.0.1"
        self.port = max(0, int(port))
        self.language = str(language or "en-US").strip() or "en-US"
        self.open_browser = bool(open_browser)
        self.on_final = on_final
        self.on_interim = on_interim
        self.on_status = on_status
        self.stop_event = threading.Event()
        self.server: Optional[http.server.ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.active = False
        self.last_final_text = ""
        self.last_final_at = 0.0
        self.url = ""

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return

        self.stop_event.clear()
        self.server = self._create_server()
        bound_host, bound_port = self.server.server_address[:2]
        display_host = "127.0.0.1" if bound_host in {"0.0.0.0", ""} else bound_host
        self.url = f"http://{display_host}:{bound_port}/"
        self.thread = threading.Thread(
            target=self._serve,
            name="BrowserSpeechAsr",
            daemon=True,
        )
        self.thread.start()
        self.active = True
        self.on_status(f"Open browser speech bridge: {self.url}")
        if self.open_browser:
            webbrowser.open(self.url)

    def stop(self) -> None:
        self.stop_event.set()
        server = self.server
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
        self.server = None
        self.active = False

    def _serve(self) -> None:
        try:
            if self.server is not None:
                self.server.serve_forever(poll_interval=0.25)
        finally:
            self.active = False

    def _create_server(self) -> http.server.ThreadingHTTPServer:
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in {"/", "/index.html"}:
                    self._send_bytes(owner._html().encode("utf-8"), "text/html; charset=utf-8")
                    return
                if self.path == "/health":
                    self._send_json({"ok": True})
                    return
                self.send_error(404)

            def do_POST(self) -> None:
                if self.path != "/transcript":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length") or "0")
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    self.send_error(400, "invalid JSON")
                    return
                owner._handle_browser_payload(payload)
                self._send_json({"ok": True})

            def log_message(self, _format: str, *_args) -> None:
                return

            def _send_json(self, payload: dict[str, object]) -> None:
                self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json")

            def _send_bytes(self, payload: bytes, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

        try:
            return http.server.ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError:
            if self.port == 0:
                raise
            return http.server.ThreadingHTTPServer((self.host, 0), Handler)

    def _handle_browser_payload(self, payload: dict[str, object]) -> None:
        kind = str(payload.get("kind") or "").strip().lower()
        text = pose_client.normalize_subtitle_text(str(payload.get("text") or ""))
        if kind == "status":
            status = str(payload.get("status") or text or "Browser speech bridge")
            self.on_status(status)
            return
        if kind == "error":
            detail = str(payload.get("error") or text or "Browser speech error")
            self.on_status(f"Browser ASR error: {detail}")
            return
        if kind == "interim":
            self.on_interim(text)
            return
        if kind == "final":
            self._emit_final(text)

    def _emit_final(self, text: str) -> None:
        normalized = pose_client.normalize_pipeline_text(text, "en")
        if not normalized:
            return

        now = time.perf_counter()
        duplicate = (
            normalized.lower() == self.last_final_text.lower()
            and now - self.last_final_at < pose_client.ASR_DUPLICATE_WINDOW_SECONDS
        )
        if duplicate:
            return

        self.last_final_text = normalized
        self.last_final_at = now
        self.on_final(normalized)
        self.on_interim("")

    def _html(self) -> str:
        language = json.dumps(self.language)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Speech Bridge</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Segoe UI, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: #eef2f6;
      color: #111827;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
    }}
    main {{
      width: min(720px, calc(100vw - 32px));
      background: #ffffff;
      border: 1px solid #d7dde5;
      padding: 24px;
      box-sizing: border-box;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      color: #536274;
      line-height: 1.5;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      margin: 22px 0;
      flex-wrap: wrap;
    }}
    button {{
      border: 0;
      padding: 10px 16px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    #start {{
      background: #1f6feb;
      color: #ffffff;
    }}
    #stop {{
      background: #e8edf4;
      color: #172033;
    }}
    .panel {{
      background: #f8fafc;
      border: 1px solid #e1e6ee;
      padding: 14px;
      margin-top: 12px;
    }}
    .label {{
      font-size: 12px;
      font-weight: 800;
      color: #526175;
      margin-bottom: 6px;
    }}
    #status {{
      color: #2457a6;
      font-weight: 700;
    }}
    #text {{
      min-height: 72px;
      font-size: 22px;
      font-weight: 800;
      overflow-wrap: anywhere;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Browser Speech Bridge</h1>
    <p>Use Chrome or Edge, allow microphone access, then keep this tab open while the dashboard runs.</p>
    <div class="actions">
      <button id="start" type="button">Start Listening</button>
      <button id="stop" type="button">Stop</button>
    </div>
    <div class="panel">
      <div class="label">STATUS</div>
      <div id="status">Ready</div>
    </div>
    <div class="panel">
      <div class="label">TRANSCRIPT</div>
      <div id="text">Waiting for speech...</div>
    </div>
  </main>
  <script>
    const language = {language};
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const statusEl = document.getElementById("status");
    const textEl = document.getElementById("text");
    const startButton = document.getElementById("start");
    const stopButton = document.getElementById("stop");
    let recognition = null;
    let running = false;
    let restartTimer = null;

    async function send(payload) {{
      try {{
        await fetch("/transcript", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload)
        }});
      }} catch (error) {{
        statusEl.textContent = "Bridge disconnected: " + error;
      }}
    }}

    function setStatus(message) {{
      statusEl.textContent = message;
      send({{ kind: "status", status: message }});
    }}

    function buildRecognition() {{
      if (!Recognition) {{
        setStatus("SpeechRecognition is not supported. Use Chrome or Edge.");
        return null;
      }}
      const instance = new Recognition();
      instance.lang = language;
      instance.continuous = true;
      instance.interimResults = true;
      instance.maxAlternatives = 1;
      instance.onstart = () => setStatus("Listening");
      instance.onspeechstart = () => setStatus("Speech detected");
      instance.onspeechend = () => setStatus("Processing");
      instance.onerror = (event) => {{
        const message = event.error || "unknown";
        statusEl.textContent = "Speech error: " + message;
        send({{ kind: "error", error: message }});
      }};
      instance.onend = () => {{
        if (!running) {{
          setStatus("Stopped");
          return;
        }}
        clearTimeout(restartTimer);
        restartTimer = setTimeout(() => {{
          try {{
            instance.start();
          }} catch (error) {{
            statusEl.textContent = "Restarting...";
          }}
        }}, 350);
      }};
      instance.onresult = (event) => {{
        let interim = "";
        let finalText = "";
        for (let index = event.resultIndex; index < event.results.length; index += 1) {{
          const transcript = event.results[index][0].transcript.trim();
          if (!transcript) continue;
          if (event.results[index].isFinal) {{
            finalText += transcript + " ";
          }} else {{
            interim += transcript + " ";
          }}
        }}
        finalText = finalText.trim();
        interim = interim.trim();
        if (interim) {{
          textEl.textContent = interim;
          send({{ kind: "interim", text: interim }});
        }}
        if (finalText) {{
          textEl.textContent = finalText;
          send({{ kind: "final", text: finalText }});
        }}
      }};
      return instance;
    }}

    function startListening() {{
      if (running) return;
      recognition = recognition || buildRecognition();
      if (!recognition) return;
      running = true;
      try {{
        recognition.start();
      }} catch (error) {{
        setStatus("Already listening");
      }}
    }}

    function stopListening() {{
      running = false;
      clearTimeout(restartTimer);
      if (recognition) {{
        try {{ recognition.stop(); }} catch (error) {{}}
      }}
      setStatus("Stopped");
    }}

    startButton.addEventListener("click", startListening);
    stopButton.addEventListener("click", stopListening);
    send({{ kind: "status", status: "Browser bridge ready" }});
  </script>
</body>
</html>"""


class FasterWhisperStreamingAsr:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        compute_type: str,
        language: str,
        beam_size: int,
        cpu_threads: int,
        vad_filter: bool,
        sample_rate: int,
        vad_threshold: float,
        silence_seconds: float,
        min_utterance_seconds: float,
        max_utterance_seconds: float,
        on_final: Callable[[str], None],
        on_interim: Callable[[str], None],
        on_status: Callable[[str], None],
    ):
        self.model_name = str(model_name or "small.en").strip() or "small.en"
        self.device = str(device or "cpu").strip().lower() or "cpu"
        self.compute_type = str(compute_type or "int8").strip().lower() or "int8"
        self.language = str(language or "en").strip() or "en"
        self.beam_size = max(1, int(beam_size))
        self.cpu_threads = max(0, int(cpu_threads))
        self.vad_filter = bool(vad_filter)
        self.sample_rate = int(sample_rate)
        self.vad_threshold = max(0.001, float(vad_threshold))
        self.silence_seconds = max(0.2, float(silence_seconds))
        self.min_utterance_seconds = max(0.1, float(min_utterance_seconds))
        self.max_utterance_seconds = max(1.0, float(max_utterance_seconds))
        self.on_final = on_final
        self.on_interim = on_interim
        self.on_status = on_status
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.active = False
        self.last_final_text = ""
        self.last_final_at = 0.0

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name="FasterWhisperStreamingAsr",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _run(self) -> None:
        try:
            import sounddevice as sd
            from faster_whisper import WhisperModel
        except ImportError as exc:
            self.active = False
            self.on_status(
                "Install Faster-Whisper dependencies: pip install -U faster-whisper sounddevice "
                f"({exc})"
            )
            return

        if self.sample_rate != 16_000:
            self.on_status("Faster-Whisper expects 16 kHz audio; using configured sample rate anyway.")

        blocksize = max(1024, int(self.sample_rate * 0.25))
        audio_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=12)

        def audio_callback(indata, _frames, _time_info, status) -> None:
            if status:
                print(f"[Faster-Whisper ASR] audio status: {status}", file=sys.stderr)
            try:
                audio_queue.put_nowait(bytes(indata))
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    audio_queue.put_nowait(bytes(indata))
                except queue.Full:
                    pass

        self.on_status(f"Loading Faster-Whisper {self.model_name}...")
        try:
            model_kwargs: dict[str, object] = {
                "device": self.device,
                "compute_type": self.compute_type,
            }
            if self.cpu_threads > 0:
                model_kwargs["cpu_threads"] = self.cpu_threads
            model = WhisperModel(self.model_name, **model_kwargs)
        except Exception as exc:
            self.active = False
            self.on_status(f"Faster-Whisper model error: {exc}")
            print(f"[Faster-Whisper ASR] model load error: {exc}", file=sys.stderr)
            return

        phrase_chunks: list[np.ndarray] = []
        phrase_started_at = 0.0
        last_voice_at = 0.0

        def flush_phrase(reason: str) -> None:
            nonlocal phrase_chunks, phrase_started_at, last_voice_at
            if not phrase_chunks:
                return

            audio = np.concatenate(phrase_chunks).astype(np.float32, copy=False)
            phrase_chunks = []
            phrase_started_at = 0.0
            last_voice_at = 0.0

            duration = len(audio) / max(1, self.sample_rate)
            if duration < self.min_utterance_seconds:
                self.on_status("Listening (Faster-Whisper)")
                return

            self.on_status("Transcribing (Faster-Whisper)")
            try:
                segments, _info = model.transcribe(
                    audio,
                    language=self.language or None,
                    beam_size=self.beam_size,
                    vad_filter=self.vad_filter,
                    vad_parameters={"min_silence_duration_ms": 500},
                    condition_on_previous_text=False,
                )
                text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
            except Exception as exc:
                self.on_status(f"Faster-Whisper error: {exc}")
                print(f"[Faster-Whisper ASR] transcribe error ({reason}): {exc}", file=sys.stderr)
                return

            text = pose_client.normalize_pipeline_text(text, self.language or "en")
            if text:
                self._emit_final(text)
            self.on_status("Listening (Faster-Whisper)")

        try:
            self.active = True
            self.on_status("Listening (Faster-Whisper)")
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=blocksize,
                dtype="int16",
                channels=1,
                latency="high",
                callback=audio_callback,
            ):
                while not self.stop_event.is_set():
                    try:
                        raw_audio = audio_queue.get(timeout=0.25)
                    except queue.Empty:
                        if (
                            phrase_chunks
                            and last_voice_at
                            and time.perf_counter() - last_voice_at >= self.silence_seconds
                        ):
                            flush_phrase("queue-silence")
                        continue

                    samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                    if samples.size == 0:
                        continue
                    rms = float(np.sqrt(np.mean(np.square(samples))))
                    now = time.perf_counter()
                    speaking = rms >= self.vad_threshold

                    if speaking and not phrase_chunks:
                        phrase_started_at = now
                        self.on_interim("Listening...")
                    if speaking or phrase_chunks:
                        phrase_chunks.append(samples)
                    if speaking:
                        last_voice_at = now

                    phrase_age = now - phrase_started_at if phrase_started_at else 0.0
                    silence_age = now - last_voice_at if last_voice_at else 0.0
                    if phrase_chunks and silence_age >= self.silence_seconds:
                        flush_phrase("silence")
                    elif phrase_chunks and phrase_age >= self.max_utterance_seconds:
                        flush_phrase("max-duration")

            flush_phrase("stop")
            self.on_status("Stopped")
        except Exception as exc:
            self.on_status(f"Faster-Whisper ASR error: {exc}")
            print(f"[Faster-Whisper ASR] exception: {exc}", file=sys.stderr)
        finally:
            self.active = False

    def _emit_final(self, text: str) -> None:
        normalized = pose_client.normalize_pipeline_text(text, self.language or "en")
        if not normalized:
            return

        now = time.perf_counter()
        duplicate = (
            normalized.lower() == self.last_final_text.lower()
            and now - self.last_final_at < pose_client.ASR_DUPLICATE_WINDOW_SECONDS
        )
        if duplicate:
            return

        self.last_final_text = normalized
        self.last_final_at = now
        self.on_final(normalized)
        self.on_interim("")


class WhisperStreamingAsr:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        language: str,
        sample_rate: int,
        vad_threshold: float,
        silence_seconds: float,
        min_utterance_seconds: float,
        max_utterance_seconds: float,
        on_final: Callable[[str], None],
        on_interim: Callable[[str], None],
        on_status: Callable[[str], None],
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.sample_rate = int(sample_rate)
        self.vad_threshold = max(0.001, float(vad_threshold))
        self.silence_seconds = max(0.2, float(silence_seconds))
        self.min_utterance_seconds = max(0.1, float(min_utterance_seconds))
        self.max_utterance_seconds = max(1.0, float(max_utterance_seconds))
        self.on_final = on_final
        self.on_interim = on_interim
        self.on_status = on_status
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.active = False
        self.last_final_text = ""
        self.last_final_at = 0.0

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="WhisperStreamingAsr", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _run(self) -> None:
        try:
            import sounddevice as sd
            import whisper
        except ImportError as exc:
            self.active = False
            self.on_status(
                "Install Whisper ASR dependencies: pip install -U openai-whisper sounddevice "
                f"({exc})"
            )
            return

        blocksize = max(1024, int(self.sample_rate * 0.25))
        audio_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=12)

        def audio_callback(indata, _frames, _time_info, status) -> None:
            if status:
                print(f"[Whisper ASR] audio status: {status}", file=sys.stderr)
            try:
                audio_queue.put_nowait(bytes(indata))
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    audio_queue.put_nowait(bytes(indata))
                except queue.Full:
                    pass

        self.on_status(f"Loading Whisper {self.model_name}...")
        try:
            model = whisper.load_model(self.model_name, device=self.device)
        except Exception as exc:
            self.active = False
            self.on_status(f"Whisper model error: {exc}")
            print(f"[Whisper ASR] model load error: {exc}", file=sys.stderr)
            return

        phrase_chunks: list[np.ndarray] = []
        phrase_started_at = 0.0
        last_voice_at = 0.0

        def flush_phrase(reason: str) -> None:
            nonlocal phrase_chunks, phrase_started_at, last_voice_at
            if not phrase_chunks:
                return

            audio = np.concatenate(phrase_chunks)
            phrase_chunks = []
            phrase_started_at = 0.0
            last_voice_at = 0.0

            duration = len(audio) / max(1, self.sample_rate)
            if duration < self.min_utterance_seconds:
                self.on_status("Listening (Whisper)")
                return

            self.on_status("Transcribing (Whisper)")
            try:
                result = model.transcribe(
                    audio,
                    language=self.language or None,
                    fp16=self.device.lower() != "cpu",
                    verbose=False,
                )
            except Exception as exc:
                self.on_status(f"Whisper error: {exc}")
                print(f"[Whisper ASR] transcribe error ({reason}): {exc}", file=sys.stderr)
                return

            text = pose_client.normalize_pipeline_text(str(result.get("text") or ""), self.language or "en")
            if text:
                self._emit_final(text)
            self.on_status("Listening (Whisper)")

        try:
            self.active = True
            self.on_status("Listening (Whisper)")
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=blocksize,
                dtype="int16",
                channels=1,
                latency="high",
                callback=audio_callback,
            ):
                while not self.stop_event.is_set():
                    try:
                        raw_audio = audio_queue.get(timeout=0.25)
                    except queue.Empty:
                        if phrase_chunks and last_voice_at and time.perf_counter() - last_voice_at >= self.silence_seconds:
                            flush_phrase("queue-silence")
                        continue

                    samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                    if samples.size == 0:
                        continue
                    rms = float(np.sqrt(np.mean(np.square(samples))))
                    now = time.perf_counter()
                    speaking = rms >= self.vad_threshold

                    if speaking and not phrase_chunks:
                        phrase_started_at = now
                        self.on_interim("Listening...")
                    if speaking or phrase_chunks:
                        phrase_chunks.append(samples)
                    if speaking:
                        last_voice_at = now

                    phrase_age = now - phrase_started_at if phrase_started_at else 0.0
                    silence_age = now - last_voice_at if last_voice_at else 0.0
                    if phrase_chunks and silence_age >= self.silence_seconds:
                        flush_phrase("silence")
                    elif phrase_chunks and phrase_age >= self.max_utterance_seconds:
                        flush_phrase("max-duration")

            flush_phrase("stop")
            self.on_status("Stopped")
        except Exception as exc:
            self.on_status(f"Whisper ASR error: {exc}")
            print(f"[Whisper ASR] exception: {exc}", file=sys.stderr)
        finally:
            self.active = False

    def _emit_final(self, text: str) -> None:
        normalized = pose_client.normalize_pipeline_text(text, self.language or "en")
        if not normalized:
            return

        now = time.perf_counter()
        duplicate = (
            normalized.lower() == self.last_final_text.lower()
            and now - self.last_final_at < pose_client.ASR_DUPLICATE_WINDOW_SECONDS
        )
        if duplicate:
            return

        self.last_final_text = normalized
        self.last_final_at = now
        self.on_final(normalized)
        self.on_interim("")


class SpeechPosePanel:
    def __init__(self, root: tk.Tk, parent: tk.Widget, args: argparse.Namespace, bridge: TkEventBridge):
        self.root = root
        self.args = args
        self.bridge = bridge
        self.items: list[pose_client.QueueItem] = []
        self.next_id = 1
        self.next_to_play = 1
        self.active_item: Optional[pose_client.QueueItem] = None
        self.interim_transcript = ""
        self.asr_status = "Starting"
        self.api_status = "Idle"
        self.generation = 0
        self.cache: dict[str, pose_client.PoseSequence] = {}
        self.executor = ThreadPoolExecutor(max_workers=args.concurrency)
        self.closed = False
        self.overlay_photo: Optional[tk.PhotoImage] = None
        self.overlay_photo_size: tuple[int, int] = (0, 0)
        self.current_overlay_text = "Listening..."
        self.next_overlay_text = "Waiting for speech"
        self.status_overlay_text = "ASR: Starting  |  API: Idle"

        self.container = tk.Frame(parent, bg="#f7f8f6", highlightthickness=0, bd=0)
        self.container.grid(row=0, column=1, sticky="nsew")
        self.container.rowconfigure(0, weight=1)
        self.container.columnconfigure(0, weight=1)

        self._build_ui()
        pose_client.TARGET_RENDER_FPS = float(args.pose_render_fps)
        self.renderer = pose_client.SkeletonPoseRenderer(
            self.root,
            self.canvas,
            args.playback_rate,
            use_interpolation=args.interpolate,
        )
        self._install_renderer_overlay()
        if args.asr_backend == "faster-whisper":
            self.asr = FasterWhisperStreamingAsr(
                model_name=args.faster_whisper_model,
                device=args.faster_whisper_device,
                compute_type=args.faster_whisper_compute_type,
                language=args.faster_whisper_language,
                beam_size=args.faster_whisper_beam_size,
                cpu_threads=args.faster_whisper_cpu_threads,
                vad_filter=args.faster_whisper_vad_filter,
                sample_rate=args.sample_rate,
                vad_threshold=args.whisper_vad_threshold,
                silence_seconds=args.whisper_silence_seconds,
                min_utterance_seconds=args.whisper_min_utterance_seconds,
                max_utterance_seconds=args.whisper_max_utterance_seconds,
                on_final=lambda text: self.safe_after(lambda: self.handle_asr_final(text)),
                on_interim=lambda text: self.safe_after(lambda: self.set_interim_transcript(text)),
                on_status=lambda status: self.safe_after(lambda: self.set_asr_status(status)),
            )
        elif args.asr_backend == "browser":
            self.asr = BrowserSpeechAsr(
                host=args.browser_asr_host,
                port=args.browser_asr_port,
                language=args.browser_asr_language,
                open_browser=args.browser_asr_open,
                on_final=lambda text: self.safe_after(lambda: self.handle_asr_final(text)),
                on_interim=lambda text: self.safe_after(lambda: self.set_interim_transcript(text)),
                on_status=lambda status: self.safe_after(lambda: self.set_asr_status(status)),
            )
        elif args.asr_backend == "whisper":
            self.asr = WhisperStreamingAsr(
                model_name=args.whisper_model,
                device=args.whisper_device,
                language=args.whisper_language,
                sample_rate=args.sample_rate,
                vad_threshold=args.whisper_vad_threshold,
                silence_seconds=args.whisper_silence_seconds,
                min_utterance_seconds=args.whisper_min_utterance_seconds,
                max_utterance_seconds=args.whisper_max_utterance_seconds,
                on_final=lambda text: self.safe_after(lambda: self.handle_asr_final(text)),
                on_interim=lambda text: self.safe_after(lambda: self.set_interim_transcript(text)),
                on_status=lambda status: self.safe_after(lambda: self.set_asr_status(status)),
            )
        else:
            self.asr = pose_client.VoskStreamingAsr(
                model_path=args.model,
                sample_rate=args.sample_rate,
                on_final=lambda text: self.safe_after(lambda: self.handle_asr_final(text)),
                on_interim=lambda text: self.safe_after(lambda: self.set_interim_transcript(text)),
                on_status=lambda status: self.safe_after(lambda: self.set_asr_status(status)),
            )

    def _install_renderer_overlay(self) -> None:
        original_render = self.renderer.render_current_frame

        def render_with_overlay() -> None:
            original_render()
            self.draw_overlay()

        self.renderer.render_current_frame = render_with_overlay

    def _build_ui(self) -> None:
        self.canvas = tk.Canvas(
            self.container,
            bg="#f7f8f6",
            highlightthickness=0,
            bd=0,
        )
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.canvas.bind("<Configure>", lambda _event: self.draw_overlay())

    def draw_overlay(self) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        if width <= 8 or height <= 8:
            return

        self.canvas.delete("subtitle_overlay")
        overlay_width = max(260, int(width * 0.92))
        overlay_height = min(max(132, int(height * 0.22)), 178)
        x = int((width - overlay_width) / 2)
        y = max(12, height - overlay_height - 22)

        if self.overlay_photo is None or self.overlay_photo_size != (overlay_width, overlay_height):
            self.overlay_photo = make_translucent_overlay(overlay_width, overlay_height, alpha=178)
            self.overlay_photo_size = (overlay_width, overlay_height)

        self.canvas.create_image(
            x,
            y,
            image=self.overlay_photo,
            anchor=tk.NW,
            tags=("subtitle_overlay",),
        )
        self.canvas.create_rectangle(
            x,
            y,
            x + overlay_width,
            y + overlay_height,
            outline="#334155",
            width=1,
            tags=("subtitle_overlay",),
        )
        self.canvas.create_text(
            x + 16,
            y + 13,
            text="ENGLISH TO ASL POSE",
            fill="#86efac",
            font=("Segoe UI", 8, "bold"),
            anchor=tk.NW,
            tags=("subtitle_overlay",),
        )
        self.canvas.create_text(
            x + 16,
            y + 35,
            text=self.status_overlay_text,
            fill="#cbd5e1",
            font=("Segoe UI", 9, "bold"),
            anchor=tk.NW,
            width=overlay_width - 32,
            tags=("subtitle_overlay",),
        )
        self.canvas.create_text(
            x + overlay_width / 2,
            y + overlay_height - 56,
            text=self.current_overlay_text,
            fill="#ffffff",
            font=("Segoe UI", 18, "bold"),
            anchor=tk.CENTER,
            justify=tk.CENTER,
            width=overlay_width - 32,
            tags=("subtitle_overlay",),
        )
        self.canvas.create_text(
            x + overlay_width / 2,
            y + overlay_height - 22,
            text=self.next_overlay_text,
            fill="#cbd5e1",
            font=("Segoe UI", 10),
            anchor=tk.CENTER,
            justify=tk.CENTER,
            width=overlay_width - 32,
            tags=("subtitle_overlay",),
        )

    def start(self) -> None:
        self.renderer.paint_idle("Listening")
        self.render_subtitles()
        self.asr.start()
        if self.args.demo_text:
            for offset, text in enumerate(self.args.demo_text):
                self.root.after(600 + offset * 250, lambda value=text: self.enqueue_text(value, "demo"))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.generation += 1
        self.asr.stop()
        self.renderer.stop(resolve=False)
        self.executor.shutdown(wait=False, cancel_futures=True)

    def safe_after(self, callback: Callable[[], None]) -> None:
        self.bridge.post(callback)

    def handle_asr_final(self, text: str) -> None:
        self.enqueue_text(text, "speech")

    def enqueue_text(self, raw_text: str, source: str) -> Optional[pose_client.QueueItem]:
        text = pose_client.normalize_pipeline_text(raw_text, self.args.spoken)
        if not text:
            return None

        item = pose_client.QueueItem(item_id=self.next_id, text=text, source=source)
        self.next_id += 1
        self.items.append(item)
        self.api_status = "Sending"
        self.interim_transcript = ""
        self.render_subtitles()
        self.pump_playback()

        cache_key = (
            f"{self.args.spoken}:{self.args.signed}:"
            f"{pose_client.normalize_subtitle_text(text).lower()}"
        )
        if cache_key in self.cache:
            self.handle_translation_ready(item.item_id, self.generation, self.cache[cache_key])
            return item

        future = self.executor.submit(
            pose_client.fetch_pose_sequence,
            text,
            self.args.endpoint,
            self.args.spoken,
            self.args.signed,
            self.args.timeout,
        )
        future.add_done_callback(
            lambda done, item_id=item.item_id, generation=self.generation, key=cache_key: self.safe_after(
                lambda: self.handle_translation_done(item_id, generation, key, done)
            )
        )
        return item

    def handle_translation_done(
        self,
        item_id: int,
        generation: int,
        cache_key: str,
        future: Future,
    ) -> None:
        if generation != self.generation:
            return
        item = self.find_item(item_id)
        if not item:
            return
        try:
            sequence = future.result()
            self.cache[cache_key] = sequence
            self.handle_translation_ready(item_id, generation, sequence)
        except BaseException as error:
            item.error = error
            item.status = "error"
            item.status_text = "Error"
            self.api_status = "Error"
            print(f"[API] translation error: {error}", file=sys.stderr)
            self.render_subtitles()
            self.pump_playback()

    def handle_translation_ready(
        self,
        item_id: int,
        generation: int,
        sequence: pose_client.PoseSequence,
    ) -> None:
        if generation != self.generation:
            return
        item = self.find_item(item_id)
        if not item:
            return
        item.sequence = sequence
        item.ready_at = time.perf_counter()
        item.status = "ready"
        item.status_text = "Ready"
        item.error = None
        self.api_status = "Ready"
        self.render_subtitles()
        self.pump_playback()

    def pump_playback(self) -> None:
        if self.renderer.playing:
            return

        next_item = self.find_item(self.next_to_play)
        while next_item and next_item.status in ("done", "skipped"):
            self.next_to_play += 1
            next_item = self.find_item(self.next_to_play)

        while next_item and next_item.status == "error":
            next_item.status = "skipped"
            next_item.status_text = "Skipped"
            next_item.finished_at = time.perf_counter()
            self.next_to_play += 1
            next_item = self.find_item(self.next_to_play)

        if not next_item:
            self.active_item = None
            self.renderer.paint_idle("Listening")
            self.api_status = "Idle" if not self.items else self.api_status
            self.render_subtitles()
            return

        if next_item.status != "ready" or not next_item.sequence:
            self.renderer.paint_idle("Waiting")
            self.render_subtitles()
            return

        self.play_item(next_item)

    def play_item(self, item: pose_client.QueueItem) -> None:
        if not item.sequence or self.renderer.playing:
            return
        self.active_item = item
        item.status = "playing"
        item.status_text = "Playing"
        item.started_at = time.perf_counter()
        self.api_status = "Rendering"
        self.render_subtitles()
        self.renderer.play(
            item.sequence,
            lambda item_id=item.item_id: self.on_playback_finished(item_id),
        )

    def on_playback_finished(self, item_id: int) -> None:
        item = self.find_item(item_id)
        if item:
            item.status = "done"
            item.status_text = "Done"
            item.finished_at = time.perf_counter()
        self.active_item = None
        if self.next_to_play <= item_id:
            self.next_to_play = item_id + 1
        self.render_subtitles()
        self.pump_playback()

    def find_item(self, item_id: int) -> Optional[pose_client.QueueItem]:
        return next((item for item in self.items if item.item_id == item_id), None)

    def set_interim_transcript(self, text: str) -> None:
        self.interim_transcript = pose_client.normalize_subtitle_text(text)
        self.render_subtitles()

    def set_asr_status(self, status: str) -> None:
        self.asr_status = status
        self.render_subtitles()

    def render_subtitles(self) -> None:
        waiting = len([item for item in self.items if item.status not in ("done", "skipped")])
        status_bits = [
            f"ASR: {self.asr_status[:32]}",
            f"API: {self.api_status[:24]}",
            f"Queue: {waiting}",
            f"Rendering: {'Yes' if self.active_item else 'Idle'}",
            f"Cache: {len(self.cache)}",
        ]
        self.current_overlay_text = self.current_subtitle_text()
        self.next_overlay_text = self.next_subtitle_text()
        self.status_overlay_text = "  |  ".join(status_bits)
        self.draw_overlay()

    def current_subtitle_text(self) -> str:
        if self.active_item:
            return self.active_item.text
        if self.interim_transcript:
            return self.interim_transcript
        if self.asr_status:
            return self.asr_status
        return "Listening..."

    def next_subtitle_text(self) -> str:
        next_item = self.next_subtitle_item()
        if next_item:
            if next_item.status in {"ready", "translating"}:
                return next_item.text
            if next_item.status == "error" and next_item.error:
                return f"Skipped: {next_item.error}"
        return "Waiting for speech"

    def next_subtitle_item(self) -> Optional[pose_client.QueueItem]:
        minimum_id = self.active_item.item_id + 1 if self.active_item else self.next_to_play
        candidates = [
            item
            for item in self.items
            if item.item_id >= minimum_id and item.status not in ("done", "skipped", "playing")
        ]
        return min(candidates, key=lambda item: item.item_id) if candidates else None


class DashboardRealtimeChunkClient(sign_client.RealtimeChunkClient):
    def __init__(
        self,
        args: argparse.Namespace,
        bridge: TkEventBridge,
        panel: SignToSpeechPanel,
        *,
        ui_fps: float,
    ):
        super().__init__(args)
        self.base_url = str(args.base_url).strip().rstrip("/")
        self.args.base_url = self.base_url
        self.bridge = bridge
        self.panel = panel
        self.ui_frame_interval = 1.0 / max(1.0, float(ui_fps))
        self.last_ui_frame_at = 0.0
        self.current_chunk_len = 0

    def _update_preview_state(
        self,
        *,
        status: Optional[str] = None,
        detail: Optional[str] = None,
        draft: Optional[str] = None,
        final: Optional[str] = None,
    ) -> None:
        super()._update_preview_state(status=status, detail=detail, draft=draft, final=final)
        self.publish_state()

    def dashboard_snapshot(self) -> dict[str, object]:
        status, detail, draft, final = self._snapshot_preview_state()
        with self._state_lock:
            pending_uploads = len(self._pending_futures)
            inflight_uploads = self.inflight_uploads
            chunks_sent = self.chunks_sent
            frames_sent = self.frames_sent
        return {
            "status": status,
            "detail": detail,
            "draft": draft,
            "final": final,
            "session_short": self.session_id[:8] if self.session_id else "none",
            "chunks_sent": chunks_sent,
            "frames_sent": frames_sent,
            "pending_uploads": pending_uploads,
            "inflight_uploads": inflight_uploads,
            "current_chunk_len": self.current_chunk_len,
        }

    def publish_state(self) -> None:
        snapshot = self.dashboard_snapshot()
        self.bridge.post(lambda snapshot=snapshot: self.panel.update_state(snapshot))

    def publish_frame(self, frame: np.ndarray) -> None:
        now = time.perf_counter()
        if now - self.last_ui_frame_at < self.ui_frame_interval:
            return
        self.last_ui_frame_at = now
        self.bridge.post_camera_frame(frame.copy(), self.dashboard_snapshot())

    def _connect_session_async(self) -> None:
        try:
            session_id = self._create_session_with_warmup()
            if self.stop_event.is_set():
                return
            self.upload_executor = sign_client.concurrent.futures.ThreadPoolExecutor(
                max_workers=self.upload_workers,
                thread_name_prefix="chunk-sender",
            )
            self.session_id = session_id
            print(f"[session] connected session_id={self.session_id}")
            self._update_preview_state(status="Connected", detail=f"Session {self.session_id[:8]} ready")
            self.audio_cues.play("session_ready", replace_current=True, priority=3)
            self.speech_output.start()
        except Exception as exc:
            if self.stop_event.is_set():
                return
            print(f"[session] failed to create realtime session: {exc}")
            self.audio_cues.play("network_down", replace_current=True, priority=3)
            self.sender_error = str(exc)
            self._update_preview_state(status="Session error", detail=str(exc))

    def _create_session_with_warmup(self) -> str:
        session_url = f"{self.base_url}/api/session"

        while not self.stop_event.is_set():
            response = self.session.post(session_url, timeout=self.args.request_timeout)
            payload = self._safe_json(response)
            session_id = payload.get("session_id")
            if session_id:
                return str(session_id)

            status_label = payload.get("status_label") or payload.get("detail") or "runtime warming"
            print(f"[session] {status_label}")
            self._update_preview_state(status="Connecting", detail=status_label)
            if response.status_code >= 400 and not payload.get("warming", False):
                raise RuntimeError(status_label)
            time.sleep(self.args.session_warmup_seconds)

        raise RuntimeError("session creation interrupted")

    def run(self) -> None:
        self.audio_cues.start()
        self.publish_state()
        try:
            self._open_camera()
        except Exception as exc:
            print(f"[camera] failed: {exc}")
            self.audio_cues.play("network_down", replace_current=True, priority=3)
            self.sender_error = str(exc)
            self._update_preview_state(status="Camera error", detail=str(exc))
            self.stop_event.set()
            time.sleep(0.35)
            self._close_camera()
            self._print_summary(finalize_remote=False)
            self.speech_output.close()
            self.audio_cues.close()
            self.session.close()
            return

        session_thread = threading.Thread(
            target=self._connect_session_async,
            name="ASLToEnglishSession",
            daemon=True,
        )
        session_thread.start()

        current_chunk: list[bytes] = []
        current_chunk_start_ts_ms = 0.0
        last_sample_at = 0.0
        sampled_frame_count = 0
        camera_read_seconds_total = 0.0
        resize_seconds_total = 0.0
        encode_seconds_total = 0.0
        loop_count = 0
        chunk_bytes_total = 0
        chunk_frame_sizes: list[int] = []
        brightness_mean_total = 0.0
        brightness_std_total = 0.0
        run_started_at = time.perf_counter()

        try:
            while not self.stop_event.is_set():
                if (
                    self.args.duration_seconds > 0
                    and time.perf_counter() - run_started_at >= self.args.duration_seconds
                ):
                    print(f"[session] duration reached {self.args.duration_seconds:.1f}s, stopping capture")
                    break

                loop_start = time.perf_counter()
                read_started_at = time.perf_counter()
                ok, frame = self.cap.read()
                camera_read_seconds_total += time.perf_counter() - read_started_at
                loop_count += 1
                if not ok:
                    if self.video_path:
                        print("[video] reached end of file, stopping capture")
                    else:
                        print("Camera read failed, stopping.")
                    break

                resize_started_at = time.perf_counter()
                frame = cv2.resize(frame, (self.args.width, self.args.height))
                resize_seconds_total += time.perf_counter() - resize_started_at

                now = time.perf_counter()
                upload_ready = (
                    self.session_id is not None
                    and self.upload_executor is not None
                    and not self.sender_error
                )
                if upload_ready and now - last_sample_at >= self.sample_interval:
                    frame_brightness_mean, frame_brightness_std = self._compute_frame_diagnostics(frame)
                    encode_started_at = time.perf_counter()
                    encoded = self._encode_frame(frame)
                    encode_elapsed = time.perf_counter() - encode_started_at
                    if encoded is not None:
                        if not current_chunk:
                            current_chunk_start_ts_ms = time.time() * 1000.0
                        current_chunk.append(encoded)
                        frame_size = len(encoded)
                        chunk_frame_sizes.append(frame_size)
                        sampled_frame_count += 1
                        encode_seconds_total += encode_elapsed
                        chunk_bytes_total += frame_size
                        brightness_mean_total += frame_brightness_mean
                        brightness_std_total += frame_brightness_std
                        self._maybe_dump_frame_sample(
                            encoded=encoded,
                            chunk_index=self.chunk_index,
                            frame_index_in_chunk=sampled_frame_count - 1,
                            frame_size=frame_size,
                            brightness_mean=frame_brightness_mean,
                            brightness_std=frame_brightness_std,
                        )
                        last_sample_at = now

                    if len(current_chunk) >= self.args.chunk_frames:
                        if not self._submit_chunk(
                            frame_bytes_list=current_chunk,
                            frame_sizes_bytes=chunk_frame_sizes,
                            start_ts_ms=current_chunk_start_ts_ms,
                            end_ts_ms=time.time() * 1000.0,
                            sampled_frame_count=sampled_frame_count,
                            camera_read_seconds_total=camera_read_seconds_total,
                            resize_seconds_total=resize_seconds_total,
                            encode_seconds_total=encode_seconds_total,
                            loop_count=loop_count,
                            chunk_bytes_total=chunk_bytes_total,
                            brightness_mean_avg=brightness_mean_total / max(sampled_frame_count, 1),
                            brightness_std_avg=brightness_std_total / max(sampled_frame_count, 1),
                        ):
                            break
                        current_chunk = []
                        chunk_frame_sizes = []
                        current_chunk_start_ts_ms = 0.0
                        sampled_frame_count = 0
                        camera_read_seconds_total = 0.0
                        resize_seconds_total = 0.0
                        encode_seconds_total = 0.0
                        loop_count = 0
                        chunk_bytes_total = 0
                        brightness_mean_total = 0.0
                        brightness_std_total = 0.0

                self.current_chunk_len = len(current_chunk)
                self.publish_frame(frame)

                elapsed = time.perf_counter() - loop_start
                sleep_seconds = max(0.0, (1.0 / max(self.args.camera_fps, 1)) - elapsed)
                if sleep_seconds:
                    time.sleep(sleep_seconds)

            if (
                current_chunk
                and self.session_id is not None
                and self.upload_executor is not None
                and not self.sender_error
            ):
                self._submit_chunk(
                    frame_bytes_list=current_chunk,
                    frame_sizes_bytes=chunk_frame_sizes,
                    start_ts_ms=current_chunk_start_ts_ms,
                    end_ts_ms=time.time() * 1000.0,
                    sampled_frame_count=sampled_frame_count,
                    camera_read_seconds_total=camera_read_seconds_total,
                    resize_seconds_total=resize_seconds_total,
                    encode_seconds_total=encode_seconds_total,
                    loop_count=loop_count,
                    chunk_bytes_total=chunk_bytes_total,
                    brightness_mean_avg=brightness_mean_total / max(sampled_frame_count, 1),
                    brightness_std_avg=brightness_std_total / max(sampled_frame_count, 1),
                )
        finally:
            self.stop_event.set()
            session_thread.join(timeout=1.0)
            uploads_drained = self._shutdown_uploads(
                wait_seconds=min(max(5.0, self.args.request_timeout + 2.0), 30.0)
            )
            self._close_camera()
            self._print_summary(finalize_remote=uploads_drained)
            self.speech_output.close()
            self.audio_cues.close()
            self._close_upload_sessions()
            self.session.close()
            self._update_preview_state(status="Stopped", detail="Camera pipeline stopped")


class DualTranslatorDashboard:
    def __init__(
        self,
        client_args: argparse.Namespace,
        pose_args: argparse.Namespace,
        dashboard_args: argparse.Namespace,
    ):
        self.client_args = client_args
        self.pose_args = pose_args
        self.dashboard_args = dashboard_args
        self.started = False
        self.closing = False

        self.root = tk.Tk()
        self.root.title("Bidirectional English / ASL Interpreter")
        self.root.geometry(str(dashboard_args.dashboard_geometry))
        self.root.minsize(1024, 620)
        if dashboard_args.fullscreen:
            self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#eef2f6")

        self.bridge = TkEventBridge(self.root)
        self._build_ui()
        self.bridge.set_camera_handler(self.sign_panel.update_frame)

        self.sign_client = DashboardRealtimeChunkClient(
            self.client_args,
            self.bridge,
            self.sign_panel,
            ui_fps=self.dashboard_args.ui_fps,
        )
        self.sign_thread = threading.Thread(
            target=self.sign_client.run,
            name="ASLToEnglishRealtime",
            daemon=True,
        )
        self.pose_panel = SpeechPosePanel(self.root, self.main_area, self.pose_args, self.bridge)

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _event: self.close())
        self.bridge.start()

    def _build_ui(self) -> None:
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.main_area = tk.Frame(self.root, bg="#111827")
        self.main_area.grid(row=0, column=0, sticky="nsew")
        self.main_area.rowconfigure(0, weight=1)
        self.main_area.columnconfigure(0, weight=1, uniform="halves")
        self.main_area.columnconfigure(1, weight=1, uniform="halves")

        self.sign_panel = SignToSpeechPanel(
            self.main_area,
            mirror_camera=bool(self.dashboard_args.mirror_camera),
        )
        divider = tk.Frame(self.main_area, bg="#0f172a", width=2)
        divider.place(relx=0.5, rely=0, relheight=1, anchor="n")

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.sign_thread.start()
        self.pose_panel.start()
        if self.pose_args.auto_close_seconds > 0:
            self.root.after(int(self.pose_args.auto_close_seconds * 1000), self.close)

    def stop(self) -> None:
        self.sign_client.stop_event.set()
        self.pose_panel.close()

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.stop()
        self.bridge.close()
        try:
            self.sign_thread.join(timeout=1.0)
        except RuntimeError:
            pass
        self.root.after(50, self.root.destroy)

    def run(self) -> None:
        if not self.dashboard_args.no_auto_start:
            self.root.after(250, self.start)
        self.root.mainloop()


def run_pose_self_test(args: argparse.Namespace) -> None:
    sequence = pose_client.fetch_pose_sequence(
        args.self_test,
        args.endpoint,
        args.spoken,
        args.signed,
        args.timeout,
    )
    print(
        pose_client.json.dumps(
            {
                "text": sequence.text,
                "fps": sequence.fps,
                "frame_count": sequence.frame_count,
                "duration": sequence.duration,
                "content_type": sequence.content_type,
                "pose": {
                    "version": sequence.pose.header.version,
                    "width": sequence.pose.header.width,
                    "height": sequence.pose.header.height,
                    "depth": sequence.pose.header.depth,
                    "components": [
                        {
                            "name": component.name,
                            "format": component.fmt,
                            "points": len(component.points),
                        }
                        for component in sequence.pose.header.components
                    ],
                },
            },
            indent=2,
        )
    )


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    client_args, pose_args, dashboard_args = parse_dual_args(argv)

    if pose_args.self_test is not None:
        run_pose_self_test(pose_args)
        return

    app = DualTranslatorDashboard(client_args, pose_args, dashboard_args)

    def handle_signal(_signum, _frame) -> None:
        app.close()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    app.run()


if __name__ == "__main__":
    main()
