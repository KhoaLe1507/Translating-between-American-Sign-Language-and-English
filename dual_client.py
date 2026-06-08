from __future__ import annotations

import argparse
import base64
import io
import importlib.util
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import tkinter as tk


def ensure_runtime_dependencies() -> None:
    required = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "requests": "requests",
        "sounddevice": "sounddevice",
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
import requests

import client as sign_client
import khoa as pose_client


START_CLIENT_DEFAULT_SERVER_URL = (
    "https://thanhhoang12032005--openslt-realtime-realtime-app.modal.run"
)
REMOTE_POSE_DEFAULT_SERVER_URL = (
    "https://lequanganhkhoa2005--speech-to-pose-server-api-dev.modal.run/speech-to-pose"
)
REMOTE_AUDIO_DEFAULT_SAMPLE_RATE = 48_000


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


def normalize_remote_pose_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""

    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme and parsed.netloc and parsed.path.rstrip("/") in {"", "/"}:
        return urllib.parse.urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                "/speech-to-pose",
                parsed.query,
                parsed.fragment,
            )
        )
    return value.rstrip("/")


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
    parser.add_argument(
        "--remote-pose-url",
        default=os.environ.get("REMOTE_POSE_URL", REMOTE_POSE_DEFAULT_SERVER_URL).strip(),
        help="Server endpoint that accepts audio and returns binary pose, e.g. Modal /speech-to-pose.",
    )
    parser.add_argument(
        "--remote-pose-timeout",
        type=float,
        default=float(os.environ.get("REMOTE_POSE_TIMEOUT", "120")),
        help="HTTP timeout for the remote Speech-to-Pose endpoint.",
    )
    parser.add_argument(
        "--remote-pose-language",
        default=os.environ.get("REMOTE_POSE_LANGUAGE", "en"),
        help="Language hint sent to the remote ASR server.",
    )
    parser.add_argument(
        "--remote-vad-threshold",
        type=float,
        default=float(os.environ.get("REMOTE_VAD_THRESHOLD", "0.012")),
        help="RMS threshold that marks microphone audio as speech.",
    )
    parser.add_argument(
        "--remote-silence-seconds",
        type=float,
        default=float(os.environ.get("REMOTE_SILENCE_SECONDS", "0.45")),
        help="Silence duration used to finalize one utterance.",
    )
    parser.add_argument(
        "--remote-min-utterance-seconds",
        type=float,
        default=float(os.environ.get("REMOTE_MIN_UTTERANCE_SECONDS", "0.25")),
        help="Ignore utterances shorter than this duration.",
    )
    parser.add_argument(
        "--remote-min-speech-seconds",
        type=float,
        default=float(os.environ.get("REMOTE_MIN_SPEECH_SECONDS", "0.20")),
        help="Require this much voiced audio before sending a phrase to the server.",
    )
    parser.add_argument(
        "--remote-speech-start-blocks",
        type=int,
        default=int(os.environ.get("REMOTE_SPEECH_START_BLOCKS", "2")),
        help="Require this many consecutive voiced blocks before starting a phrase.",
    )
    parser.add_argument(
        "--remote-min-peak-rms",
        type=float,
        default=float(os.environ.get("REMOTE_MIN_PEAK_RMS", "0.020")),
        help="Drop phrases whose peak RMS is below this value.",
    )
    parser.add_argument(
        "--remote-send-threshold-margin",
        type=float,
        default=float(os.environ.get("REMOTE_SEND_THRESHOLD_MARGIN", "1.35")),
        help="Require phrase peak RMS to exceed the calibrated VAD threshold by this multiplier.",
    )
    parser.add_argument(
        "--remote-min-voiced-ratio",
        type=float,
        default=float(os.environ.get("REMOTE_MIN_VOICED_RATIO", "0.25")),
        help="Drop phrases if too little of the captured audio was voiced.",
    )
    parser.add_argument(
        "--remote-max-utterance-seconds",
        type=float,
        default=float(os.environ.get("REMOTE_MAX_UTTERANCE_SECONDS", "7.0")),
        help="Force-finalize a long utterance after this duration.",
    )
    parser.add_argument(
        "--remote-noise-calibration-seconds",
        type=float,
        default=float(os.environ.get("REMOTE_NOISE_CALIBRATION_SECONDS", "0.8")),
        help="Measure initial microphone noise and raise the VAD threshold above that floor.",
    )
    parser.add_argument(
        "--remote-audio-block-ms",
        type=float,
        default=float(os.environ.get("REMOTE_AUDIO_BLOCK_MS", "100")),
        help="Microphone block size for remote-pose capture; lower reduces latency but uses more callbacks.",
    )
    parser.add_argument(
        "--remote-audio-latency",
        default=os.environ.get("REMOTE_AUDIO_LATENCY", "low"),
        help="sounddevice latency hint for remote-pose: low, high, or a numeric value.",
    )
    parser.add_argument(
        "--remote-max-pending-requests",
        type=int,
        default=int(os.environ.get("REMOTE_MAX_PENDING_REQUESTS", "2")),
        help="Maximum queued remote-pose HTTP requests while the mic keeps listening.",
    )
    parser.add_argument(
        "--remote-pose-upload-format",
        choices=["raw", "multipart"],
        default=os.environ.get("REMOTE_POSE_UPLOAD_FORMAT", "raw").strip().lower(),
        help="Upload audio as raw audio/wav body or multipart form-data.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=int(os.environ.get("REMOTE_SAMPLE_RATE", str(REMOTE_AUDIO_DEFAULT_SAMPLE_RATE))),
        help="Microphone sample rate sent to the remote Speech-to-Pose server.",
    )
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=float(os.environ.get("POSE_PLAYBACK_RATE", "2.0")),
        help="Pose playback speed.",
    )
    parser.add_argument(
        "--pose-render-fps",
        type=float,
        default=float(os.environ.get("POSE_RENDER_FPS", "16")),
        help="Target UI redraw FPS for the skeleton pose renderer.",
    )
    parser.add_argument(
        "--interpolate",
        action=argparse.BooleanOptionalAction,
        default=parse_bool_env(os.environ.get("POSE_INTERPOLATE", "0")),
        help="Interpolate between pose frames for smoother skeleton motion.",
    )
    parser.add_argument("--auto-close-seconds", type=float, default=0.0)
    return parser


def build_dashboard_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--profile",
        choices=["pi"],
        default="pi",
        help="Raspberry Pi profile is always used.",
    )
    parser.add_argument("--dashboard-geometry", default="1366x768")
    parser.add_argument("--fullscreen", action="store_true", default=False)
    parser.add_argument("--ui-fps", type=float, default=float(os.environ.get("UI_FPS", "6")))
    parser.add_argument(
        "--event-poll-ms",
        type=int,
        default=int(os.environ.get("EVENT_POLL_MS", "33")),
        help="Tk event bridge poll interval in milliseconds.",
    )
    parser.add_argument("--mirror-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--camera-render-format",
        choices=["auto", "ppm", "png"],
        default=os.environ.get("CAMERA_RENDER_FORMAT", "ppm").strip().lower(),
        help="Tk camera preview image format. ppm is lighter on Raspberry Pi; png is a compatibility fallback.",
    )
    parser.add_argument(
        "--camera-display-max-width",
        type=int,
        default=int(os.environ.get("CAMERA_DISPLAY_MAX_WIDTH", "640")),
        help="Cap rendered camera preview width; 0 means fit the canvas.",
    )
    parser.add_argument(
        "--camera-display-max-height",
        type=int,
        default=int(os.environ.get("CAMERA_DISPLAY_MAX_HEIGHT", "480")),
        help="Cap rendered camera preview height; 0 means fit the canvas.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=int,
        default=int(os.environ.get("OVERLAY_ALPHA", "150")),
        help="Subtitle overlay alpha from 0 to 255.",
    )
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


def _env_present(*names: str) -> bool:
    return any(name in os.environ for name in names)


def _set_default_if_unset(
    namespace: argparse.Namespace,
    attr: str,
    value: object,
    argv: list[str],
    *options_and_envs: str,
) -> None:
    options = [item for item in options_and_envs if item.startswith("--")]
    env_names = [item for item in options_and_envs if not item.startswith("--")]
    if option_provided(argv, *options):
        return
    if _env_present(*env_names):
        return
    setattr(namespace, attr, value)


def apply_performance_profile_defaults(
    client_args: argparse.Namespace,
    pose_args: argparse.Namespace,
    dashboard_args: argparse.Namespace,
    argv: list[str],
) -> None:
    dashboard_args.profile = "pi"

    _set_default_if_unset(client_args, "camera_fps", 15, argv, "--camera-fps", "CAMERA_FPS", "OPENSLT_CAMERA_FPS")
    _set_default_if_unset(
        client_args,
        "camera_warmup_seconds",
        0.8,
        argv,
        "--camera-warmup-seconds",
        "CAMERA_WARMUP_SECONDS",
        "OPENSLT_CAMERA_WARMUP_SECONDS",
    )
    _set_default_if_unset(client_args, "sample_fps", 8, argv, "--sample-fps", "SAMPLE_FPS", "OPENSLT_SAMPLE_FPS")
    _set_default_if_unset(client_args, "chunk_frames", 8, argv, "--chunk-frames", "CHUNK_FRAMES", "OPENSLT_CHUNK_FRAMES")
    _set_default_if_unset(client_args, "width", 480, argv, "--width", "FRAME_WIDTH", "OPENSLT_FRAME_WIDTH")
    _set_default_if_unset(client_args, "height", 360, argv, "--height", "FRAME_HEIGHT", "OPENSLT_FRAME_HEIGHT")
    _set_default_if_unset(
        client_args,
        "jpeg_quality",
        55,
        argv,
        "--jpeg-quality",
        "JPEG_QUALITY",
        "OPENSLT_JPEG_QUALITY",
    )
    _set_default_if_unset(
        client_args,
        "upload_workers",
        1,
        argv,
        "--upload-workers",
        "UPLOAD_WORKERS",
        "OPENSLT_UPLOAD_WORKERS",
    )
    _set_default_if_unset(
        client_args,
        "max_pending_uploads",
        2,
        argv,
        "--max-pending-uploads",
        "MAX_PENDING_UPLOADS",
        "OPENSLT_MAX_PENDING_UPLOADS",
    )
    _set_default_if_unset(
        client_args,
        "server_stats_interval",
        0,
        argv,
        "--server-stats-interval",
        "SERVER_STATS_INTERVAL",
        "OPENSLT_SERVER_STATS_INTERVAL",
    )
    _set_default_if_unset(
        client_args,
        "debug_timing",
        False,
        argv,
        "--debug-timing",
        "--no-debug-timing",
        "DEBUG_TIMING",
        "OPENSLT_DEBUG_TIMING",
    )
    _set_default_if_unset(
        client_args,
        "speech_debug",
        False,
        argv,
        "--speech-debug",
        "--no-speech-debug",
        "SPEECH_DEBUG",
        "OPENSLT_SPEECH_DEBUG",
    )
    _set_default_if_unset(
        client_args,
        "audio_cues",
        False,
        argv,
        "--audio-cues",
        "--no-audio-cues",
        "AUDIO_CUES",
        "OPENSLT_AUDIO_CUES",
    )

    _set_default_if_unset(
        pose_args,
        "sample_rate",
        REMOTE_AUDIO_DEFAULT_SAMPLE_RATE,
        argv,
        "--sample-rate",
        "REMOTE_SAMPLE_RATE",
    )
    _set_default_if_unset(
        pose_args,
        "remote_silence_seconds",
        0.45,
        argv,
        "--remote-silence-seconds",
        "REMOTE_SILENCE_SECONDS",
    )
    _set_default_if_unset(
        pose_args,
        "remote_min_utterance_seconds",
        0.25,
        argv,
        "--remote-min-utterance-seconds",
        "REMOTE_MIN_UTTERANCE_SECONDS",
    )
    _set_default_if_unset(
        pose_args,
        "remote_max_utterance_seconds",
        7.0,
        argv,
        "--remote-max-utterance-seconds",
        "REMOTE_MAX_UTTERANCE_SECONDS",
    )
    _set_default_if_unset(
        pose_args,
        "remote_audio_block_ms",
        100.0,
        argv,
        "--remote-audio-block-ms",
        "REMOTE_AUDIO_BLOCK_MS",
    )
    _set_default_if_unset(pose_args, "pose_render_fps", 16.0, argv, "--pose-render-fps", "POSE_RENDER_FPS")
    _set_default_if_unset(
        pose_args,
        "interpolate",
        False,
        argv,
        "--interpolate",
        "--no-interpolate",
        "POSE_INTERPOLATE",
    )
    _set_default_if_unset(pose_args, "playback_rate", 2.0, argv, "--playback-rate")

    _set_default_if_unset(dashboard_args, "ui_fps", 6.0, argv, "--ui-fps")
    _set_default_if_unset(dashboard_args, "event_poll_ms", 33, argv, "--event-poll-ms", "EVENT_POLL_MS")
    _set_default_if_unset(
        dashboard_args,
        "camera_render_format",
        "ppm",
        argv,
        "--camera-render-format",
        "CAMERA_RENDER_FORMAT",
    )
    _set_default_if_unset(
        dashboard_args,
        "camera_display_max_width",
        640,
        argv,
        "--camera-display-max-width",
        "CAMERA_DISPLAY_MAX_WIDTH",
    )
    _set_default_if_unset(
        dashboard_args,
        "camera_display_max_height",
        480,
        argv,
        "--camera-display-max-height",
        "CAMERA_DISPLAY_MAX_HEIGHT",
    )
    _set_default_if_unset(dashboard_args, "overlay_alpha", 150, argv, "--overlay-alpha", "OVERLAY_ALPHA")


def parse_dual_args(argv: list[str]) -> tuple[argparse.Namespace, argparse.Namespace, argparse.Namespace]:
    if option_provided(argv, "-h", "--help"):
        print_combined_help()
        raise SystemExit(0)

    client_parser = sign_client.build_arg_parser()
    client_args, remaining = client_parser.parse_known_args(argv)
    apply_start_client_env_defaults(client_args, argv)

    pose_parser = build_pose_arg_parser()
    pose_args, remaining = pose_parser.parse_known_args(remaining)
    pose_args.playback_rate = max(0.1, float(pose_args.playback_rate))
    pose_args.pose_render_fps = max(1.0, float(pose_args.pose_render_fps))
    pose_args.remote_pose_url = normalize_remote_pose_url(pose_args.remote_pose_url)
    pose_args.remote_pose_timeout = max(1.0, float(pose_args.remote_pose_timeout))
    pose_args.remote_vad_threshold = max(0.001, float(pose_args.remote_vad_threshold))
    pose_args.remote_silence_seconds = max(0.2, float(pose_args.remote_silence_seconds))
    pose_args.remote_min_utterance_seconds = max(0.1, float(pose_args.remote_min_utterance_seconds))
    pose_args.remote_min_speech_seconds = max(0.05, float(pose_args.remote_min_speech_seconds))
    pose_args.remote_speech_start_blocks = max(1, int(pose_args.remote_speech_start_blocks))
    pose_args.remote_min_peak_rms = max(0.0, float(pose_args.remote_min_peak_rms))
    pose_args.remote_send_threshold_margin = max(1.0, float(pose_args.remote_send_threshold_margin))
    pose_args.remote_min_voiced_ratio = max(0.0, min(1.0, float(pose_args.remote_min_voiced_ratio)))
    pose_args.remote_max_utterance_seconds = max(1.0, float(pose_args.remote_max_utterance_seconds))
    pose_args.remote_noise_calibration_seconds = max(0.0, float(pose_args.remote_noise_calibration_seconds))
    pose_args.remote_audio_block_ms = max(40.0, float(pose_args.remote_audio_block_ms))
    pose_args.remote_audio_latency = str(pose_args.remote_audio_latency).strip().lower() or "low"
    pose_args.remote_max_pending_requests = max(1, int(pose_args.remote_max_pending_requests))
    pose_args.remote_pose_upload_format = str(pose_args.remote_pose_upload_format).strip().lower()
    if pose_args.remote_pose_upload_format not in {"raw", "multipart"}:
        pose_parser.error("--remote-pose-upload-format must be raw or multipart")
    pose_args.sample_rate = max(8_000, int(pose_args.sample_rate))

    dashboard_parser = build_dashboard_arg_parser()
    dashboard_args, remaining = dashboard_parser.parse_known_args(remaining)
    apply_performance_profile_defaults(client_args, pose_args, dashboard_args, argv)
    dashboard_args.ui_fps = max(1.0, float(dashboard_args.ui_fps))
    dashboard_args.event_poll_ms = max(8, int(dashboard_args.event_poll_ms))
    dashboard_args.camera_render_format = str(dashboard_args.camera_render_format).strip().lower()
    if dashboard_args.camera_render_format not in {"auto", "ppm", "png"}:
        dashboard_parser.error("--camera-render-format must be auto, ppm, or png")
    dashboard_args.camera_display_max_width = max(0, int(dashboard_args.camera_display_max_width))
    dashboard_args.camera_display_max_height = max(0, int(dashboard_args.camera_display_max_height))
    dashboard_args.overlay_alpha = max(0, min(255, int(dashboard_args.overlay_alpha)))

    if remaining:
        dashboard_parser.error(f"unrecognized arguments: {' '.join(remaining)}")

    # The dashboard itself is the preview. Keeping OpenCV preview enabled would
    # open a third window and break the single-screen operator workflow.
    client_args.show_preview = False
    return client_args, pose_args, dashboard_args


class TkEventBridge:
    def __init__(self, root: tk.Tk, *, poll_ms: int = 16):
        self.root = root
        self.poll_ms = max(8, int(poll_ms))
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
        self.root.after(self.poll_ms, self.drain)

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

        self.root.after(self.poll_ms, self.drain)


class SignToSpeechPanel:
    def __init__(
        self,
        parent: tk.Widget,
        *,
        mirror_camera: bool,
        render_format: str,
        display_max_width: int,
        display_max_height: int,
        overlay_alpha: int,
    ):
        self.mirror_camera = mirror_camera
        self.render_format = str(render_format or "auto").strip().lower()
        if self.render_format not in {"auto", "ppm", "png"}:
            self.render_format = "auto"
        self.display_max_width = max(0, int(display_max_width))
        self.display_max_height = max(0, int(display_max_height))
        self.overlay_alpha = max(0, min(255, int(overlay_alpha)))
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
        if self.display_max_width > 0:
            scale = min(scale, self.display_max_width / max(1, frame_w))
        if self.display_max_height > 0:
            scale = min(scale, self.display_max_height / max(1, frame_h))
        target_w = max(1, int(frame_w * scale))
        target_h = max(1, int(frame_h * scale))
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        self.photo = self._frame_to_photo(resized)
        if self.photo is None:
            self.canvas.create_text(
                width / 2,
                height / 2,
                text="Camera frame render error",
                fill="#fecaca",
                font=("Segoe UI", 13, "bold"),
            )
            return
        self.canvas.create_image(width / 2, height / 2, image=self.photo, anchor=tk.CENTER)
        self._draw_overlay(width, height)

    def _frame_to_photo(self, frame_bgr: np.ndarray) -> Optional[tk.PhotoImage]:
        if self.render_format in {"auto", "ppm"}:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                height, width = rgb.shape[:2]
                ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes()
                return tk.PhotoImage(data=ppm, format="PPM")
            except tk.TclError as exc:
                if self.render_format == "ppm":
                    print(f"[ui] PPM camera render failed, falling back to PNG: {exc}")
                self.render_format = "png"
            except Exception as exc:
                print(f"[ui] PPM camera render failed: {exc}")
                if self.render_format == "ppm":
                    self.render_format = "png"

        ok, png = cv2.imencode(".png", frame_bgr)
        if not ok:
            return None
        encoded = base64.b64encode(png.tobytes()).decode("ascii")
        return tk.PhotoImage(data=encoded, format="PNG")

    def _draw_overlay(self, width: int, height: int) -> None:
        overlay_width = max(260, int(width * 0.92))
        overlay_height = min(max(112, int(height * 0.18)), 160)
        x = int((width - overlay_width) / 2)
        y = max(12, height - overlay_height - 22)

        if self.overlay_photo is None or self.overlay_photo_size != (overlay_width, overlay_height):
            self.overlay_photo = make_translucent_overlay(overlay_width, overlay_height, alpha=self.overlay_alpha)
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


class NoSpeechDetected(RuntimeError):
    pass


class RemoteSpeechPoseClient:
    def __init__(
        self,
        *,
        endpoint_url: str,
        language: str,
        sample_rate: int,
        vad_threshold: float,
        silence_seconds: float,
        min_utterance_seconds: float,
        min_speech_seconds: float,
        speech_start_blocks: int,
        min_peak_rms: float,
        send_threshold_margin: float,
        min_voiced_ratio: float,
        max_utterance_seconds: float,
        timeout: float,
        audio_block_ms: float,
        audio_latency: str,
        noise_calibration_seconds: float,
        max_pending_requests: int,
        upload_format: str,
        on_pose: Callable[[pose_client.PoseSequence], None],
        on_interim: Callable[[str], None],
        on_status: Callable[[str], None],
    ):
        self.endpoint_url = str(endpoint_url or "").strip()
        self.language = str(language or "en").strip() or "en"
        self.sample_rate = int(sample_rate)
        self.vad_threshold = max(0.001, float(vad_threshold))
        self.silence_seconds = max(0.2, float(silence_seconds))
        self.min_utterance_seconds = max(0.1, float(min_utterance_seconds))
        self.min_speech_seconds = max(0.05, float(min_speech_seconds))
        self.speech_start_blocks = max(1, int(speech_start_blocks))
        self.min_peak_rms = max(0.0, float(min_peak_rms))
        self.send_threshold_margin = max(1.0, float(send_threshold_margin))
        self.min_voiced_ratio = max(0.0, min(1.0, float(min_voiced_ratio)))
        self.max_utterance_seconds = max(1.0, float(max_utterance_seconds))
        self.timeout = max(1.0, float(timeout))
        self.audio_block_seconds = max(0.04, float(audio_block_ms) / 1000.0)
        self.audio_latency = str(audio_latency or "low").strip().lower() or "low"
        self.noise_calibration_seconds = max(0.0, float(noise_calibration_seconds))
        self.max_pending_requests = max(1, int(max_pending_requests))
        self.upload_format = str(upload_format or "raw").strip().lower()
        if self.upload_format not in {"raw", "multipart"}:
            self.upload_format = "raw"
        self.on_pose = on_pose
        self.on_interim = on_interim
        self.on_status = on_status
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.active = False
        self.session = requests.Session()
        self.session_closed = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="RemotePoseUpload")
        self.pending_lock = threading.Lock()
        self.pending_requests = 0

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        if self.session_closed:
            self.session = requests.Session()
            self.session_closed = False
        if getattr(self.executor, "_shutdown", False):
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="RemotePoseUpload")
        self.thread = threading.Thread(
            target=self._run,
            name="RemoteSpeechPoseClient",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.session.close()
        self.session_closed = True
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _run(self) -> None:
        if not self.endpoint_url:
            self.on_status("Set --remote-pose-url to the Modal /speech-to-pose endpoint.")
            return

        try:
            import sounddevice as sd
        except ImportError as exc:
            self.on_status(f"Install sounddevice to send microphone audio. ({exc})")
            return

        try:
            input_device = sd.query_devices(kind="input")
            input_channels = int(input_device.get("max_input_channels", 0))
            input_name = str(input_device.get("name", "default input"))
            if input_channels < 1:
                self.on_status("No microphone input device detected")
                print(f"[Remote Speech Pose] no input channels on device: {input_device}", file=sys.stderr)
                return
            print(
                "[Remote Speech Pose] input_device "
                f"name={input_name!r} channels={input_channels} sample_rate={self.sample_rate}"
            )
        except Exception as exc:
            self.on_status(f"No microphone input device detected. ({exc})")
            print(f"[Remote Speech Pose] input device check failed: {exc}", file=sys.stderr)
            return

        blocksize = max(256, int(self.sample_rate * self.audio_block_seconds))
        queue_size = max(10, int(2.0 / self.audio_block_seconds))
        audio_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=queue_size)

        def audio_callback(indata, _frames, _time_info, status) -> None:
            if status:
                print(f"[Remote Speech Pose] audio status: {status}", file=sys.stderr)
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

        phrase_chunks: list[np.ndarray] = []
        phrase_started_at = 0.0
        last_voice_at = 0.0
        phrase_voiced_samples = 0
        phrase_voiced_blocks = 0
        phrase_peak_rms = 0.0
        speech_run_blocks = 0

        def flush_phrase(reason: str) -> None:
            nonlocal phrase_chunks, phrase_started_at, last_voice_at
            nonlocal phrase_voiced_samples, phrase_voiced_blocks, phrase_peak_rms
            nonlocal speech_run_blocks
            if not phrase_chunks:
                return

            audio_i16 = np.concatenate(phrase_chunks).astype(np.int16, copy=False)
            phrase_chunks = []
            phrase_started_at = 0.0
            last_voice_at = 0.0
            voiced_duration = phrase_voiced_samples / max(1, self.sample_rate)
            required_voiced_blocks = max(1, int((self.min_speech_seconds / self.audio_block_seconds) + 0.999))
            voiced_blocks = phrase_voiced_blocks
            peak_rms = phrase_peak_rms
            phrase_voiced_samples = 0
            phrase_voiced_blocks = 0
            phrase_peak_rms = 0.0
            speech_run_blocks = 0

            duration = len(audio_i16) / max(1, self.sample_rate)
            voiced_ratio = voiced_duration / max(duration, 0.001)
            required_peak_rms = max(
                self.min_peak_rms,
                effective_vad_threshold * self.send_threshold_margin,
            )
            if (
                duration < self.min_utterance_seconds
                or voiced_duration < self.min_speech_seconds
                or voiced_blocks < required_voiced_blocks
                or voiced_ratio < self.min_voiced_ratio
                or peak_rms < required_peak_rms
            ):
                print(
                    "[Remote Speech Pose] dropped_audio "
                    f"reason={reason} duration={duration:.2f}s voiced={voiced_duration:.2f}s "
                    f"voiced_ratio={voiced_ratio:.2f} "
                    f"voiced_blocks={voiced_blocks}/{required_voiced_blocks} "
                    f"peak_rms={peak_rms:.4f} required_peak={required_peak_rms:.4f}"
                )
                self.on_status("Listening (remote pose)")
                return

            wav_bytes = self._encode_wav(audio_i16)
            if not self._submit_request(wav_bytes, reason):
                return

        try:
            self.active = True
            effective_vad_threshold = self.vad_threshold
            calibration_values: list[float] = []
            calibration_started_at = time.perf_counter()
            calibrated = self.noise_calibration_seconds <= 0.0
            if calibrated:
                self.on_status("Listening (remote pose)")
            else:
                self.on_status("Calibrating microphone noise")
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=blocksize,
                dtype="int16",
                channels=1,
                latency=self._sounddevice_latency(),
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

                    samples_i16 = np.frombuffer(raw_audio, dtype=np.int16)
                    if samples_i16.size == 0:
                        continue
                    samples_f32 = samples_i16.astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(np.square(samples_f32))))
                    now = time.perf_counter()

                    if not calibrated:
                        calibration_values.append(rms)
                        if now - calibration_started_at < self.noise_calibration_seconds:
                            continue
                        if calibration_values:
                            noise_mean = float(np.mean(calibration_values))
                            noise_p95 = float(np.percentile(calibration_values, 95))
                            effective_vad_threshold = max(
                                self.vad_threshold,
                                noise_mean * 3.0,
                                noise_p95 * 1.8,
                            )
                            print(
                                "[Remote Speech Pose] noise_calibration "
                                f"mean={noise_mean:.4f} p95={noise_p95:.4f} "
                                f"threshold={effective_vad_threshold:.4f}"
                            )
                        calibrated = True
                        self.on_status("Listening (remote pose)")
                        continue

                    speaking = rms >= effective_vad_threshold
                    if speaking:
                        speech_run_blocks += 1
                    else:
                        speech_run_blocks = 0

                    if speaking and not phrase_chunks:
                        if speech_run_blocks < self.speech_start_blocks:
                            continue
                        phrase_started_at = now
                        self.on_interim("Listening...")
                    if speaking or phrase_chunks:
                        phrase_chunks.append(samples_i16.copy())
                    if speaking:
                        last_voice_at = now
                        phrase_voiced_samples += samples_i16.size
                        phrase_voiced_blocks += 1
                        phrase_peak_rms = max(phrase_peak_rms, rms)

                    phrase_age = now - phrase_started_at if phrase_started_at else 0.0
                    silence_age = now - last_voice_at if last_voice_at else 0.0
                    if phrase_chunks and silence_age >= self.silence_seconds:
                        flush_phrase("silence")
                    elif phrase_chunks and phrase_age >= self.max_utterance_seconds:
                        flush_phrase("max-duration")

            flush_phrase("stop")
            self.on_status("Stopped")
        except Exception as exc:
            self.on_status(f"Remote pose audio error: {exc}")
            print(f"[Remote Speech Pose] exception: {exc}", file=sys.stderr)
        finally:
            self.active = False

    def _encode_wav(self, samples_i16: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(samples_i16.tobytes())
        return buffer.getvalue()

    def _sounddevice_latency(self):
        try:
            return float(self.audio_latency)
        except ValueError:
            return self.audio_latency

    def _estimate_wav_duration_ms(self, wav_bytes: bytes) -> float:
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
                frame_rate = max(1, int(wav.getframerate()))
                return (wav.getnframes() / frame_rate) * 1000.0
        except Exception:
            return 0.0

    def _header_float(self, response: requests.Response, name: str) -> Optional[float]:
        value = response.headers.get(name)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _request_url(self) -> str:
        if self.upload_format != "raw":
            return self.endpoint_url

        parsed = urllib.parse.urlsplit(self.endpoint_url)
        path = parsed.path.rstrip("/")
        if path in {"", "/", "/speech-to-pose"}:
            path = "/speech-to-pose-raw"
        elif path.endswith("/speech-to-pose"):
            path = path[: -len("/speech-to-pose")] + "/speech-to-pose-raw"
        return urllib.parse.urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                path,
                parsed.query,
                parsed.fragment,
            )
        )

    def _submit_request(self, wav_bytes: bytes, reason: str) -> bool:
        with self.pending_lock:
            if self.pending_requests >= self.max_pending_requests:
                self.on_status("Remote pose busy; dropped an old utterance")
                return False
            self.pending_requests += 1

        self.on_status("Sending audio to Speech-to-Pose server")
        self.executor.submit(self._request_worker, wav_bytes, reason)
        return True

    def _request_worker(self, wav_bytes: bytes, reason: str) -> None:
        try:
            sequence = self._request_pose(wav_bytes)
        except NoSpeechDetected:
            self.on_status("Listening (remote pose)")
            return
        except Exception as exc:
            self.on_status(f"Remote pose error: {exc}")
            print(f"[Remote Speech Pose] request failed ({reason}): {exc}", file=sys.stderr)
            return
        finally:
            with self.pending_lock:
                self.pending_requests = max(0, self.pending_requests - 1)

        self.on_pose(sequence)
        self.on_status("Listening (remote pose)")

    def _request_pose(self, wav_bytes: bytes) -> pose_client.PoseSequence:
        audio_duration_ms = self._estimate_wav_duration_ms(wav_bytes)
        audio_kb = len(wav_bytes) / 1024.0
        print(
            "[Remote Speech Pose] sending_audio "
            f"format={self.upload_format} sample_rate={self.sample_rate} "
            f"audio_ms={audio_duration_ms:.0f} audio_kb={audio_kb:.1f}"
        )

        request_url = self._request_url()
        params = {"language": self.language}
        headers = {"Accept-Encoding": "gzip"}
        started_at = time.perf_counter()
        if self.upload_format == "raw":
            response = self.session.post(
                request_url,
                params=params,
                data=wav_bytes,
                headers={**headers, "Content-Type": "audio/wav"},
                timeout=self.timeout,
            )
        else:
            response = self.session.post(
                request_url,
                data=params,
                files={"audio": ("speech.wav", wav_bytes, "audio/wav")},
                headers=headers,
                timeout=self.timeout,
        )
        http_ms = (time.perf_counter() - started_at) * 1000.0
        if response.status_code == 204:
            print(
                "[Remote Speech Pose] no_speech "
                f"http_ms={http_ms:.1f} "
                f"server_total_ms={response.headers.get('X-Total-MS', '?')} "
                f"asr_ms={response.headers.get('X-ASR-MS', '?')}"
            )
            raise NoSpeechDetected()
        if not response.ok:
            snippet = response.text[:300].strip()
            raise RuntimeError(f"HTTP {response.status_code}: {snippet}")

        content_type = response.headers.get("content-type", "application/pose")
        pose_bytes = response.content
        if "application/json" in content_type:
            payload = response.json()
            pose_b64 = str(payload.get("pose_base64") or "")
            if not pose_b64:
                raise RuntimeError("remote server returned JSON without pose_base64")
            pose_bytes = base64.b64decode(pose_b64)
            transcript = str(payload.get("text") or "Speech")
            content_type = str(payload.get("content_type") or "application/pose")
        else:
            transcript = response.headers.get("X-Transcript", "Speech")

        server_total_ms = self._header_float(response, "X-Total-MS")
        asgi_ms = self._header_float(response, "X-ASGI-MS")
        upload_parse_ms = self._header_float(response, "X-Upload-Parse-MS")
        content_length = self._header_float(response, "Content-Length")
        wire_kb = content_length / 1024.0 if content_length is not None else None
        wire_kb_text = f"{wire_kb:.1f}" if wire_kb is not None else "?"
        pose_kb = len(pose_bytes) / 1024.0
        original_pose_bytes = self._header_float(response, "X-Pose-Original-Bytes")
        original_pose_kb = original_pose_bytes / 1024.0 if original_pose_bytes else pose_kb
        pose_ratio = self._header_float(response, "X-Pose-Size-Ratio")
        pose_ratio_text = f"{pose_ratio:.3f}" if pose_ratio is not None else "?"
        transport_baseline_ms = asgi_ms if asgi_ms is not None else server_total_ms
        client_transport_ms = (
            max(0.0, http_ms - transport_baseline_ms)
            if transport_baseline_ms is not None
            else None
        )
        client_transport_text = (
            f"{client_transport_ms:.1f}" if client_transport_ms is not None else "?"
        )

        print(
            "[Remote Speech Pose] "
            f"format={self.upload_format} "
            f"sample_rate={self.sample_rate} "
            f"audio_ms={audio_duration_ms:.0f} "
            f"audio_kb={audio_kb:.1f} "
            f"pose_kb={pose_kb:.1f} "
            f"original_pose_kb={original_pose_kb:.1f} "
            f"pose_ratio={pose_ratio_text} "
            f"wire_kb={wire_kb_text} "
            f"http_ms={http_ms:.1f} "
            f"server_total_ms={response.headers.get('X-Total-MS', '?')} "
            f"asr_ms={response.headers.get('X-ASR-MS', '?')} "
            f"signmt_ms={response.headers.get('X-Sign-MT-MS', '?')} "
            f"upload_parse_ms={response.headers.get('X-Upload-Parse-MS', '?')} "
            f"client_transport_ms={client_transport_text} "
            f"pose_fps={response.headers.get('X-Pose-Original-FPS', '?')}->{response.headers.get('X-Pose-FPS', '?')} "
            f"pose_frames={response.headers.get('X-Pose-Original-Frames', '?')}->{response.headers.get('X-Pose-Frames', '?')} "
            f"downsample={response.headers.get('X-Pose-Downsample', '?')} "
            f"encoding={response.headers.get('Content-Encoding', 'identity')} "
            f"cache={response.headers.get('X-Pose-Cache', '?')}"
        )

        pose = pose_client.parse_pose_binary(pose_bytes)
        fps = float(pose.body.fps or 25.0)
        frame_count = int(pose.body.frame_count or 0)
        if frame_count <= 0:
            raise RuntimeError("remote pose has no frames")

        text = pose_client.normalize_subtitle_text(transcript) or "Speech"
        return pose_client.PoseSequence(
            text=text,
            pose=pose,
            fps=fps,
            frame_count=frame_count,
            duration=frame_count / fps,
            content_type=content_type,
        )


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
        self.closed = False
        self.overlay_photo: Optional[tk.PhotoImage] = None
        self.overlay_photo_size: tuple[int, int] = (0, 0)
        self.overlay_alpha = max(0, min(255, int(getattr(args, "overlay_alpha", 178))))
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
        self.asr = RemoteSpeechPoseClient(
            endpoint_url=args.remote_pose_url,
            language=args.remote_pose_language,
            sample_rate=args.sample_rate,
            vad_threshold=args.remote_vad_threshold,
            silence_seconds=args.remote_silence_seconds,
            min_utterance_seconds=args.remote_min_utterance_seconds,
            min_speech_seconds=args.remote_min_speech_seconds,
            speech_start_blocks=args.remote_speech_start_blocks,
            min_peak_rms=args.remote_min_peak_rms,
            send_threshold_margin=args.remote_send_threshold_margin,
            min_voiced_ratio=args.remote_min_voiced_ratio,
            max_utterance_seconds=args.remote_max_utterance_seconds,
            timeout=args.remote_pose_timeout,
            audio_block_ms=args.remote_audio_block_ms,
            audio_latency=args.remote_audio_latency,
            noise_calibration_seconds=args.remote_noise_calibration_seconds,
            max_pending_requests=args.remote_max_pending_requests,
            upload_format=args.remote_pose_upload_format,
            on_pose=lambda sequence: self.safe_after(lambda: self.enqueue_pose_sequence(sequence, "remote-speech")),
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
            self.overlay_photo = make_translucent_overlay(overlay_width, overlay_height, alpha=self.overlay_alpha)
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

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.generation += 1
        self.asr.stop()
        self.renderer.stop(resolve=False)

    def safe_after(self, callback: Callable[[], None]) -> None:
        self.bridge.post(callback)

    def enqueue_pose_sequence(
        self,
        sequence: pose_client.PoseSequence,
        source: str,
    ) -> Optional[pose_client.QueueItem]:
        text = (
            pose_client.normalize_pipeline_text(sequence.text, self.args.remote_pose_language)
            or sequence.text
            or "Speech"
        )
        item = pose_client.QueueItem(
            item_id=self.next_id,
            text=text,
            source=source,
            status="ready",
            status_text="Ready",
            sequence=sequence,
            ready_at=time.perf_counter(),
        )
        self.next_id += 1
        self.items.append(item)
        self.api_status = "Ready"
        self.interim_transcript = ""
        self.render_subtitles()
        self.pump_playback()
        return item

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

        self.bridge = TkEventBridge(self.root, poll_ms=int(self.dashboard_args.event_poll_ms))
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
        self.pose_args.overlay_alpha = self.dashboard_args.overlay_alpha
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
            render_format=str(self.dashboard_args.camera_render_format),
            display_max_width=int(self.dashboard_args.camera_display_max_width),
            display_max_height=int(self.dashboard_args.camera_display_max_height),
            overlay_alpha=int(self.dashboard_args.overlay_alpha),
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


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    client_args, pose_args, dashboard_args = parse_dual_args(argv)

    app = DualTranslatorDashboard(client_args, pose_args, dashboard_args)

    def handle_signal(_signum, _frame) -> None:
        app.close()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    app.run()


if __name__ == "__main__":
    main()
