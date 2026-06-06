from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import queue
import signal
import shutil
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tempfile
import wave

import cv2
import numpy as np
import requests


DEFAULT_BASE_URL = os.environ.get("OPENSLT_REALTIME_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_VIDEO_PATH = os.environ.get("OPENSLT_VIDEO_PATH", "").strip()
DEFAULT_CAMERA_INDEX = int(os.environ.get("OPENSLT_CAMERA_INDEX", "0"))
DEFAULT_CAMERA_FPS = int(os.environ.get("OPENSLT_CAMERA_FPS", "30"))
DEFAULT_CAMERA_WARMUP_SECONDS = float(os.environ.get("OPENSLT_CAMERA_WARMUP_SECONDS", "1.2"))
DEFAULT_SAMPLE_FPS = int(os.environ.get("OPENSLT_SAMPLE_FPS", "15"))
DEFAULT_CHUNK_FRAMES = int(os.environ.get("OPENSLT_CHUNK_FRAMES", "15"))
DEFAULT_WIDTH = int(os.environ.get("OPENSLT_FRAME_WIDTH", "640"))
DEFAULT_HEIGHT = int(os.environ.get("OPENSLT_FRAME_HEIGHT", "480"))
DEFAULT_JPEG_QUALITY = int(os.environ.get("OPENSLT_JPEG_QUALITY", "72"))
DEFAULT_UPLOAD_WORKERS = int(os.environ.get("OPENSLT_UPLOAD_WORKERS", "3"))
DEFAULT_MAX_PENDING_UPLOADS = int(os.environ.get("OPENSLT_MAX_PENDING_UPLOADS", "6"))
DEFAULT_CHUNK_ENDPOINT = os.environ.get("OPENSLT_CHUNK_ENDPOINT", "/api/chunk_early").strip() or "/api/chunk_early"
DEFAULT_REQUEST_TIMEOUT = float(os.environ.get("OPENSLT_REQUEST_TIMEOUT", "10"))
DEFAULT_SESSION_WARMUP_SECONDS = float(os.environ.get("OPENSLT_SESSION_WARMUP_SECONDS", "3"))
DEFAULT_DURATION_SECONDS = float(os.environ.get("OPENSLT_DURATION_SECONDS", "0"))
DEFAULT_SHOW_PREVIEW = os.environ.get("OPENSLT_SHOW_PREVIEW", "1").strip() not in {"0", "false", "False"}
DEFAULT_DEBUG_TIMING = os.environ.get("OPENSLT_DEBUG_TIMING", "0").strip() not in {"0", "false", "False"}
DEFAULT_SERVER_STATS_INTERVAL = int(os.environ.get("OPENSLT_SERVER_STATS_INTERVAL", "0"))
DEFAULT_DIAG_DUMP_FRAMES = int(os.environ.get("OPENSLT_DIAG_DUMP_FRAMES", "0"))
DEFAULT_DIAG_DUMP_DIR = os.environ.get("OPENSLT_DIAG_DUMP_DIR", "").strip()
DEFAULT_SPEAK = os.environ.get("OPENSLT_SPEAK", "1").strip() not in {"0", "false", "False"}
DEFAULT_SPEECH_MODE = os.environ.get("OPENSLT_SPEECH_MODE", "final").strip().lower()
DEFAULT_SPEECH_BACKEND = os.environ.get("OPENSLT_SPEECH_BACKEND", "auto").strip().lower()
DEFAULT_SPEECH_APLAY_DEVICE = os.environ.get(
    "OPENSLT_SPEECH_APLAY_DEVICE",
    os.environ.get("OPENSLT_AUDIO_DEVICE", ""),
).strip()
DEFAULT_SPEECH_VOICE = os.environ.get("OPENSLT_SPEECH_VOICE", "").strip()
DEFAULT_SPEECH_RATE = int(os.environ.get("OPENSLT_SPEECH_RATE", "175"))
DEFAULT_SPEECH_MIN_WORDS = int(os.environ.get("OPENSLT_SPEECH_MIN_WORDS", "2"))
DEFAULT_SPEECH_MIN_INTERVAL_SECONDS = float(os.environ.get("OPENSLT_SPEECH_MIN_INTERVAL_SECONDS", "0.45"))
DEFAULT_SPEECH_STABILITY_SECONDS = float(os.environ.get("OPENSLT_SPEECH_STABILITY_SECONDS", "0.35"))
DEFAULT_SPEECH_STABLE_REPEATS = int(os.environ.get("OPENSLT_SPEECH_STABLE_REPEATS", "2"))
DEFAULT_SPEECH_COMMAND_TIMEOUT = float(os.environ.get("OPENSLT_SPEECH_COMMAND_TIMEOUT", "12"))
DEFAULT_SPEECH_DEBUG = os.environ.get("OPENSLT_SPEECH_DEBUG", "0").strip() not in {"0", "false", "False"}
DEFAULT_SPEECH_TEST_TEXT = os.environ.get("OPENSLT_SPEECH_TEST_TEXT", "").strip()
DEFAULT_AUDIO_CUES = os.environ.get("OPENSLT_AUDIO_CUES", "1").strip() not in {"0", "false", "False"}
DEFAULT_AUDIO_CUE_BACKEND = os.environ.get("OPENSLT_AUDIO_CUE_BACKEND", "auto").strip().lower()
DEFAULT_AUDIO_CUE_DEVICE = os.environ.get(
    "OPENSLT_AUDIO_CUE_DEVICE",
    os.environ.get("OPENSLT_AUDIO_DEVICE", ""),
).strip()
DEFAULT_AUDIO_CUE_VOLUME = float(os.environ.get("OPENSLT_AUDIO_CUE_VOLUME", "0.35"))
DEFAULT_AUDIO_CUE_DEBUG = os.environ.get("OPENSLT_AUDIO_CUE_DEBUG", "0").strip() not in {"0", "false", "False"}
DEFAULT_AUDIO_CUE_MIN_INTERVAL_SECONDS = float(os.environ.get("OPENSLT_AUDIO_CUE_MIN_INTERVAL_SECONDS", "1.0"))
DEFAULT_AUDIO_DROP_GRACE_SECONDS = float(os.environ.get("OPENSLT_AUDIO_DROP_GRACE_SECONDS", "1.2"))
DEFAULT_AUDIO_HIGH_LATENCY_MS = float(os.environ.get("OPENSLT_AUDIO_HIGH_LATENCY_MS", "6000"))
DEFAULT_AUDIO_HIGH_LATENCY_INTERVAL_SECONDS = float(
    os.environ.get("OPENSLT_AUDIO_HIGH_LATENCY_INTERVAL_SECONDS", "12")
)
DEFAULT_AUDIO_UTTERANCE_STARTED_CUE = (
    os.environ.get("OPENSLT_AUDIO_UTTERANCE_STARTED_CUE", "0").strip() not in {"0", "false", "False"}
)


@dataclass
class ChunkPayload:
    chunk_index: int
    start_ts_ms: float
    end_ts_ms: float
    frame_bytes_list: List[bytes]
    frame_sizes_bytes: List[int]
    sampled_frame_count: int
    camera_read_seconds_total: float
    resize_seconds_total: float
    encode_seconds_total: float
    loop_count: int
    chunk_bytes_total: int
    brightness_mean_avg: float
    brightness_std_avg: float
    queue_depth_after_enqueue: int = 0
    enqueued_at_perf: float = 0.0


@dataclass
class SpeechRequest:
    text: str
    replace_current: bool
    priority: int


@dataclass
class CueRequest:
    name: str
    replace_current: bool
    priority: int


class AudioCueOutput:
    _PATTERNS: Dict[str, List[Tuple[float, float]]] = {
        "session_ready": [(660.0, 0.07), (0.0, 0.035), (880.0, 0.09)],
        "retry_drop": [(880.0, 0.065), (0.0, 0.035), (660.0, 0.065), (0.0, 0.035), (440.0, 0.09)],
        "network_down": [(300.0, 0.12), (0.0, 0.08), (300.0, 0.12)],
        "network_recovered": [(440.0, 0.06), (0.0, 0.035), (660.0, 0.08)],
        "high_latency_wait": [(520.0, 0.06)],
        "utterance_started": [(700.0, 0.04)],
        "session_stopped": [(660.0, 0.07), (0.0, 0.035), (440.0, 0.09)],
        "warmup_waiting": [(500.0, 0.04)],
    }

    def __init__(self, args: argparse.Namespace):
        self.enabled = bool(args.audio_cues)
        self.backend_name = str(args.audio_cue_backend).strip().lower()
        self.aplay_device = str(args.audio_cue_device).strip()
        self.volume = max(0.0, min(1.0, float(args.audio_cue_volume)))
        self.debug = bool(args.audio_cue_debug)
        self.default_min_interval_seconds = max(0.0, float(args.audio_cue_min_interval_seconds))
        self.high_latency_interval_seconds = max(1.0, float(args.audio_high_latency_interval_seconds))
        self._queue: "queue.Queue[Optional[CueRequest]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._closed = threading.Event()
        self._current_process: Optional[subprocess.Popen[str]] = None
        self._current_process_lock = threading.Lock()
        self._last_played_at: Dict[str, float] = {}
        self._last_played_lock = threading.Lock()
        self._backend_resolution_error = ""
        self._resolved_backend = self._resolve_backend()
        self._tmpdir: Optional[tempfile.TemporaryDirectory] = None
        self._cue_paths: Dict[str, Path] = {}
        self._cue_durations: Dict[str, float] = {
            name: sum(duration for _frequency, duration in pattern)
            for name, pattern in self._PATTERNS.items()
        }
        if self.enabled and self._resolved_backend is None:
            detail = f": {self._backend_resolution_error}" if self._backend_resolution_error else ""
            print(f"[audio] cues disabled: no supported cue backend found{detail}")
            self.enabled = False

    def start(self) -> None:
        if not self.enabled or self._worker is not None:
            return
        if self._resolved_backend != "bell":
            self._prepare_cue_files()
        self._worker = threading.Thread(target=self._worker_loop, name="audio-cues", daemon=True)
        self._worker.start()
        backend_path = shutil.which(str(self._resolved_backend)) if self._resolved_backend else None
        print(
            f"[audio] cues enabled backend={self._resolved_backend} "
            f"volume={self.volume:.2f} device={self.aplay_device or 'default'} "
            f"backend_path={backend_path or 'n/a'}"
        )

    def close(self) -> None:
        if not self.enabled or self._worker is None:
            return
        self._closed.set()
        self._stop_current_process()
        self._queue.put(None)
        self._worker.join(timeout=2.0)
        self._worker = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def play(
        self,
        name: str,
        *,
        replace_current: bool = False,
        priority: int = 1,
        min_interval_seconds: Optional[float] = None,
    ) -> None:
        if not self.enabled or self._worker is None:
            return
        if name not in self._PATTERNS:
            if self.debug:
                print(f"[audio] cue skipped unknown name={name}")
            return

        interval = self._event_min_interval(name, min_interval_seconds)
        now = time.perf_counter()
        with self._last_played_lock:
            last_played = self._last_played_at.get(name, 0.0)
            if now - last_played < interval:
                return
            self._last_played_at[name] = now

        if replace_current:
            self._clear_pending_requests()
            self._stop_current_process()
        self._queue.put(CueRequest(name=name, replace_current=replace_current, priority=priority))

    def _event_min_interval(self, name: str, override: Optional[float]) -> float:
        if override is not None:
            return max(0.0, float(override))
        if name == "high_latency_wait":
            return self.high_latency_interval_seconds
        if name in {"network_down", "network_recovered"}:
            return max(2.0, self.default_min_interval_seconds)
        return self.default_min_interval_seconds

    def _worker_loop(self) -> None:
        while not self._closed.is_set():
            try:
                request = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if request is None:
                break
            try:
                self._play_cue(request.name)
            except Exception as exc:
                if self.debug:
                    print(f"[audio] cue playback failed name={request.name}: {exc}")

    def _play_cue(self, name: str) -> None:
        backend = self._resolved_backend
        if backend is None:
            return
        if self.debug:
            print(f"[audio] play cue={name} backend={backend}")
        if backend == "bell":
            sys.stdout.write("\a")
            sys.stdout.flush()
            time.sleep(self._cue_durations.get(name, 0.08))
            return

        path = self._cue_paths.get(name)
        if path is None:
            return
        command = self._build_command(backend, path)
        process: Optional[subprocess.Popen[str]] = None
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            with self._current_process_lock:
                self._current_process = process
            _stdout, stderr = process.communicate(timeout=3.0)
            if process.returncode != 0 and self.debug:
                print(f"[audio] cue command rc={process.returncode} stderr={stderr.strip()}")
        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
                process.communicate(timeout=1.0)
            if self.debug:
                print(f"[audio] cue command timed out cue={name}")
        finally:
            with self._current_process_lock:
                if self._current_process is process:
                    self._current_process = None

    def _prepare_cue_files(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="openslt_audio_cues_")
        cue_dir = Path(self._tmpdir.name)
        for name, pattern in self._PATTERNS.items():
            path = cue_dir / f"{name}.wav"
            self._write_tone_wav(path, pattern)
            self._cue_paths[name] = path

    def _write_tone_wav(self, path: Path, pattern: List[Tuple[float, float]]) -> None:
        sample_rate = 22050
        max_amplitude = int(32767 * self.volume)
        fade_samples = max(1, int(sample_rate * 0.005))
        samples = bytearray()

        for frequency, duration_seconds in pattern:
            sample_count = max(1, int(sample_rate * duration_seconds))
            for sample_index in range(sample_count):
                if frequency <= 0.0:
                    value = 0
                else:
                    t = sample_index / sample_rate
                    fade_in = min(1.0, sample_index / fade_samples)
                    fade_out = min(1.0, (sample_count - sample_index - 1) / fade_samples)
                    envelope = min(fade_in, fade_out)
                    value = int(max_amplitude * envelope * math.sin(2.0 * math.pi * frequency * t))
                samples.extend(struct.pack("<h", value))

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(bytes(samples))

    def _resolve_backend(self) -> Optional[str]:
        if not self.enabled:
            return None
        supported = {"auto", "aplay", "paplay", "pw-play", "ffplay", "bell"}
        if self.backend_name not in supported:
            self._backend_resolution_error = (
                "--audio-cue-backend must be one of: auto, aplay, paplay, pw-play, ffplay, bell"
            )
            return None
        if self.backend_name == "bell":
            return "bell"
        if self.backend_name != "auto":
            if shutil.which(self.backend_name):
                return self.backend_name
            self._backend_resolution_error = f"audio cue backend '{self.backend_name}' was requested but not found in PATH"
            return None
        for candidate in ("aplay", "paplay", "pw-play", "ffplay"):
            if shutil.which(candidate):
                return candidate
        return "bell"

    def _build_command(self, backend: str, path: Path) -> List[str]:
        path_text = str(path)
        if backend == "aplay":
            command = ["aplay", "-q"]
            if self.aplay_device:
                command.extend(["-D", self.aplay_device])
            command.append(path_text)
            return command
        if backend == "paplay":
            return ["paplay", path_text]
        if backend == "pw-play":
            return ["pw-play", path_text]
        if backend == "ffplay":
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path_text]
        raise RuntimeError(f"unsupported audio cue backend: {backend}")

    def _stop_current_process(self) -> None:
        with self._current_process_lock:
            process = self._current_process
            self._current_process = None
        if process is None or process.poll() is not None:
            return
        process.kill()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass

    def _clear_pending_requests(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return


class LocalSpeechOutput:
    def __init__(self, args: argparse.Namespace):
        self.enabled = bool(args.speak)
        self.mode = str(args.speech_mode).strip().lower()
        if self.mode == "hybrid":
            self.mode = "balanced"
        self.backend_name = str(args.speech_backend).strip().lower()
        self.aplay_device = str(args.speech_aplay_device).strip()
        self.voice = str(args.speech_voice).strip()
        self.rate = int(args.speech_rate)
        self.min_words = max(1, int(args.speech_min_words))
        self.min_interval_seconds = max(0.0, float(args.speech_min_interval_seconds))
        self.stability_seconds = max(0.0, float(args.speech_stability_seconds))
        self.stable_repeats = max(1, int(args.speech_stable_repeats))
        self.command_timeout = max(1.0, float(args.speech_command_timeout))
        self.debug = bool(args.speech_debug)
        self.test_text = str(args.speech_test_text).strip()
        self._queue: "queue.Queue[Optional[SpeechRequest]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._current_process: Optional[subprocess.Popen[str]] = None
        self._current_process_lock = threading.Lock()
        self._stopped_process_ids: set[int] = set()
        self._closed = threading.Event()
        self._last_announced_text = ""
        self._last_spoken_final = ""
        self._last_submit_at = 0.0
        self._draft_observed_text = ""
        self._draft_observed_since = 0.0
        self._draft_repeat_count = 0
        self._last_spoken_draft_basis = ""
        self._backend_resolution_error = ""
        self._resolved_backend = self._resolve_backend()
        if self.enabled and self.mode not in {"final", "balanced"}:
            raise ValueError("--speech-mode must be one of: final, balanced")
        if self.enabled and self._resolved_backend is None:
            detail = f": {self._backend_resolution_error}" if self._backend_resolution_error else ""
            print(
                "[speech] disabled: no supported local TTS backend found "
                "(looked for espeak-ng/espeak/spd-say on Linux, 'say' on macOS, PowerShell SAPI on Windows)"
                f"{detail}"
            )
            self.enabled = False

    def start(self) -> None:
        if not self.enabled or self._worker is not None:
            return
        self._worker = threading.Thread(target=self._worker_loop, name="speech-output", daemon=True)
        self._worker.start()
        backend_path = shutil.which(str(self._resolved_backend)) if self._resolved_backend else None
        print(
            f"[speech] enabled backend={self._resolved_backend} mode={self.mode} "
            f"rate={self.rate} min_words={self.min_words} "
            f"stable_after={self.stability_seconds:.2f}s repeats={self.stable_repeats} "
            f"aplay_device={self.aplay_device or 'default'} backend_path={backend_path or 'n/a'}"
        )
        if self.test_text:
            self._submit(
                SpeechRequest(text=self.test_text, replace_current=True, priority=3),
                announced_text=self.test_text,
            )

    def close(self) -> None:
        if not self.enabled or self._worker is None:
            return
        self._closed.set()
        self._stop_current_process()
        self._queue.put(None)
        self._worker.join(timeout=2.0)
        self._worker = None

    def handle_server_payload(self, *, draft: str, final: str) -> None:
        if not self.enabled:
            return

        final_text = self._normalize_text(final)
        draft_text = self._normalize_text(draft)

        if final_text and final_text != self._last_spoken_final:
            self._handle_final_text(final_text)
            return

        if self.mode != "balanced" or not draft_text:
            return

        self._handle_balanced_draft(draft_text)

    def _handle_final_text(self, final_text: str) -> None:
        self._reset_draft_tracking()
        text_to_speak = self._build_candidate_text(
            previous=self._last_announced_text,
            current=final_text,
            allow_full_rewrite=True,
            min_words=1,
        )
        self._last_spoken_final = final_text
        if not text_to_speak:
            self._last_announced_text = final_text
            if self.debug:
                print(f"[speech] final skipped because it was already announced: '{final_text}'")
            return
        if self.debug:
            print(f"[speech] enqueue final text='{text_to_speak}' basis='{final_text}'")
        self._submit(
            SpeechRequest(text=text_to_speak, replace_current=True, priority=2),
            announced_text=final_text,
        )

    def _handle_balanced_draft(self, draft_text: str) -> None:
        now = time.perf_counter()
        if draft_text != self._draft_observed_text:
            self._draft_observed_text = draft_text
            self._draft_observed_since = now
            self._draft_repeat_count = 1
            return

        self._draft_repeat_count += 1
        if draft_text == self._last_spoken_draft_basis:
            return
        if now - self._last_submit_at < self.min_interval_seconds:
            return

        stable_for = now - self._draft_observed_since
        if self._draft_repeat_count < self.stable_repeats and stable_for < self.stability_seconds:
            return

        allow_full_rewrite = self._should_treat_as_new_utterance(
            previous=self._last_announced_text,
            current=draft_text,
        )
        text_to_speak = self._build_candidate_text(
            previous=self._last_announced_text,
            current=draft_text,
            allow_full_rewrite=allow_full_rewrite,
            min_words=self.min_words,
        )
        if not text_to_speak:
            if self.debug:
                print(f"[speech] draft stable but skipped by prefix/min_words basis='{draft_text}'")
            return

        if self.debug:
            print(f"[speech] enqueue draft text='{text_to_speak}' basis='{draft_text}'")
        self._submit(
            SpeechRequest(text=text_to_speak, replace_current=False, priority=1),
            announced_text=draft_text,
        )
        self._last_spoken_draft_basis = draft_text

    def _reset_draft_tracking(self) -> None:
        self._draft_observed_text = ""
        self._draft_observed_since = 0.0
        self._draft_repeat_count = 0
        self._last_spoken_draft_basis = ""

    def _should_treat_as_new_utterance(self, *, previous: str, current: str) -> bool:
        if not previous or not current:
            return False
        if previous != self._last_spoken_final:
            return False
        previous_words = previous.split()
        current_words = current.split()
        if not previous_words or not current_words:
            return False
        return previous_words[0].lower() != current_words[0].lower()

    def _submit(self, request: SpeechRequest, *, announced_text: str) -> None:
        self._clear_pending_requests()
        if request.replace_current:
            self._stop_current_process()
        self._queue.put(request)
        self._last_announced_text = announced_text
        self._last_submit_at = time.perf_counter()

    def _worker_loop(self) -> None:
        while not self._closed.is_set():
            try:
                request = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if request is None:
                break
            try:
                self._speak(request.text)
            except Exception as exc:
                print(f"[speech] playback failed: {exc}")

    def _speak(self, text: str) -> None:
        if self.aplay_device and self._resolved_backend in {"espeak-ng", "espeak"}:
            self._speak_espeak_via_aplay(text)
            return

        command, stdin_text = self._build_command(text)
        process: Optional[subprocess.Popen[str]] = None
        try:
            if self.debug:
                print(f"[speech] running command={' '.join(command)} text='{text}'")
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if stdin_text is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            with self._current_process_lock:
                self._current_process = process
            stdout, stderr = process.communicate(input=stdin_text, timeout=self.command_timeout)
            if self.debug:
                if stdout.strip():
                    print(f"[speech] stdout: {stdout.strip()}")
                if stderr.strip():
                    print(f"[speech] stderr: {stderr.strip()}")
                print(f"[speech] command exited returncode={process.returncode}")
            if process.returncode != 0:
                if process.returncode < 0 and (self._closed.is_set() or self._was_process_stopped(process)):
                    if self.debug:
                        print("[speech] command stopped because a newer utterance replaced it")
                    return
                detail = stderr.strip() or stdout.strip() or "no command output"
                raise RuntimeError(f"speech command failed rc={process.returncode}: {detail}")
        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
                process.communicate(timeout=1.0)
            raise RuntimeError("speech command timed out")
        finally:
            with self._current_process_lock:
                if self._current_process is process:
                    self._current_process = None

    def _speak_espeak_via_aplay(self, text: str) -> None:
        backend = self._resolved_backend
        if backend not in {"espeak-ng", "espeak"}:
            raise RuntimeError("speech wav bridge requires espeak-ng or espeak")
        if not shutil.which("aplay"):
            raise RuntimeError("OPENSLT_SPEECH_APLAY_DEVICE was set but aplay was not found")

        tmp_file = tempfile.NamedTemporaryFile(prefix="openslt_speech_", suffix=".wav", delete=False)
        wav_path = tmp_file.name
        tmp_file.close()
        try:
            synth_command = [backend, "--stdin", "-w", wav_path, "-s", str(self.rate)]
            if self.voice:
                synth_command.extend(["-v", self.voice])
            self._run_speech_command(synth_command, stdin_text=text, label="synthesize")

            play_command = ["aplay", "-q", "-D", self.aplay_device, wav_path]
            self._run_speech_command(play_command, stdin_text=None, label="play")
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def _run_speech_command(self, command: List[str], *, stdin_text: Optional[str], label: str) -> None:
        process: Optional[subprocess.Popen[str]] = None
        try:
            if self.debug:
                print(f"[speech] {label} command={' '.join(command)}")
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if stdin_text is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            with self._current_process_lock:
                self._current_process = process
            stdout, stderr = process.communicate(input=stdin_text, timeout=self.command_timeout)
            if self.debug:
                if stdout.strip():
                    print(f"[speech] {label} stdout: {stdout.strip()}")
                if stderr.strip():
                    print(f"[speech] {label} stderr: {stderr.strip()}")
                print(f"[speech] {label} command exited returncode={process.returncode}")
            if process.returncode != 0:
                if process.returncode < 0 and (self._closed.is_set() or self._was_process_stopped(process)):
                    if self.debug:
                        print(f"[speech] {label} command stopped because a newer utterance replaced it")
                    return
                detail = stderr.strip() or stdout.strip() or "no command output"
                raise RuntimeError(f"speech {label} command failed rc={process.returncode}: {detail}")
        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
                process.communicate(timeout=1.0)
            raise RuntimeError(f"speech {label} command timed out")
        finally:
            with self._current_process_lock:
                if self._current_process is process:
                    self._current_process = None

    def _stop_current_process(self) -> None:
        with self._current_process_lock:
            process = self._current_process
            self._current_process = None
            if process is not None:
                self._stopped_process_ids.add(id(process))
        if process is None or process.poll() is not None:
            return
        process.kill()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass

    def _was_process_stopped(self, process: subprocess.Popen[str]) -> bool:
        process_id = id(process)
        with self._current_process_lock:
            if process_id not in self._stopped_process_ids:
                return False
            self._stopped_process_ids.discard(process_id)
            return True

    def _clear_pending_requests(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _resolve_backend(self) -> Optional[str]:
        if not self.enabled:
            return None
        if self.backend_name != "auto":
            if sys.platform.startswith("linux") and self.backend_name not in {"espeak-ng", "espeak", "spd-say"}:
                self._backend_resolution_error = f"unsupported Linux speech backend '{self.backend_name}'"
                return None
            if sys.platform.startswith("linux") and not shutil.which(self.backend_name):
                self._backend_resolution_error = f"speech backend '{self.backend_name}' was requested but not found in PATH"
                return None
            return self.backend_name
        if sys.platform.startswith("linux"):
            for candidate in ("espeak-ng", "espeak", "spd-say"):
                if shutil.which(candidate):
                    return candidate
            self._backend_resolution_error = "install espeak-ng, espeak, or speech-dispatcher/spd-say"
            return None
        if sys.platform == "darwin":
            return "say" if shutil.which("say") else None
        if os.name == "nt":
            return "powershell-sapi"
        return None

    def _build_command(self, text: str) -> Tuple[List[str], Optional[str]]:
        backend = self._resolved_backend
        if backend is None:
            raise RuntimeError("speech backend unavailable")

        if backend in {"espeak-ng", "espeak"}:
            command = [backend, "--stdin", "-s", str(self.rate)]
            if self.voice:
                command.extend(["-v", self.voice])
            return command, text

        if backend == "spd-say":
            # spd-say uses a relative speed range; map a speech-rate-like value into it.
            speed_percent = max(-100, min(100, int(round((self.rate - 175) * 100 / 175))))
            command = ["spd-say", "-r", str(speed_percent)]
            if self.voice:
                command.extend(["-o", self.voice])
            command.append(text)
            return command, None

        if backend == "say":
            command = ["say", "-r", str(self.rate)]
            if self.voice:
                command.extend(["-v", self.voice])
            command.append(text)
            return command, None

        if backend == "powershell-sapi":
            speech_rate = max(-10, min(10, int(round((self.rate - 175) / 20.0))))
            voice_clause = ""
            if self.voice:
                escaped_voice = self.voice.replace("'", "''")
                voice_clause = f"try {{ $s.SelectVoice('{escaped_voice}') }} catch {{}}; "
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Rate = {speech_rate}; "
                + voice_clause
                + "$text = [Console]::In.ReadToEnd(); "
                "if (-not [string]::IsNullOrWhiteSpace($text)) { $s.Speak($text) }"
            )
            return ["powershell", "-NoProfile", "-Command", script], text

        raise RuntimeError(f"unsupported speech backend: {backend}")

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text).strip().split())

    def _build_candidate_text(
        self,
        *,
        previous: str,
        current: str,
        allow_full_rewrite: bool,
        min_words: int,
    ) -> str:
        previous_words = previous.split()
        current_words = current.split()
        prefix_len = 0
        while (
            prefix_len < len(previous_words)
            and prefix_len < len(current_words)
            and previous_words[prefix_len].lower() == current_words[prefix_len].lower()
        ):
            prefix_len += 1

        if prefix_len == len(current_words):
            return ""

        if prefix_len == 0 and previous_words and not allow_full_rewrite:
            return ""

        candidate_words = current_words[prefix_len:] if prefix_len > 0 else current_words
        if len(candidate_words) < min_words:
            return ""

        candidate = " ".join(candidate_words).strip()
        if not candidate:
            return ""
        return candidate


class RealtimeChunkClient:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base_url = args.base_url.rstrip("/")
        self.chunk_endpoint = self._normalize_chunk_endpoint(str(args.chunk_endpoint))
        self.video_path = str(args.video_path).strip()
        self.sample_interval = 1.0 / max(args.sample_fps, 1)
        self.stop_event = threading.Event()
        self.session = requests.Session()
        self.session.headers.update({"Connection": "keep-alive"})
        self.session_id: Optional[str] = None
        self.chunk_index = 0
        self.frames_sent = 0
        self.chunks_sent = 0
        self.pending_upload_peak = 0
        self.dropped_chunks = 0
        self.inflight_uploads = 0
        self.inflight_upload_peak = 0
        self.cancelled_uploads = 0
        self.last_payload: Optional[Dict[str, object]] = None
        self.last_payload_updated_at_ms = 0.0
        self.cap = None
        self.sender_error: Optional[str] = None
        self.last_server_stats: Optional[Dict[str, Any]] = None
        self._state_lock = threading.Lock()
        self._preview_lock = threading.Lock()
        self._upload_thread_local = threading.local()
        self._upload_sessions: List[requests.Session] = []
        self._upload_sessions_lock = threading.Lock()
        self._pending_futures: set[concurrent.futures.Future[None]] = set()
        self.preview_status = "Khoi dong client"
        self.preview_detail = "Dang cho ket noi server realtime"
        self.preview_draft = ""
        self.preview_final = ""
        self.upload_workers = max(1, args.upload_workers)
        self.upload_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.diag_dump_frames = max(0, args.diag_dump_frames)
        self.diag_dump_dir = Path(args.diag_dump_dir) if args.diag_dump_dir else None
        self._diag_dumped_frames = 0
        self._diag_dump_lock = threading.Lock()
        self.speech_output = LocalSpeechOutput(args)
        self.audio_cues = AudioCueOutput(args)
        self._audio_state_lock = threading.Lock()
        self._network_down = False
        self._utterance_active = False
        self._utterance_final_seen = False
        self._utterance_last_active_at = 0.0
        self._utterance_started_cued = False
        self._last_audio_final = ""

    def run(self) -> None:
        self.audio_cues.start()
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

        try:
            self.session_id = self._create_session_with_warmup()
            print(f"[session] connected session_id={self.session_id}")
            self._update_preview_state(status="Connected", detail=f"Session {self.session_id[:8]} ready")
            self.audio_cues.play("session_ready", replace_current=True, priority=3)
            self.speech_output.start()
        except Exception as exc:
            print(f"[session] failed to create realtime session: {exc}")
            self.audio_cues.play("network_down", replace_current=True, priority=3)
            self.sender_error = str(exc)
            self._update_preview_state(status="Session error", detail=str(exc))
            self.stop_event.set()
            self._close_camera()
            self._print_summary(finalize_remote=False)
            self.speech_output.close()
            self.audio_cues.close()
            self.session.close()
            return

        self.upload_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.upload_workers,
            thread_name_prefix="chunk-sender",
        )

        current_chunk: List[bytes] = []
        current_chunk_start_ts_ms = 0.0
        last_sample_at = 0.0
        sampled_frame_count = 0
        camera_read_seconds_total = 0.0
        resize_seconds_total = 0.0
        encode_seconds_total = 0.0
        loop_count = 0
        chunk_bytes_total = 0
        chunk_frame_sizes: List[int] = []
        brightness_mean_total = 0.0
        brightness_std_total = 0.0
        run_started_at = time.perf_counter()

        try:
            while not self.stop_event.is_set():
                if self.args.duration_seconds > 0 and time.perf_counter() - run_started_at >= self.args.duration_seconds:
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
                if now - last_sample_at >= self.sample_interval:
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

                if self.args.show_preview:
                    preview = self._render_preview(frame, current_chunk_len=len(current_chunk))
                    cv2.imshow("OpenSLT RPi Client", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop_event.set()
                        break

                elapsed = time.perf_counter() - loop_start
                sleep_seconds = max(0.0, (1.0 / max(self.args.camera_fps, 1)) - elapsed)
                if sleep_seconds:
                    time.sleep(sleep_seconds)

            if current_chunk:
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
            uploads_drained = self._shutdown_uploads(
                wait_seconds=min(max(5.0, self.args.request_timeout + 2.0), 30.0)
            )
            self._close_camera()
            self._print_summary(finalize_remote=uploads_drained)
            self.speech_output.close()
            self.audio_cues.close()
            self._close_upload_sessions()
            self.session.close()

    def _open_camera(self) -> None:
        if self.video_path:
            source_path = Path(self.video_path).expanduser()
            if not source_path.exists():
                raise RuntimeError(f"Cannot open video path '{source_path}' because it does not exist")
            self.cap = cv2.VideoCapture(str(source_path))
            source_label = f"video={source_path}"
        else:
            self.cap = cv2.VideoCapture(self.args.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.args.camera_fps)
            source_label = f"camera_index={self.args.camera_index}"
        if not self.cap.isOpened():
            if self.video_path:
                raise RuntimeError(f"Cannot open video path '{self.video_path}'")
            raise RuntimeError(f"Cannot open camera index {self.args.camera_index}")
        actual_width = int(round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        actual_fps = round(float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0), 3)
        frame_count = int(round(float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
        backend_name = "unknown"
        if hasattr(self.cap, "getBackendName"):
            try:
                backend_name = str(self.cap.getBackendName())
            except cv2.error:
                backend_name = "unknown"
        print(
            f"[source] {source_label} requested={self.args.width}x{self.args.height}@{self.args.camera_fps} "
            f"actual={actual_width}x{actual_height}@{actual_fps} backend={backend_name} "
            f"frames={frame_count if frame_count > 0 else 'n/a'} jpeg_quality={self.args.jpeg_quality}"
        )
        if self.diag_dump_dir:
            self.diag_dump_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[diag] frame dumps enabled dir={self.diag_dump_dir} "
                f"limit={self.diag_dump_frames}"
            )
        self._warmup_camera()

    def _close_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.args.show_preview:
            cv2.destroyAllWindows()

    def _warmup_camera(self) -> None:
        warmup_seconds = max(0.0, float(self.args.camera_warmup_seconds))
        if warmup_seconds <= 0.0 or self.cap is None or self.video_path:
            return

        started_at = time.perf_counter()
        frames_read = 0
        last_mean = 0.0
        last_std = 0.0
        while time.perf_counter() - started_at < warmup_seconds:
            ok, frame = self.cap.read()
            if not ok:
                break
            frames_read += 1
            last_mean, last_std = self._compute_frame_diagnostics(frame)
            time.sleep(0.01)

        elapsed = time.perf_counter() - started_at
        print(
            f"[camera] warmup seconds={elapsed:.2f} frames={frames_read} "
            f"last_brightness_mean={last_mean:.3f} last_brightness_std={last_std:.3f}"
        )

    def _update_preview_state(
        self,
        *,
        status: Optional[str] = None,
        detail: Optional[str] = None,
        draft: Optional[str] = None,
        final: Optional[str] = None,
    ) -> None:
        with self._preview_lock:
            if status is not None:
                self.preview_status = status
            if detail is not None:
                self.preview_detail = detail
            if draft is not None:
                self.preview_draft = draft
            if final is not None:
                self.preview_final = final

    def _snapshot_preview_state(self) -> Tuple[str, str, str, str]:
        with self._preview_lock:
            return (
                self.preview_status,
                self.preview_detail,
                self.preview_draft,
                self.preview_final,
            )

    def _render_preview(self, frame, *, current_chunk_len: int):
        preview_frame = cv2.resize(frame, (self.args.preview_width, self.args.preview_height))
        panel_height = 240
        panel = np.full((panel_height, self.args.preview_width, 3), 20, dtype=np.uint8)
        status, detail, draft, final = self._snapshot_preview_state()
        with self._state_lock:
            pending_uploads = len(self._pending_futures)
            inflight_uploads = self.inflight_uploads
        current_fill = f"{current_chunk_len}/{self.args.chunk_frames}"
        session_short = self.session_id[:8] if self.session_id else "none"

        lines: List[Tuple[str, Tuple[int, int, int], float, int]] = [
            ("OpenSLT RPi Client Preview", (180, 220, 180), 0.72, 2),
            (f"Session: {session_short}", (220, 220, 220), 0.55, 1),
            (
                f"Inflight: {inflight_uploads}/{self.upload_workers} | Pending: {pending_uploads} | "
                f"Current chunk: {current_fill} | Sent: {self.chunks_sent}",
                (220, 220, 220),
                0.55,
                1,
            ),
            (f"Status: {status}", (120, 210, 255), 0.6, 2),
        ]

        y = 28
        for text, color, scale, thickness in lines:
            cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
            y += 28

        y = self._draw_wrapped_text(
            panel,
            f"Detail: {detail}",
            origin=(12, y + 4),
            max_width=self.args.preview_width - 24,
            line_height=22,
            color=(220, 220, 220),
            scale=0.5,
            thickness=1,
        )
        y += 8
        y = self._draw_wrapped_text(
            panel,
            f"Draft: {draft or '(empty)'}",
            origin=(12, y),
            max_width=self.args.preview_width - 24,
            line_height=22,
            color=(130, 255, 200),
            scale=0.52,
            thickness=1,
        )
        y += 8
        self._draw_wrapped_text(
            panel,
            f"Final: {final or '(empty)'}",
            origin=(12, y),
            max_width=self.args.preview_width - 24,
            line_height=22,
            color=(255, 230, 160),
            scale=0.52,
            thickness=1,
            max_lines=4,
        )

        return np.vstack([preview_frame, panel])

    @staticmethod
    def _draw_wrapped_text(
        image,
        text: str,
        *,
        origin: Tuple[int, int],
        max_width: int,
        line_height: int,
        color: Tuple[int, int, int],
        scale: float,
        thickness: int,
        max_lines: int = 3,
    ) -> int:
        x, y = origin
        words = text.split()
        lines: List[str] = []
        current = ""

        for word in words:
            candidate = word if not current else f"{current} {word}"
            width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
            if width <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = word
            if len(lines) >= max_lines:
                break

        if len(lines) < max_lines and current:
            lines.append(current)

        if words and len(lines) >= max_lines:
            remaining_words = words[len(" ".join(lines).split()):]
            if remaining_words:
                if lines:
                    lines[-1] = lines[-1].rstrip(".") + " ..."

        for line in lines[:max_lines]:
            cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
            y += line_height
        return y

    def _encode_frame(self, frame) -> Optional[bytes]:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
        )
        if not ok:
            return None
        return encoded.tobytes()

    def _submit_chunk(
        self,
        *,
        frame_bytes_list: List[bytes],
        frame_sizes_bytes: List[int],
        start_ts_ms: float,
        end_ts_ms: float,
        sampled_frame_count: int,
        camera_read_seconds_total: float,
        resize_seconds_total: float,
        encode_seconds_total: float,
        loop_count: int,
        chunk_bytes_total: int,
        brightness_mean_avg: float,
        brightness_std_avg: float,
    ) -> bool:
        chunk = ChunkPayload(
            chunk_index=self.chunk_index,
            start_ts_ms=start_ts_ms,
            end_ts_ms=end_ts_ms,
            frame_bytes_list=list(frame_bytes_list),
            frame_sizes_bytes=list(frame_sizes_bytes),
            sampled_frame_count=sampled_frame_count,
            camera_read_seconds_total=camera_read_seconds_total,
            resize_seconds_total=resize_seconds_total,
            encode_seconds_total=encode_seconds_total,
            loop_count=loop_count,
            chunk_bytes_total=chunk_bytes_total,
            brightness_mean_avg=brightness_mean_avg,
            brightness_std_avg=brightness_std_avg,
        )
        chunk.enqueued_at_perf = time.perf_counter()
        if self.sender_error:
            self._update_preview_state(status="Sender stopped", detail=self.sender_error)
            return False
        if self.upload_executor is None:
            self._update_preview_state(status="Sender stopped", detail="upload executor unavailable")
            return False

        max_pending_uploads = max(0, int(self.args.max_pending_uploads))
        if max_pending_uploads > 0:
            with self._state_lock:
                pending_count = len(self._pending_futures)
                if pending_count >= max_pending_uploads:
                    self.dropped_chunks += 1
                    self.pending_upload_peak = max(self.pending_upload_peak, pending_count)
                    dropped_total = self.dropped_chunks
            if pending_count >= max_pending_uploads:
                if self.args.debug_timing:
                    self._print_timing_record(
                        "chunk-dropped-backpressure",
                        {
                            "chunk_index": chunk.chunk_index,
                            "pending_uploads": pending_count,
                            "max_pending_uploads": max_pending_uploads,
                            "dropped_chunks": dropped_total,
                            "sampled_frames": chunk.sampled_frame_count,
                            "chunk_bytes": chunk.chunk_bytes_total,
                            "brightness_mean_avg": round(chunk.brightness_mean_avg, 3),
                            "brightness_std_avg": round(chunk.brightness_std_avg, 3),
                        },
                    )
                self.chunk_index += 1
                return True

        future = self.upload_executor.submit(self._send_chunk, chunk)
        with self._state_lock:
            self._pending_futures.add(future)
            chunk.queue_depth_after_enqueue = len(self._pending_futures)
            self.pending_upload_peak = max(self.pending_upload_peak, len(self._pending_futures))
        future.add_done_callback(self._handle_upload_future_done)

        if self.args.debug_timing:
            self._print_timing_record(
                "chunk-built",
                {
                    "chunk_index": chunk.chunk_index,
                    "sampled_frames": chunk.sampled_frame_count,
                    "camera_loops": chunk.loop_count,
                    "chunk_bytes": chunk.chunk_bytes_total,
                    "camera_read_ms_total": round(chunk.camera_read_seconds_total * 1000.0, 2),
                    "resize_ms_total": round(chunk.resize_seconds_total * 1000.0, 2),
                    "encode_ms_total": round(chunk.encode_seconds_total * 1000.0, 2),
                    "frame_bytes_min": min(chunk.frame_sizes_bytes) if chunk.frame_sizes_bytes else 0,
                    "frame_bytes_max": max(chunk.frame_sizes_bytes) if chunk.frame_sizes_bytes else 0,
                    "frame_bytes_mean": round(
                        chunk.chunk_bytes_total / max(len(chunk.frame_sizes_bytes), 1),
                        2,
                    ),
                    "frame_bytes_head": chunk.frame_sizes_bytes[:3],
                    "brightness_mean_avg": round(chunk.brightness_mean_avg, 3),
                    "brightness_std_avg": round(chunk.brightness_std_avg, 3),
                    "jpeg_quality": self.args.jpeg_quality,
                    "queue_depth_after_enqueue": chunk.queue_depth_after_enqueue,
                },
            )
        self.chunk_index += 1
        return True

    def _handle_upload_future_done(self, future: concurrent.futures.Future[None]) -> None:
        with self._state_lock:
            self._pending_futures.discard(future)
        if future.cancelled():
            with self._state_lock:
                self.cancelled_uploads += 1
            return
        exc = future.exception()
        if exc is None:
            return
        if self.sender_error:
            return
        self.sender_error = str(exc)
        print(f"[chunk] send failed: {exc}")
        self._mark_network_down(str(exc))
        self._update_preview_state(status="Send error", detail=str(exc))
        self.stop_event.set()

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
            self._update_preview_state(status="Warming", detail=status_label)
            if response.status_code >= 500 and not payload.get("warming", False):
                raise RuntimeError(status_label)
            time.sleep(self.args.session_warmup_seconds)

        raise RuntimeError("session creation interrupted")

    def _mark_network_down(self, reason: str) -> None:
        should_play = False
        with self._audio_state_lock:
            if not self._network_down:
                self._network_down = True
                should_play = True
        if should_play:
            if self.args.audio_cue_debug:
                print(f"[audio] network_down reason={reason}")
            self.audio_cues.play("network_down", replace_current=True, priority=3)

    def _mark_network_recovered(self) -> None:
        should_play = False
        with self._audio_state_lock:
            if self._network_down:
                self._network_down = False
                should_play = True
        if should_play:
            self.audio_cues.play("network_recovered", priority=2)

    def _handle_audio_latency(self, *, upload_seconds: float, queue_residence_seconds: float) -> None:
        threshold_ms = max(0.0, float(self.args.audio_high_latency_ms))
        if threshold_ms <= 0.0:
            return
        upload_ms = upload_seconds * 1000.0
        queue_ms = queue_residence_seconds * 1000.0
        if upload_ms >= threshold_ms or queue_ms >= threshold_ms:
            if self.args.audio_cue_debug:
                print(
                    f"[audio] high_latency upload_ms={upload_ms:.1f} "
                    f"queue_ms={queue_ms:.1f} threshold_ms={threshold_ms:.1f}"
                )
            self.audio_cues.play(
                "high_latency_wait",
                priority=1,
                min_interval_seconds=self.args.audio_high_latency_interval_seconds,
            )

    def _handle_audio_payload_state(self, *, payload: Dict[str, Any]) -> None:
        draft_text = LocalSpeechOutput._normalize_text(str(payload.get("draft_text") or ""))
        final_text = LocalSpeechOutput._normalize_text(str(payload.get("final_text") or ""))
        status = str(payload.get("status") or "").strip().lower()
        status_label = str(payload.get("status_label") or "").strip().lower()
        now = time.perf_counter()

        if final_text:
            with self._audio_state_lock:
                self._last_audio_final = final_text
                self._utterance_active = False
                self._utterance_final_seen = True
                self._utterance_last_active_at = 0.0
                self._utterance_started_cued = False
            return

        active_signal = self._is_utterance_active_signal(
            status=status,
            status_label=status_label,
            draft_text=draft_text,
        )
        play_started = False
        play_drop = False
        with self._audio_state_lock:
            if active_signal:
                if not self._utterance_active:
                    self._utterance_active = True
                    self._utterance_final_seen = False
                    self._utterance_started_cued = False
                    play_started = bool(self.args.audio_utterance_started_cue)
                self._utterance_last_active_at = now

            is_idle_empty = status == "listening" and not draft_text
            if (
                is_idle_empty
                and self._utterance_active
                and not self._utterance_final_seen
                and now - self._utterance_last_active_at >= self.args.audio_drop_grace_seconds
            ):
                play_drop = True
                self._utterance_active = False
                self._utterance_final_seen = False
                self._utterance_last_active_at = 0.0
                self._utterance_started_cued = False

            if play_started and self._utterance_started_cued:
                play_started = False
            elif play_started:
                self._utterance_started_cued = True

        if play_started:
            self.audio_cues.play("utterance_started", priority=1)
        if play_drop:
            self.audio_cues.play("retry_drop", replace_current=True, priority=3)

    @staticmethod
    def _is_utterance_active_signal(*, status: str, status_label: str, draft_text: str) -> bool:
        if draft_text:
            return True
        if status in {"waiting", "live", "collecting", "decoding", "finalizing"}:
            return True
        active_label_terms = (
            "xac nhan",
            "giu ngu canh",
            "cho tiep",
            "confirm",
            "context",
            "waiting",
            "hold",
        )
        return any(term in status_label for term in active_label_terms)

    def _get_upload_session(self) -> requests.Session:
        upload_session = getattr(self._upload_thread_local, "session", None)
        if upload_session is None:
            upload_session = requests.Session()
            upload_session.headers.update({"Connection": "keep-alive"})
            self._upload_thread_local.session = upload_session
            with self._upload_sessions_lock:
                self._upload_sessions.append(upload_session)
        return upload_session

    @staticmethod
    def _normalize_chunk_endpoint(value: str) -> str:
        endpoint = (value or "/api/chunk_early").strip()
        aliases = {
            "form": "/api/chunk",
            "default": "/api/chunk_early",
            "early": "/api/chunk_early",
            "stream": "/api/chunk_stream",
            "stream_body": "/api/chunk_stream",
        }
        endpoint = aliases.get(endpoint.lower(), endpoint)
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        return endpoint.rstrip("/")

    def _send_chunk(self, chunk: ChunkPayload) -> None:
        if not self.session_id:
            raise RuntimeError("missing realtime session id")

        session_id = self.session_id
        http_session = self._get_upload_session()
        queue_residence_seconds = max(0.0, time.perf_counter() - chunk.enqueued_at_perf)
        with self._state_lock:
            self.inflight_uploads += 1
            inflight_uploads = self.inflight_uploads
            self.inflight_upload_peak = max(self.inflight_upload_peak, self.inflight_uploads)
        files = [
            (
                "frames",
                (
                    f"chunk_{chunk.chunk_index}_{index}.jpg",
                    frame_bytes,
                    "image/jpeg",
                ),
            )
            for index, frame_bytes in enumerate(chunk.frame_bytes_list)
        ]
        data = {
            "chunk_index": str(chunk.chunk_index),
            "start_ts_ms": f"{chunk.start_ts_ms:.3f}",
            "end_ts_ms": f"{chunk.end_ts_ms:.3f}",
        }

        try:
            upload_started_at = time.perf_counter()
            response = http_session.post(
                f"{self.base_url}{self.chunk_endpoint}/{session_id}",
                data=data,
                files=files,
                timeout=self.args.request_timeout,
            )
            upload_seconds = time.perf_counter() - upload_started_at
            payload = self._safe_json(response)
            if not response.ok:
                raise RuntimeError(payload.get("detail") or f"chunk upload failed: HTTP {response.status_code}")

            self._mark_network_recovered()
            self._handle_audio_latency(
                upload_seconds=upload_seconds,
                queue_residence_seconds=queue_residence_seconds,
            )
            payload_ts = float(payload.get("updated_at_ms") or 0.0)
            with self._state_lock:
                self.chunks_sent += 1
                self.frames_sent += len(chunk.frame_bytes_list)
                should_apply_payload = payload_ts >= self.last_payload_updated_at_ms
                if should_apply_payload:
                    self.last_payload_updated_at_ms = payload_ts
                    self.last_payload = payload
            status = payload.get("status")
            label = payload.get("status_label")
            draft = payload.get("draft_text") or ""
            final = payload.get("final_text") or ""
            if should_apply_payload:
                self._update_preview_state(
                    status=str(status or "Streaming"),
                    detail=str(label or "Dang gui chunk"),
                    draft=str(draft),
                    final=str(final),
                )
                self.speech_output.handle_server_payload(
                    draft=str(draft),
                    final=str(final),
                )
                self._handle_audio_payload_state(payload=payload)
            if self.args.debug_timing:
                upload_mbps = 0.0
                if upload_seconds > 0:
                    upload_mbps = (chunk.chunk_bytes_total * 8.0) / (upload_seconds * 1_000_000.0)
                self._print_timing_record(
                    "chunk-uploaded",
                    {
                        "chunk_index": chunk.chunk_index,
                        "sampled_frames": chunk.sampled_frame_count,
                        "chunk_bytes": chunk.chunk_bytes_total,
                        "chunk_megabytes": round(chunk.chunk_bytes_total / (1024.0 * 1024.0), 3),
                        "camera_read_ms_total": round(chunk.camera_read_seconds_total * 1000.0, 2),
                        "resize_ms_total": round(chunk.resize_seconds_total * 1000.0, 2),
                        "encode_ms_total": round(chunk.encode_seconds_total * 1000.0, 2),
                        "frame_bytes_min": min(chunk.frame_sizes_bytes) if chunk.frame_sizes_bytes else 0,
                        "frame_bytes_max": max(chunk.frame_sizes_bytes) if chunk.frame_sizes_bytes else 0,
                        "frame_bytes_mean": round(
                            chunk.chunk_bytes_total / max(len(chunk.frame_sizes_bytes), 1),
                            2,
                        ),
                        "brightness_mean_avg": round(chunk.brightness_mean_avg, 3),
                        "brightness_std_avg": round(chunk.brightness_std_avg, 3),
                        "jpeg_quality": self.args.jpeg_quality,
                        "queue_residence_ms": round(queue_residence_seconds * 1000.0, 2),
                        "request_roundtrip_ms": round(upload_seconds * 1000.0, 2),
                        "effective_upload_mbps": round(upload_mbps, 3),
                        "queue_depth_after_enqueue": chunk.queue_depth_after_enqueue,
                        "inflight_uploads": inflight_uploads,
                        "response_status": response.status_code,
                        "status": status,
                        "status_label": label,
                        "draft_length": len(draft),
                        "final_length": len(final),
                    },
                )
            should_fetch_stats = False
            if self.args.server_stats_interval > 0:
                with self._state_lock:
                    should_fetch_stats = self.chunks_sent % self.args.server_stats_interval == 0
            if should_fetch_stats:
                try:
                    self._fetch_and_log_server_stats(http_session)
                except Exception as exc:
                    self._print_timing_record("server-stats-error", {"message": str(exc)})
            print(
                f"[chunk] sent={chunk.chunk_index} frames={len(chunk.frame_bytes_list)} "
                f"status={status} label={label} draft='{draft}' final='{final}'"
            )
        finally:
            with self._state_lock:
                self.inflight_uploads = max(0, self.inflight_uploads - 1)

    def finalize_remote_session(self) -> None:
        if not self.session_id:
            return

        try:
            response = self.session.post(
                f"{self.base_url}/api/session/{self.session_id}/stop",
                timeout=self.args.request_timeout,
            )
            payload = self._safe_json(response)
            if response.ok:
                self.last_payload = payload
                draft_text = str(payload.get("draft_text", ""))
                final_text = str(payload.get("final_text", ""))
                final_source = str(payload.get("final_source", ""))
                stop_stats = payload.get("stats")
                if isinstance(stop_stats, dict):
                    self.last_server_stats = stop_stats
                self._update_preview_state(
                    status="Stopped",
                    detail="Da chot cau cuoi",
                    draft=draft_text,
                    final=final_text,
                )
                self.speech_output.handle_server_payload(draft=draft_text, final=final_text)
                print(
                    f"[session] finalized final_text='{final_text}' "
                    f"draft_text='{draft_text}' final_source='{final_source}'"
                )
                if isinstance(stop_stats, dict) and self.args.debug_timing:
                    self._print_timing_record("server-stop-stats-final", stop_stats)
            else:
                print(f"[session] finalize failed: {payload}")
                self._update_preview_state(status="Stop error", detail=str(payload))
        except Exception as exc:
            print(f"[session] finalize request failed: {exc}")
            self._update_preview_state(status="Stop error", detail=str(exc))
        finally:
            self.session_id = None

    def _print_summary(self, *, finalize_remote: bool) -> None:
        if finalize_remote:
            self.finalize_remote_session()
        elif self.session_id:
            print("[summary] skipped remote finalize because uploads were still in flight at shutdown")
        print(
            f"[summary] chunks_sent={self.chunks_sent} "
            f"frames_sent={self.frames_sent} pending_upload_peak={self.pending_upload_peak} "
            f"inflight_upload_peak={self.inflight_upload_peak} dropped_chunks={self.dropped_chunks} "
            f"cancelled_uploads={self.cancelled_uploads}"
        )
        if self.last_payload:
            print(
                f"[summary] final_text='{self.last_payload.get('final_text', '')}' "
                f"draft_text='{self.last_payload.get('draft_text', '')}' "
                f"final_source='{self.last_payload.get('final_source', '')}'"
            )
        if self.sender_error:
            print(f"[summary] sender_error='{self.sender_error}'")
        if self.last_server_stats and self.args.debug_timing:
            self._print_timing_record("server-stats-final", self.last_server_stats)

    def _shutdown_uploads(self, *, wait_seconds: float) -> bool:
        if self.upload_executor is None:
            return True

        deadline = time.perf_counter() + max(0.0, wait_seconds)
        while time.perf_counter() < deadline:
            with self._state_lock:
                pending_futures = list(self._pending_futures)
            if not pending_futures:
                break
            if all(future.done() for future in pending_futures):
                break
            time.sleep(0.05)

        with self._state_lock:
            pending_futures = list(self._pending_futures)
        for future in pending_futures:
            if not future.running() and not future.done():
                future.cancel()

        while time.perf_counter() < deadline:
            with self._state_lock:
                remaining_futures = [future for future in self._pending_futures if not future.done()]
            if not remaining_futures:
                break
            time.sleep(0.05)

        with self._state_lock:
            remaining_futures = [future for future in self._pending_futures if not future.done()]

        if remaining_futures:
            print(
                f"[shutdown] upload drain timed out with {len(remaining_futures)} in-flight upload(s); "
                "skipping remote finalize"
            )
            self.upload_executor.shutdown(wait=False, cancel_futures=True)
            self.upload_executor = None
            return False

        self.upload_executor.shutdown(wait=True, cancel_futures=False)
        self.upload_executor = None
        return True

    def _close_upload_sessions(self) -> None:
        with self._upload_sessions_lock:
            sessions = list(self._upload_sessions)
            self._upload_sessions.clear()
        for upload_session in sessions:
            upload_session.close()

    @staticmethod
    def _compute_frame_diagnostics(frame: np.ndarray) -> Tuple[float, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_value, std_value = cv2.meanStdDev(gray)
        return float(mean_value[0][0]), float(std_value[0][0])

    def _maybe_dump_frame_sample(
        self,
        *,
        encoded: bytes,
        chunk_index: int,
        frame_index_in_chunk: int,
        frame_size: int,
        brightness_mean: float,
        brightness_std: float,
    ) -> None:
        if self.diag_dump_dir is None or self.diag_dump_frames <= 0:
            return
        with self._diag_dump_lock:
            if self._diag_dumped_frames >= self.diag_dump_frames:
                return
            dump_index = self._diag_dumped_frames
            self._diag_dumped_frames += 1
        filename = (
            f"frame_{dump_index:03d}_chunk_{chunk_index:04d}_idx_{frame_index_in_chunk:02d}"
            f"_size_{frame_size}_mean_{brightness_mean:.1f}_std_{brightness_std:.1f}.jpg"
        )
        try:
            (self.diag_dump_dir / filename).write_bytes(encoded)
        except OSError as exc:
            print(f"[diag] failed to dump frame sample '{filename}': {exc}")

    @staticmethod
    def _safe_json(response: requests.Response) -> Dict[str, object]:
        try:
            payload = response.json()
        except ValueError as exc:
            snippet = response.text[:200].strip()
            raise RuntimeError(
                f"backend did not return JSON (HTTP {response.status_code}): {snippet}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"unexpected response payload type: {type(payload).__name__}")
        return payload

    def _fetch_and_log_server_stats(self, http_session: Optional[requests.Session] = None) -> None:
        if not self.session_id:
            return

        active_session = http_session or self.session
        stats_response = active_session.get(
            f"{self.base_url}/api/session/{self.session_id}/stats",
            timeout=min(self.args.request_timeout, 30.0),
        )
        stats_payload = self._safe_json(stats_response)
        self.last_server_stats = stats_payload
        if self.args.debug_timing:
            compact_stats = {
                "session_id": stats_payload.get("session_id"),
                "inflight_upload_peak": self.inflight_upload_peak,
                "pending_upload_peak": self.pending_upload_peak,
                "raw_queue_depth": stats_payload.get("raw_queue_depth"),
                "raw_queue_depth_peak": stats_payload.get("raw_queue_depth_peak"),
                "prepared_queue_depth": stats_payload.get("prepared_queue_depth"),
                "prepared_queue_depth_peak": stats_payload.get("prepared_queue_depth_peak"),
                "chunks_received": stats_payload.get("chunks_received"),
                "chunks_coalesced": stats_payload.get("chunks_coalesced"),
                "chunks_processed": stats_payload.get("chunks_processed"),
                "prepared_chunks_processed": stats_payload.get("prepared_chunks_processed"),
                "frames_received_total": stats_payload.get("frames_received_total"),
                "frames_processed_total": stats_payload.get("frames_processed_total"),
                "frames_collectable_total": stats_payload.get("frames_collectable_total"),
                "frames_not_collectable_total": stats_payload.get("frames_not_collectable_total"),
                "chunk_receive_rate": stats_payload.get("chunk_receive_rate"),
                "chunk_process_rate": stats_payload.get("chunk_process_rate"),
                "frame_receive_rate": stats_payload.get("frame_receive_rate"),
                "frame_process_rate": stats_payload.get("frame_process_rate"),
                "frame_collectable_rate": stats_payload.get("frame_collectable_rate"),
                "ingest_parse_seconds_total": stats_payload.get("ingest_parse_seconds_total"),
                "ingest_enqueue_seconds_total": stats_payload.get("ingest_enqueue_seconds_total"),
                "ingest_endpoint_seconds_total": stats_payload.get("ingest_endpoint_seconds_total"),
                "raw_queue_wait_seconds_total": stats_payload.get("raw_queue_wait_seconds_total"),
                "prepared_queue_wait_seconds_total": stats_payload.get("prepared_queue_wait_seconds_total"),
                "prepared_enqueue_wait_seconds_total": stats_payload.get("prepared_enqueue_wait_seconds_total"),
                "jpeg_decode_seconds_total": stats_payload.get("jpeg_decode_seconds_total"),
                "mediapipe_chunk_seconds_total": stats_payload.get("mediapipe_chunk_seconds_total"),
                "gpu_chunk_seconds_total": stats_payload.get("gpu_chunk_seconds_total"),
                "preview_decode_seconds_total": stats_payload.get("preview_decode_seconds_total"),
                "final_decode_seconds_total": stats_payload.get("final_decode_seconds_total"),
                "preview_skipped_due_to_backpressure": stats_payload.get("preview_skipped_due_to_backpressure"),
                "preview_skipped_due_to_gpu_backpressure": stats_payload.get("preview_skipped_due_to_gpu_backpressure"),
                "last_status": stats_payload.get("last_status"),
                "last_status_label": stats_payload.get("last_status_label"),
            }
            self._print_timing_record("server-stats", compact_stats)

    @staticmethod
    def _print_timing_record(label: str, payload: Dict[str, Any]) -> None:
        print(f"[timing] {label} {json.dumps(payload, ensure_ascii=True, sort_keys=True)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Raspberry Pi client that streams chunked frames to OpenSLT realtime server."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenSLT realtime server base URL")
    parser.add_argument(
        "--video-path",
        default=DEFAULT_VIDEO_PATH,
        help="optional local video file to stream instead of a camera, useful for laptop/benchmark testing",
    )
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX, help="OpenCV camera index")
    parser.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS, help="camera capture FPS")
    parser.add_argument(
        "--camera-warmup-seconds",
        type=float,
        default=DEFAULT_CAMERA_WARMUP_SECONDS,
        help="drop camera frames for this many seconds before the first upload so auto exposure can settle",
    )
    parser.add_argument("--sample-fps", type=int, default=DEFAULT_SAMPLE_FPS, help="sample FPS sent to server")
    parser.add_argument("--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES, help="frames per upload chunk")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="capture width")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="capture height")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG quality 0-100")
    parser.add_argument(
        "--upload-workers",
        type=int,
        default=DEFAULT_UPLOAD_WORKERS,
        help="parallel chunk upload worker count; keep >=1",
    )
    parser.add_argument(
        "--max-pending-uploads",
        type=int,
        default=DEFAULT_MAX_PENDING_UPLOADS,
        help="drop newly captured chunks when this many uploads are already pending; 0 disables backpressure dropping",
    )
    parser.add_argument(
        "--chunk-endpoint",
        default=DEFAULT_CHUNK_ENDPOINT,
        help="chunk upload endpoint: early/default=/api/chunk_early, form=/api/chunk, stream=/api/chunk_stream",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--session-warmup-seconds",
        type=float,
        default=DEFAULT_SESSION_WARMUP_SECONDS,
        help="retry delay while backend runtime is warming",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help="stop automatically after this many seconds; 0 runs until interrupted",
    )
    parser.add_argument(
        "--show-preview",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW_PREVIEW,
        help="show local preview window",
    )
    parser.add_argument(
        "--debug-timing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DEBUG_TIMING,
        help="print detailed per-chunk timing logs",
    )
    parser.add_argument(
        "--server-stats-interval",
        type=int,
        default=DEFAULT_SERVER_STATS_INTERVAL,
        help="fetch /api/session/{id}/stats every N uploaded chunks; 0 disables",
    )
    parser.add_argument(
        "--diag-dump-frames",
        type=int,
        default=DEFAULT_DIAG_DUMP_FRAMES,
        help="dump the first N sampled JPEG frames for chunk-size diagnosis; 0 disables",
    )
    parser.add_argument(
        "--diag-dump-dir",
        default=DEFAULT_DIAG_DUMP_DIR,
        help="directory used by --diag-dump-frames; empty disables dumps",
    )
    parser.add_argument(
        "--speak",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SPEAK,
        help="speak recognized text locally through the client speaker",
    )
    parser.add_argument(
        "--speech-mode",
        choices=["final", "balanced", "hybrid"],
        default=DEFAULT_SPEECH_MODE,
        help="speech strategy: final only by default; balanced/hybrid can speak stable drafts for diagnostics",
    )
    parser.add_argument(
        "--speech-backend",
        default=DEFAULT_SPEECH_BACKEND,
        help="speech backend: auto, espeak-ng, espeak, spd-say, say, powershell-sapi",
    )
    parser.add_argument(
        "--speech-aplay-device",
        default=DEFAULT_SPEECH_APLAY_DEVICE,
        help="Linux ALSA device for espeak speech playback via aplay, e.g. plughw:2,0; empty uses backend default output",
    )
    parser.add_argument(
        "--speech-voice",
        default=DEFAULT_SPEECH_VOICE,
        help="optional voice name passed to the selected local speech backend",
    )
    parser.add_argument(
        "--speech-rate",
        type=int,
        default=DEFAULT_SPEECH_RATE,
        help="target speech rate; mapped to the selected local speech backend",
    )
    parser.add_argument(
        "--speech-min-words",
        type=int,
        default=DEFAULT_SPEECH_MIN_WORDS,
        help="minimum new words required before a draft update is spoken",
    )
    parser.add_argument(
        "--speech-min-interval-seconds",
        type=float,
        default=DEFAULT_SPEECH_MIN_INTERVAL_SECONDS,
        help="minimum time between spoken draft updates",
    )
    parser.add_argument(
        "--speech-stability-seconds",
        type=float,
        default=DEFAULT_SPEECH_STABILITY_SECONDS,
        help="how long a draft should stay unchanged before balanced mode speaks it",
    )
    parser.add_argument(
        "--speech-stable-repeats",
        type=int,
        default=DEFAULT_SPEECH_STABLE_REPEATS,
        help="how many identical draft payloads are needed before balanced mode speaks",
    )
    parser.add_argument(
        "--speech-command-timeout",
        type=float,
        default=DEFAULT_SPEECH_COMMAND_TIMEOUT,
        help="timeout in seconds for each local speech command",
    )
    parser.add_argument(
        "--speech-debug",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SPEECH_DEBUG,
        help="print local speech command diagnostics",
    )
    parser.add_argument(
        "--speech-test-text",
        default=DEFAULT_SPEECH_TEST_TEXT,
        help="optional text spoken once when local speech starts, useful for speaker diagnostics",
    )
    parser.add_argument(
        "--audio-cues",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUDIO_CUES,
        help="play short non-verbal cues for ready, retry/drop, network, and high-latency states",
    )
    parser.add_argument(
        "--audio-cue-backend",
        default=DEFAULT_AUDIO_CUE_BACKEND,
        help="audio cue backend: auto, aplay, paplay, pw-play, ffplay, bell",
    )
    parser.add_argument(
        "--audio-cue-device",
        default=DEFAULT_AUDIO_CUE_DEVICE,
        help="Linux ALSA device for aplay audio cues, e.g. plughw:2,0; empty uses default output",
    )
    parser.add_argument(
        "--audio-cue-volume",
        type=float,
        default=DEFAULT_AUDIO_CUE_VOLUME,
        help="audio cue volume from 0.0 to 1.0 for generated WAV tones",
    )
    parser.add_argument(
        "--audio-cue-debug",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUDIO_CUE_DEBUG,
        help="print audio cue diagnostics",
    )
    parser.add_argument(
        "--audio-cue-min-interval-seconds",
        type=float,
        default=DEFAULT_AUDIO_CUE_MIN_INTERVAL_SECONDS,
        help="minimum repeat interval for the same audio cue",
    )
    parser.add_argument(
        "--audio-drop-grace-seconds",
        type=float,
        default=DEFAULT_AUDIO_DROP_GRACE_SECONDS,
        help="seconds to wait before treating an active utterance returning to empty listening as dropped",
    )
    parser.add_argument(
        "--audio-high-latency-ms",
        type=float,
        default=DEFAULT_AUDIO_HIGH_LATENCY_MS,
        help="play high-latency cue when chunk roundtrip or queue wait exceeds this many ms; 0 disables",
    )
    parser.add_argument(
        "--audio-high-latency-interval-seconds",
        type=float,
        default=DEFAULT_AUDIO_HIGH_LATENCY_INTERVAL_SECONDS,
        help="minimum repeat interval for high-latency wait cue",
    )
    parser.add_argument(
        "--audio-utterance-started-cue",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUDIO_UTTERANCE_STARTED_CUE,
        help="optional assistive/debug cue when an utterance first becomes active; off by default",
    )
    parser.add_argument("--preview-width", type=int, default=320, help="preview width")
    parser.add_argument("--preview-height", type=int, default=240, help="preview height")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    client = RealtimeChunkClient(args)

    def _handle_signal(_signum, _frame) -> None:
        client.stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    client.run()


if __name__ == "__main__":
    main()
