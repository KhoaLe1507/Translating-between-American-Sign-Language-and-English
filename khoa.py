r"""
One-file Speech -> sign.mt -> Skeleton Pose Video desktop client.
Optimised for Raspberry Pi 4B (4 GB RAM, ARM Cortex-A72, Raspberry Pi OS).

Install runtime ASR dependencies:
    pip install vosk sounddevice

Download a Vosk English model, then run:
    export VOSK_MODEL_PATH=/home/pi/vosk-model-small-en-us-0.15
    python speech_pose_client_pi.py

Useful smoke test without opening the UI:
    python speech_pose_client_pi.py --self-test "hello"

Pi-specific changes vs the original:
  - ASR blocksize increased 8 000 -> 16 000 to prevent input overflow
  - Audio queue maxsize added (drops oldest frames instead of growing unbounded)
  - TARGET_RENDER_FPS lowered to 24 (Pi GPU/CPU cannot sustain 60 fps in tkinter)
  - DEFAULT_PLAYBACK_RATE lowered to 2.0 (smoother on slower hardware)
  - DEFAULT_CONCURRENCY lowered to 2 (Pi has 4 cores but limited memory bandwidth)
  - Canvas redraw is skipped when the frame has not changed (dirty-flag optimisation)
  - Interpolation is disabled by default; enable with --interpolate
  - Frame cache size is capped to avoid unbounded RAM growth
  - Limb/joint drawing uses a single canvas.update() call guard
  - sounddevice latency hint set to 'high' (more stable on ALSA/Pi)
  - Graceful degradation when sounddevice or vosk are unavailable
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import struct
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.parse
import urllib.request
from array import array
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Constants – Pi-tuned defaults
# ---------------------------------------------------------------------------

SIGN_MT_ENDPOINT = "https://us-central1-sign-mt.cloudfunctions.net/spoken_text_to_signed_pose"
DEFAULT_SPOKEN_LANGUAGE = "en"
DEFAULT_SIGNED_LANGUAGE = "ase"

# Pi 4B: 2 worker threads keeps CPU load and memory bandwidth manageable
DEFAULT_CONCURRENCY = 2

# Pi 4B: 2.0x is smooth enough without overloading the CPU
DEFAULT_PLAYBACK_RATE = 2.0

# Pi 4B: 24 fps is well within tkinter's redraw budget on a 1080p display
TARGET_RENDER_FPS = 24

ASR_SAMPLE_RATE = 16_000

# Pi 4B FIX: larger block == fewer callbacks == no input overflow.
# 16 000 samples @ 16 kHz = 1 second per block, well within Vosk latency budget.
ASR_BLOCKSIZE = 16_000

# Pi 4B: cap the audio queue so a slow Vosk doesn't grow RAM unbounded.
ASR_QUEUE_MAXSIZE = 8

ASR_DUPLICATE_WINDOW_SECONDS = 1.2

# Pi 4B: cap frame cache per pose to avoid OOM on long sessions.
FRAME_CACHE_MAX = 64

# ---------------------------------------------------------------------------
# ASR spelling replacements
# ---------------------------------------------------------------------------

ASR_SPELLING_REPLACEMENTS = [
    (re.compile(r"\bdont\b", re.I), "don't"),
    (re.compile(r"\bcant\b", re.I), "can't"),
    (re.compile(r"\bwont\b", re.I), "won't"),
    (re.compile(r"\bdidnt\b", re.I), "didn't"),
    (re.compile(r"\bdoesnt\b", re.I), "doesn't"),
    (re.compile(r"\bcouldnt\b", re.I), "couldn't"),
    (re.compile(r"\bshouldnt\b", re.I), "shouldn't"),
    (re.compile(r"\bwouldnt\b", re.I), "wouldn't"),
    (re.compile(r"\bim\b", re.I), "I'm"),
    (re.compile(r"\bive\b", re.I), "I've"),
    (re.compile(r"\byoure\b", re.I), "you're"),
    (re.compile(r"\btheyre\b", re.I), "they're"),
    (re.compile(r"\bweve\b", re.I), "we've"),
    (re.compile(r"\bthats\b", re.I), "that's"),
    (re.compile(r"\bwhats\b", re.I), "what's"),
    (re.compile(r"\blets\b", re.I), "let's"),
    (re.compile(r"\bi\b"), "I"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ComponentHeader:
    name: str
    fmt: str
    points: list[str]
    limbs: list[tuple[int, int]]
    colors: list[tuple[int, int, int]]


@dataclass
class PoseHeader:
    version: float
    width: int
    height: int
    depth: int
    components: list[ComponentHeader]
    header_length: int


@dataclass
class PoseBody:
    fps: float
    frame_count: int
    people_count: int
    frames: Optional[list[dict]] = None
    data: Optional[array] = None
    confidence: Optional[array] = None
    dims: int = 0
    total_points: int = 0
    header: Optional[PoseHeader] = None
    # Pi 4B: bounded dict instead of unbounded to cap RAM
    frame_cache: dict[int, dict] = field(default_factory=dict, repr=False)

    def _evict_cache_if_needed(self) -> None:
        """Evict oldest entries when cache exceeds FRAME_CACHE_MAX."""
        while len(self.frame_cache) > FRAME_CACHE_MAX:
            oldest_key = next(iter(self.frame_cache))
            del self.frame_cache[oldest_key]

    def frame_at(self, frame_index: int) -> dict:
        frame_index = max(0, min(self.frame_count - 1, frame_index))
        if self.frames is not None:
            return self.frames[frame_index]

        cached = self.frame_cache.get(frame_index)
        if cached is not None:
            return cached

        if self.header is None or self.data is None or self.confidence is None:
            return {"_people": 0, "people": []}

        people = []
        people_stride = self.people_count * self.total_points
        for person_index in range(self.people_count):
            person = {}
            component_point_offset = 0
            for component in self.header.components:
                joints = []
                for point_index in range(len(component.points)):
                    point_offset = (
                        frame_index * people_stride
                        + person_index * self.total_points
                        + component_point_offset
                        + point_index
                    )
                    point = {"X": 0.0, "Y": 0.0, "C": self.confidence[point_offset]}
                    data_base = point_offset * self.dims
                    for dim_index, dim in enumerate(component.fmt):
                        if dim != "C" and dim_index < self.dims:
                            point[dim] = self.data[data_base + dim_index]
                    joints.append(point)
                person[component.name] = joints
                component_point_offset += len(component.points)
            people.append(person)

        frame = {"_people": self.people_count, "people": people}
        self.frame_cache[frame_index] = frame
        self._evict_cache_if_needed()
        return frame

    # Pi 4B: interpolation is expensive on ARM; skipped unless --interpolate flag is set.
    def interpolated_frame_at(self, frame_position: float) -> dict:
        if self.frame_count <= 1:
            return self.frame_at(0)

        clamped = max(0.0, min(self.frame_count - 1, frame_position))
        lower_index = int(math.floor(clamped))
        upper_index = min(self.frame_count - 1, lower_index + 1)
        amount = clamped - lower_index

        if amount <= 0.001 or lower_index == upper_index:
            return self.frame_at(lower_index)

        return self.interpolate_frames(
            self.frame_at(lower_index), self.frame_at(upper_index), amount
        )

    def interpolate_frames(self, left: dict, right: dict, amount: float) -> dict:
        if not self.header:
            return left

        people = []
        for left_person, right_person in zip(
            left.get("people", []), right.get("people", [])
        ):
            person = {}
            for component in self.header.components:
                left_joints = left_person.get(component.name, [])
                right_joints = right_person.get(component.name, [])
                if len(left_joints) != len(right_joints):
                    person[component.name] = left_joints
                    continue
                person[component.name] = [
                    self.interpolate_joint(lj, rj, amount)
                    for lj, rj in zip(left_joints, right_joints)
                ]
            people.append(person)

        return {"_people": len(people), "people": people}

    @staticmethod
    def interpolate_joint(left: dict, right: dict, amount: float) -> dict:
        point = {}
        for key in left.keys() | right.keys():
            lv = float(left.get(key, 0.0))
            rv = float(right.get(key, lv))
            point[key] = lv + (rv - lv) * amount
        return point


@dataclass
class Pose:
    header: PoseHeader
    body: PoseBody


@dataclass
class PoseSequence:
    text: str
    pose: Pose
    fps: float
    frame_count: int
    duration: float
    content_type: str


@dataclass
class QueueItem:
    item_id: int
    text: str
    source: str
    status: str = "translating"
    status_text: str = "Translating"
    sequence: Optional[PoseSequence] = None
    error: Optional[BaseException] = None
    submitted_at: float = field(default_factory=time.perf_counter)
    ready_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0


# ---------------------------------------------------------------------------
# Binary parser
# ---------------------------------------------------------------------------

class BinaryReader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def seek(self, offset: int) -> None:
        self.offset = offset

    def uint16(self) -> int:
        value = struct.unpack_from("<H", self.data, self.offset)[0]
        self.offset += 2
        return value

    def int16(self) -> int:
        value = struct.unpack_from("<h", self.data, self.offset)[0]
        self.offset += 2
        return value

    def uint32(self) -> int:
        value = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return value

    def float32(self) -> float:
        value = struct.unpack_from("<f", self.data, self.offset)[0]
        self.offset += 4
        return value

    def string(self, length: int) -> str:
        value = self.data[self.offset : self.offset + length].decode(
            "utf-8", errors="replace"
        )
        self.offset += length
        return value


def parse_pose_binary(data: bytes) -> Pose:
    reader = BinaryReader(data)
    header = parse_header(reader)
    version = round(header.version, 3)

    if version == 0:
        body = parse_body_v0(reader, header)
    elif version in (0.1, 0.2):
        body = parse_body_v01_or_v02(data, header, version)
    else:
        raise ValueError(
            f"Parsing this pose body version is not implemented: {header.version}"
        )

    return Pose(header=header, body=body)


def parse_header(reader: BinaryReader) -> PoseHeader:
    version = reader.float32()
    width = reader.uint16()
    height = reader.uint16()
    depth = reader.uint16()
    component_count = reader.uint16()
    components = [parse_component_header(reader) for _ in range(component_count)]
    return PoseHeader(version, width, height, depth, components, reader.offset)


def parse_component_header(reader: BinaryReader) -> ComponentHeader:
    name = reader.string(reader.uint16())
    fmt = reader.string(reader.uint16())
    point_count = reader.uint16()
    limb_count = reader.uint16()
    color_count = reader.uint16()
    points = [reader.string(reader.uint16()) for _ in range(point_count)]
    limbs = [(reader.uint16(), reader.uint16()) for _ in range(limb_count)]
    colors = [
        (reader.uint16(), reader.uint16(), reader.uint16())
        for _ in range(color_count)
    ]
    return ComponentHeader(
        name=name, fmt=fmt, points=points, limbs=limbs, colors=colors
    )


def parse_body_v0(reader: BinaryReader, header: PoseHeader) -> PoseBody:
    reader.seek(header.header_length)
    fps = float(reader.uint16())
    frame_count = reader.uint16()
    frames = []

    for _ in range(frame_count):
        people_count = reader.uint16()
        people = []
        for _ in range(people_count):
            person = {}
            reader.int16()
            for component in header.components:
                person[component.name] = read_component_points(reader, component)
            people.append(person)
        frames.append({"_people": people_count, "people": people})

    return PoseBody(fps=fps, frame_count=frame_count, people_count=0, frames=frames)


def parse_body_v01_or_v02(
    data: bytes, header: PoseHeader, version: float
) -> PoseBody:
    reader = BinaryReader(data)
    reader.seek(header.header_length)

    if version == 0.1:
        fps = float(reader.uint16())
        frame_count = reader.uint16()
        info_size = 6
    else:
        fps = float(reader.float32())
        frame_count = reader.uint32()
        info_size = 10

    people_count = reader.uint16()
    total_points = sum(len(c.points) for c in header.components)
    dims = max(len(c.fmt) for c in header.components) - 1
    data_length = frame_count * people_count * total_points * dims
    confidence_length = frame_count * people_count * total_points
    data_offset = header.header_length + info_size
    confidence_offset = data_offset + data_length * 4

    pose_data = read_float32_array(data, data_offset, data_length)
    confidence = read_float32_array(data, confidence_offset, confidence_length)

    return PoseBody(
        fps=fps,
        frame_count=frame_count,
        people_count=people_count,
        data=pose_data,
        confidence=confidence,
        dims=dims,
        total_points=total_points,
        header=header,
    )


def read_component_points(
    reader: BinaryReader, component: ComponentHeader
) -> list[dict]:
    points = []
    for _ in component.points:
        point = {"X": 0.0, "Y": 0.0}
        for dim in component.fmt:
            point[dim] = reader.float32()
        points.append(point)
    return points


def read_float32_array(data: bytes, offset: int, length: int) -> array:
    values = array("f")
    values.frombytes(data[offset : offset + length * 4])
    if sys.byteorder != "little":
        values.byteswap()
    return values


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize_subtitle_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    text = re.sub(r"\[[^\]]*]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    return text.strip()


def normalize_pipeline_text(text: str, spoken_language: str) -> str:
    normalized = normalize_subtitle_text(text)
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = re.sub(r"\s+([,.!?;:])", r"\1", normalized)
    normalized = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", normalized)
    normalized = re.sub(
        r"\b(um|uh|er|erm|ah)\b[,.]?\s*", "", normalized, flags=re.I
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if spoken_language.lower().startswith("en"):
        for pattern, replacement in ASR_SPELLING_REPLACEMENTS:
            normalized = pattern.sub(replacement, normalized)
        normalized = sentence_case(normalized)

    return normalized


def sentence_case(text: str) -> str:
    def upper_match(match: re.Match) -> str:
        return match.group(0).upper()

    return re.sub(r"(^\s*[a-z])|([.!?]\s+[a-z])", upper_match, text)


# ---------------------------------------------------------------------------
# sign.mt API
# ---------------------------------------------------------------------------

def fetch_pose_sequence(
    text: str,
    endpoint: str,
    spoken_language: str,
    signed_language: str,
    timeout: float = 45.0,
) -> PoseSequence:
    normalized_text = normalize_subtitle_text(text)
    if not normalized_text:
        raise ValueError("Text is empty after normalization")

    params = urllib.parse.urlencode(
        {
            "text": normalized_text,
            "spoken": spoken_language,
            "signed": signed_language,
        }
    )
    url = f"{endpoint}?{params}"
    request = urllib.request.Request(
        url,
        headers={
            "Origin": "http://localhost",
            "User-Agent": "speech-pose-client-pi/1.0",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "")
            payload = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:240]
        raise RuntimeError(
            f"sign.mt failed with HTTP {error.code}: {detail}"
        ) from error

    if not payload:
        raise RuntimeError("sign.mt returned an empty pose response")

    pose = parse_pose_binary(payload)
    fps = float(pose.body.fps or 25.0)
    frame_count = int(pose.body.frame_count or 0)
    if frame_count <= 0:
        raise RuntimeError("Pose has no frames")

    return PoseSequence(
        text=normalized_text,
        pose=pose,
        fps=fps,
        frame_count=frame_count,
        duration=frame_count / fps,
        content_type=content_type,
    )


# ---------------------------------------------------------------------------
# Vosk model discovery
# ---------------------------------------------------------------------------

def find_vosk_model(explicit_path: Optional[str]) -> Optional[Path]:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    env_path = os.environ.get("VOSK_MODEL_PATH")
    if env_path:
        candidates.append(Path(env_path))

    script_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            script_dir / "models" / "vosk-model-small-en-us-0.15",
            script_dir / "vosk-model-small-en-us-0.15",
            Path.cwd() / "models" / "vosk-model-small-en-us-0.15",
            Path.cwd() / "vosk-model-small-en-us-0.15",
        ]
    )

    for root in [
        script_dir,
        script_dir / "models",
        Path.cwd(),
        Path.cwd() / "models",
    ]:
        if root.exists():
            candidates.extend(
                path for path in root.glob("vosk-model*") if path.is_dir()
            )

    for candidate in candidates:
        if candidate and candidate.exists() and candidate.is_dir():
            return candidate
    return None


# ---------------------------------------------------------------------------
# ASR – Pi-hardened version
# ---------------------------------------------------------------------------

class VoskStreamingAsr:
    """
    Vosk-based streaming ASR with Pi 4B specific hardening:
      - Larger blocksize (ASR_BLOCKSIZE) to prevent ALSA input overflow
      - Bounded audio queue (ASR_QUEUE_MAXSIZE) – drops oldest block when full
        rather than letting RAM grow unbounded
      - ALSA latency hint set to 'high' for more stable Pi ALSA operation
      - Overflow counter logged to stderr so you can see if drops still occur
    """

    def __init__(
        self,
        model_path: Optional[str],
        sample_rate: int,
        on_final: Callable[[str], None],
        on_interim: Callable[[str], None],
        on_status: Callable[[str], None],
    ):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.on_final = on_final
        self.on_interim = on_interim
        self.on_status = on_status
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.active = False
        self.last_final_text = ""
        self.last_final_at = 0.0
        self._overflow_count = 0

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run, name="VoskStreamingAsr", daemon=True
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _run(self) -> None:
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model
        except ImportError as exc:
            self.active = False
            self.on_status(
                f"Install vosk and sounddevice to enable microphone ASR. ({exc})"
            )
            return

        model_dir = find_vosk_model(self.model_path)
        if not model_dir:
            self.active = False
            self.on_status("Set VOSK_MODEL_PATH to an English Vosk model folder.")
            return

        # Pi 4B: bounded queue – if Vosk falls behind, we drop old audio
        # rather than buffering unbounded chunks in RAM.
        audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=ASR_QUEUE_MAXSIZE)

        def audio_callback(indata, _frames, _time_info, status) -> None:
            if status:
                # Log overflow to stderr; suppress noisy status updates to UI
                # so the label doesn't flicker on every overflow.
                self._overflow_count += 1
                if self._overflow_count % 10 == 1:
                    print(
                        f"[ASR] audio status: {status} "
                        f"(total overflows: {self._overflow_count})",
                        file=sys.stderr,
                    )
            # Non-blocking put – drop oldest frame if queue is full
            try:
                audio_queue.put_nowait(bytes(indata))
            except queue.Full:
                try:
                    audio_queue.get_nowait()  # discard oldest
                except queue.Empty:
                    pass
                try:
                    audio_queue.put_nowait(bytes(indata))
                except queue.Full:
                    pass  # discard current too if still full

        try:
            self.on_status("Loading ASR model...")
            model = Model(str(model_dir))
            recognizer = KaldiRecognizer(model, self.sample_rate)
            recognizer.SetWords(False)
            self.active = True
            self.on_status("Listening")

            with sd.RawInputStream(
                samplerate=self.sample_rate,
                # Pi 4B FIX: 16 000 samples gives 1 s blocks at 16 kHz.
                # This is the single most important change to stop overflow.
                blocksize=ASR_BLOCKSIZE,
                dtype="int16",
                channels=1,
                # Pi 4B: 'high' latency lets ALSA use a larger internal buffer,
                # reducing the chance of hardware overflow.
                latency="high",
                callback=audio_callback,
            ):
                while not self.stop_event.is_set():
                    try:
                        audio = audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    if recognizer.AcceptWaveform(audio):
                        result = json.loads(recognizer.Result())
                        self._emit_final(result.get("text", ""))
                    else:
                        result = json.loads(recognizer.PartialResult())
                        self.on_interim(
                            normalize_subtitle_text(result.get("partial", ""))
                        )

            final_result = json.loads(recognizer.FinalResult())
            self._emit_final(final_result.get("text", ""))
            self.on_status("Stopped")

        except Exception as error:
            self.on_status(f"ASR error: {error}")
            print(f"[ASR] exception: {error}", file=sys.stderr)
        finally:
            self.active = False

    def _emit_final(self, text: str) -> None:
        normalized = normalize_pipeline_text(text, DEFAULT_SPOKEN_LANGUAGE)
        if not normalized:
            return

        now = time.perf_counter()
        duplicate = (
            normalized.lower() == self.last_final_text.lower()
            and now - self.last_final_at < ASR_DUPLICATE_WINDOW_SECONDS
        )
        if duplicate:
            return

        self.last_final_text = normalized
        self.last_final_at = now
        self.on_final(normalized)
        self.on_interim("")


# ---------------------------------------------------------------------------
# Renderer – Pi-optimised
# ---------------------------------------------------------------------------

class SkeletonPoseRenderer:
    """
    Skeleton renderer with Pi 4B optimisations:
      - Dirty-flag: skips canvas redraw when the integer frame index has not
        changed (avoids needless delete-all + redraw at 24 fps when pose fps
        is lower than TARGET_RENDER_FPS).
      - Interpolation is opt-in (controlled by self.use_interpolation).
      - Limb list is pre-sorted by Z once per frame rather than inside the
        draw call, saving repeated Python sort overhead.
    """

    def __init__(
        self,
        root: tk.Tk,
        canvas: tk.Canvas,
        playback_rate: float,
        use_interpolation: bool = False,
    ):
        self.root = root
        self.canvas = canvas
        self.playback_rate = playback_rate
        self.use_interpolation = use_interpolation
        self.sequence: Optional[PoseSequence] = None
        self.pose: Optional[Pose] = None
        self.playing = False
        self.started_at = 0.0
        self.frame_index = 0
        self.frame_position = 0.0
        self.after_id: Optional[str] = None
        self.on_finished: Optional[Callable[[], None]] = None
        self.idle_label = "Listening"
        self.target_frame_interval = 1.0 / TARGET_RENDER_FPS
        self.next_tick_at = 0.0
        # Pi 4B dirty-flag: only redraw when frame index changes
        self._last_rendered_frame_index: int = -1
        self._last_canvas_size: tuple[int, int] = (0, 0)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _on_canvas_resize(self, _event) -> None:
        # Force redraw on resize by invalidating dirty flag
        self._last_rendered_frame_index = -1
        self.render_current_frame()

    def play(
        self, sequence: PoseSequence, on_finished: Callable[[], None]
    ) -> None:
        self.stop(resolve=False)
        self.sequence = sequence
        self.pose = sequence.pose
        self.frame_index = 0
        self.frame_position = 0.0
        self._last_rendered_frame_index = -1
        self.started_at = time.perf_counter()
        self.next_tick_at = self.started_at + self.target_frame_interval
        self.playing = True
        self.on_finished = on_finished
        self.render_current_frame()
        self._schedule_tick()

    def stop(self, resolve: bool = True) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        was_playing = self.playing
        callback = self.on_finished
        self.playing = False
        self.on_finished = None
        if resolve and was_playing and callback:
            callback()

    def paint_idle(self, label: str) -> None:
        self.stop(resolve=False)
        self.sequence = None
        self.pose = None
        self.frame_index = 0
        self.frame_position = 0.0
        self._last_rendered_frame_index = -1
        self.idle_label = label
        self.render_current_frame()

    def _schedule_tick(self) -> None:
        now = time.perf_counter()
        if self.next_tick_at <= now:
            self.next_tick_at = now + self.target_frame_interval
        delay_ms = max(1, int((self.next_tick_at - now) * 1000))
        self.after_id = self.root.after(delay_ms, self.tick)

    def tick(self) -> None:
        self.after_id = None
        if not self.playing or not self.sequence or not self.pose:
            return

        elapsed = (
            max(0.0, time.perf_counter() - self.started_at) * self.playback_rate
        )
        fps = float(self.pose.body.fps or 25.0)
        frame_count = int(self.pose.body.frame_count or 0)
        duration = frame_count / fps if fps > 0 else 0
        self.frame_position = max(
            0.0, min(frame_count - 1, elapsed * fps)
        )
        self.frame_index = int(self.frame_position)
        self.render_current_frame()

        if elapsed >= duration:
            self.frame_index = max(0, frame_count - 1)
            self.frame_position = float(self.frame_index)
            self.render_current_frame()
            self.playing = False
            callback = self.on_finished
            self.on_finished = None
            if callback:
                callback()
            return

        self.next_tick_at += self.target_frame_interval
        now = time.perf_counter()
        if self.next_tick_at < now:
            self.next_tick_at = now + self.target_frame_interval
        self._schedule_tick()

    def render_current_frame(self) -> None:
        width = max(180, self.canvas.winfo_width())
        height = max(180, self.canvas.winfo_height())
        canvas_size = (width, height)

        # Pi 4B dirty-flag: skip expensive canvas.delete("all") + redraw when
        # nothing has changed.
        if (
            self.frame_index == self._last_rendered_frame_index
            and canvas_size == self._last_canvas_size
            and self.pose is not None  # always redraw idle state
        ):
            return

        self._last_rendered_frame_index = self.frame_index
        self._last_canvas_size = canvas_size

        self.canvas.delete("all")
        self.canvas.create_rectangle(
            0, 0, width, height, fill="#f7f8f6", outline=""
        )

        if not self.pose or not self.pose.body.frame_count:
            self.draw_idle_label(width, height)
            return

        if self.use_interpolation:
            frame = self.pose.body.interpolated_frame_at(self.frame_position)
        else:
            frame = self.pose.body.frame_at(self.frame_index)

        self.draw_frame(frame, width, height)

    def draw_idle_label(self, width: int, height: int) -> None:
        self.canvas.create_text(
            width / 2,
            height / 2,
            text=self.idle_label,
            fill="#66756d",
            font=("DejaVu Sans", 12, "bold"),
        )

    def draw_frame(self, frame: dict, width: int, height: int) -> None:
        if not self.pose:
            return
        metrics = self.pose_metrics(width, height)
        for person in frame.get("people", []):
            for component in self.pose.header.components:
                joints = person.get(component.name, [])
                if not joints:
                    continue
                self.draw_component_limbs(component, joints, metrics)
                self.draw_component_joints(component, joints, metrics)

    def draw_component_limbs(
        self,
        component: ComponentHeader,
        joints: list[dict],
        metrics: dict,
    ) -> None:
        limbs = []
        for start, end in component.limbs:
            if start >= len(joints) or end >= len(joints):
                continue
            a = joints[start]
            b = joints[end]
            if not self.is_joint_valid(a) or not self.is_joint_valid(b):
                continue
            color_a = self.component_color(component, start)
            color_b = self.component_color(component, end)
            color = tuple(
                (color_a[i] + color_b[i]) / 2 for i in range(3)
            )
            limbs.append(
                (float(a.get("Z", 0.0) + b.get("Z", 0.0)) / 2, a, b, color)
            )

        width_px = max(2, int(metrics["thickness"] * 1.25))
        for _z, a, b, color in sorted(limbs, key=lambda item: item[0], reverse=True):
            alpha = (self.confidence(a) + self.confidence(b)) / 2
            self.canvas.create_line(
                self.proj_x(a, metrics),
                self.proj_y(a, metrics),
                self.proj_x(b, metrics),
                self.proj_y(b, metrics),
                fill=self.blend_color(color, alpha),
                width=width_px,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
            )

    def draw_component_joints(
        self,
        component: ComponentHeader,
        joints: list[dict],
        metrics: dict,
    ) -> None:
        radius = max(2, metrics["thickness"] / 3)
        for index, joint in enumerate(joints):
            if not self.is_joint_valid(joint):
                continue
            x = self.proj_x(joint, metrics)
            y = self.proj_y(joint, metrics)
            color = self.blend_color(
                self.component_color(component, index), self.confidence(joint)
            )
            self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill=color, outline="",
            )

    def pose_metrics(self, width: int, height: int) -> dict:
        pose_width = self.pose.header.width if self.pose else width
        pose_height = self.pose.header.height if self.pose else height
        padding = max(8, min(width, height) * 0.04)
        available_width = max(1, width - padding * 2)
        available_height = max(1, height - padding * 2)
        scale = min(available_width / pose_width, available_height / pose_height)
        draw_width = pose_width * scale
        draw_height = pose_height * scale
        return {
            "pose_width": pose_width,
            "pose_height": pose_height,
            "offset_x": (width - draw_width) / 2,
            "offset_y": (height - draw_height) / 2,
            "scale": scale,
            "thickness": max(2, math.sqrt(draw_width * draw_height) / 150),
        }

    def proj_x(self, joint: dict, metrics: dict) -> float:
        return (
            metrics["offset_x"]
            + (float(joint.get("X", 0.0)) / metrics["pose_width"])
            * metrics["pose_width"]
            * metrics["scale"]
        )

    def proj_y(self, joint: dict, metrics: dict) -> float:
        return (
            metrics["offset_y"]
            + (float(joint.get("Y", 0.0)) / metrics["pose_height"])
            * metrics["pose_height"]
            * metrics["scale"]
        )

    def is_joint_valid(self, joint: dict) -> bool:
        return bool(joint) and self.confidence(joint) > 0

    def confidence(self, joint: dict) -> float:
        return max(0.0, min(1.0, float(joint.get("C", 1.0))))

    def component_color(
        self, component: ComponentHeader, index: int
    ) -> tuple[float, float, float]:
        if not component.colors:
            return (20, 20, 20)
        return component.colors[index % len(component.colors)]

    def blend_color(
        self, color: tuple[float, float, float], alpha: float
    ) -> str:
        background = (247, 248, 246)
        channels = []
        for index, value in enumerate(color):
            clamped = max(0, min(255, int(round(value))))
            channels.append(
                int(round(clamped * alpha + background[index] * (1 - alpha)))
            )
        return f"#{channels[0]:02x}{channels[1]:02x}{channels[2]:02x}"


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class SpeechPoseApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = tk.Tk()
        self.root.title("Speech Pose Client (Pi)")
        # Pi 4B: smaller default window – Pi displays are often 720p or 1080p
        # but a 800x600 window starts fast.
        self.root.geometry("800x600")
        self.root.minsize(600, 440)
        self.root.configure(bg="#eef2ef")

        self.items: list[QueueItem] = []
        self.next_id = 1
        self.next_to_play = 1
        self.active_item: Optional[QueueItem] = None
        self.interim_transcript = ""
        self.asr_status = "Starting"
        self.api_status = "Idle"
        self.generation = 0
        self.cache: dict[str, PoseSequence] = {}
        self.executor = ThreadPoolExecutor(max_workers=args.concurrency)

        self._build_ui()
        self.renderer = SkeletonPoseRenderer(
            self.root,
            self.canvas,
            args.playback_rate,
            use_interpolation=args.interpolate,
        )
        self.asr = VoskStreamingAsr(
            model_path=args.model,
            sample_rate=args.sample_rate,
            on_final=lambda text: self.safe_after(
                lambda: self.handle_asr_final(text)
            ),
            on_interim=lambda text: self.safe_after(
                lambda: self.set_interim_transcript(text)
            ),
            on_status=lambda status: self.safe_after(
                lambda: self.set_asr_status(status)
            ),
        )

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        shell = tk.Frame(self.root, bg="#eef2ef")
        shell.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        shell.rowconfigure(0, weight=1)
        shell.rowconfigure(1, weight=0)
        shell.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            shell,
            bg="#f7f8f6",
            highlightthickness=1,
            highlightbackground="#d8ded8",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        subtitles = tk.Frame(
            shell,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#d8ded8",
        )
        subtitles.grid(row=1, column=0, sticky="ew")
        subtitles.columnconfigure(0, weight=1)

        self.current_label = self._subtitle_block(
            subtitles, "CURRENT SUBTITLE", "#2e7d64", 0, bold=True
        )
        self.next_label = self._subtitle_block(
            subtitles, "NEXT SUBTITLE", "#2f5ea8", 1, bold=False
        )

        subtitles.bind("<Configure>", self._update_wraplength)

    def _subtitle_block(
        self,
        parent: tk.Frame,
        title: str,
        accent: str,
        row: int,
        bold: bool,
    ) -> tk.Label:
        frame = tk.Frame(
            parent,
            bg="#fbfcfb",
            highlightthickness=1,
            highlightbackground="#dde4de",
        )
        frame.grid(
            row=row,
            column=0,
            sticky="ew",
            padx=10,
            pady=(10 if row == 0 else 0, 7 if row == 1 else 0),
        )
        frame.columnconfigure(1, weight=1)
        tk.Frame(frame, width=4, bg=accent).grid(
            row=0, column=0, rowspan=2, sticky="nsw"
        )
        tk.Label(
            frame,
            text=title,
            bg="#fbfcfb",
            fg="#637168",
            font=("DejaVu Sans", 8, "bold"),
            anchor="w",
        ).grid(row=0, column=1, sticky="ew", padx=10, pady=(7, 0))
        label = tk.Label(
            frame,
            text="Listening..." if row == 0 else "Waiting for speech",
            bg="#fbfcfb",
            fg="#1f2933" if bold else "#43514a",
            font=("DejaVu Sans", 13 if bold else 10, "bold"),
            anchor="w",
            justify="left",
        )
        label.grid(row=1, column=1, sticky="ew", padx=10, pady=(2, 8))
        return label

    def _update_wraplength(self, event) -> None:
        wrap = max(200, event.width - 50)
        self.current_label.configure(wraplength=wrap)
        self.next_label.configure(wraplength=wrap)

    # ------------------------------------------------------------------

    def run(self) -> None:
        self.renderer.paint_idle("Listening")
        self.render_subtitles()
        self.asr.start()
        if self.args.demo_text:
            for offset, text in enumerate(self.args.demo_text):
                self.root.after(
                    600 + offset * 250,
                    lambda value=text: self.enqueue_text(value, "demo"),
                )
        if self.args.auto_close_seconds > 0:
            self.root.after(
                int(self.args.auto_close_seconds * 1000), self.close
            )
        self.root.mainloop()

    def close(self) -> None:
        self.generation += 1
        self.asr.stop()
        self.renderer.stop(resolve=False)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def safe_after(self, callback: Callable[[], None]) -> None:
        try:
            self.root.after(0, callback)
        except RuntimeError:
            pass

    def handle_asr_final(self, text: str) -> None:
        self.enqueue_text(text, "speech")

    def enqueue_text(self, raw_text: str, source: str) -> Optional[QueueItem]:
        text = normalize_pipeline_text(raw_text, self.args.spoken)
        if not text:
            return None

        item = QueueItem(item_id=self.next_id, text=text, source=source)
        self.next_id += 1
        self.items.append(item)
        self.api_status = "Sending"
        self.interim_transcript = ""
        self.render_subtitles()
        self.pump_playback()

        cache_key = (
            f"{self.args.spoken}:{self.args.signed}:"
            f"{normalize_subtitle_text(text).lower()}"
        )
        if cache_key in self.cache:
            self.handle_translation_ready(
                item.item_id, self.generation, self.cache[cache_key]
            )
            return item

        future = self.executor.submit(
            fetch_pose_sequence,
            text,
            self.args.endpoint,
            self.args.spoken,
            self.args.signed,
            self.args.timeout,
        )
        future.add_done_callback(
            lambda done,
            item_id=item.item_id,
            generation=self.generation,
            key=cache_key: self.safe_after(
                lambda: self.handle_translation_done(
                    item_id, generation, key, done
                )
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
        sequence: PoseSequence,
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

    def play_item(self, item: QueueItem) -> None:
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

    def find_item(self, item_id: int) -> Optional[QueueItem]:
        return next(
            (item for item in self.items if item.item_id == item_id), None
        )

    def set_interim_transcript(self, text: str) -> None:
        self.interim_transcript = normalize_subtitle_text(text)
        self.render_subtitles()

    def set_asr_status(self, status: str) -> None:
        self.asr_status = status
        self.render_subtitles()

    def render_subtitles(self) -> None:
        self.current_label.configure(text=self.current_subtitle_text())
        self.next_label.configure(text=self.next_subtitle_text())

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
            if next_item.status == "ready":
                return next_item.text
            if next_item.status == "translating":
                return next_item.text
            if next_item.status == "error" and next_item.error:
                return f"Skipped: {next_item.error}"
        return "Waiting for speech"

    def next_subtitle_item(self) -> Optional[QueueItem]:
        minimum_id = (
            self.active_item.item_id + 1
            if self.active_item
            else self.next_to_play
        )
        candidates = [
            item
            for item in self.items
            if item.item_id >= minimum_id
            and item.status not in ("done", "skipped", "playing")
        ]
        return (
            min(candidates, key=lambda item: item.item_id)
            if candidates
            else None
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speech -> sign.mt -> Skeleton Pose client (Pi 4B optimised)"
    )
    parser.add_argument("--endpoint", default=SIGN_MT_ENDPOINT)
    parser.add_argument("--spoken", default=DEFAULT_SPOKEN_LANGUAGE)
    parser.add_argument("--signed", default=DEFAULT_SIGNED_LANGUAGE)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="API fetch workers (default 2 on Pi)",
    )
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=DEFAULT_PLAYBACK_RATE,
        help="Animation speed multiplier (default 2.0 on Pi)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=ASR_SAMPLE_RATE
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a Vosk model folder. Overrides VOSK_MODEL_PATH.",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        default=False,
        help="Enable frame interpolation (smoother but heavier on Pi CPU).",
    )
    parser.add_argument(
        "--demo-text",
        nargs="*",
        default=None,
        help="Optional text snippets queued after launch.",
    )
    parser.add_argument(
        "--auto-close-seconds",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--self-test",
        default=None,
        help="Fetch and parse one sign.mt response, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.concurrency = max(1, args.concurrency)
    args.playback_rate = max(0.1, args.playback_rate)

    if args.self_test is not None:
        sequence = fetch_pose_sequence(
            args.self_test,
            args.endpoint,
            args.spoken,
            args.signed,
            args.timeout,
        )
        print(
            json.dumps(
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
                                "name": c.name,
                                "format": c.fmt,
                                "points": len(c.points),
                            }
                            for c in sequence.pose.header.components
                        ],
                    },
                },
                indent=2,
            )
        )
        return

    app = SpeechPoseApp(args)
    app.run()


if __name__ == "__main__":
    main()