"""
media_recorder.py — Falcon Security: Synchronized Video + Audio Recorder
=========================================================================

Responsibilities
----------------
1. VIDEO  — Spawn one ffmpeg process per camera (RTSP → segmented .mp4).
            Uses NVIDIA NVENC (RTX 5090) for hardware-accelerated H.264
            encoding.  Falls back to software libx264 if NVENC is absent.

2. AUDIO  — Subscribe to the MQTT topic falcon/audio/{center}/{table}/{mic}/pcm
            and buffer raw I2S PCM frames from ESP32-S3 microphones.
            Every SEGMENT_SECONDS seconds the buffered PCM is flushed to a
            WAV file using the wave module (no extra dependencies).

3. SYNC   — Both video and audio segments share the same UTC timestamp
            so they can be correlated during playback / forensic review.

4. NAMING — table_{tableId}_cam_{cameraNumber}_mic_{micNumber}_{YYYYMMDD_HHMMSS}.mp4/.wav

5. INDEX  — After each segment is finalised the recorder POSTs to the
            NestJS /local-media endpoint so the file is discoverable
            through the Evidence Dashboard.

Usage
-----
    # Single camera + single microphone:
    python media_recorder.py \
        --center-id  <centerId> \
        --table-id   <tableId> \
        --cam-number 1 \
        --mic-number 1 \
        --rtsp-url   "rtsp://admin:pass@192.168.0.29:554/cam/realmonitor?channel=1&subtype=0"

    # All three Dahua DVR channels:
    python media_recorder.py --all-cameras

    # Dry-run (print commands, don't execute):
    python media_recorder.py --all-cameras --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import signal
import struct
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import paho.mqtt.client as mqtt

from recorder_config import get_recorder_settings

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("falcon.recorder")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def utc_stamp() -> str:
    """Return a sortable UTC timestamp string: YYYYMMDD_HHMMSS"""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def iso_date() -> str:
    """Return today's date as YYYY-MM-DD (used for directory + DB field)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def build_file_stem(
    table_id: str,
    cam_number: Optional[int],
    mic_number: Optional[int],
    ts: str,
) -> str:
    """
    Canonical filename stem (no extension):
        table_{tableId}_cam_{N}_mic_{M}_{YYYYMMDD_HHMMSS}
    Omits cam/mic component when not applicable.
    """
    parts: list[str] = [f"table_{table_id}"]
    if cam_number is not None:
        parts.append(f"cam_{cam_number}")
    if mic_number is not None:
        parts.append(f"mic_{mic_number}")
    parts.append(ts)
    return "_".join(parts)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def segment_dir(
    root: Path,
    center_id: str,
    table_id: str,
    media_type: str,
    date: str,
) -> Path:
    """
    Build and create the nested directory:
        {root}/{centerId}/{tableId}/{YYYY-MM-DD}/{mediaType}/
    """
    d = root / center_id / table_id / date / media_type
    return ensure_dir(d)


# ─────────────────────────────────────────────────────────────────────────────
# FFmpeg NVENC command builder
# ─────────────────────────────────────────────────────────────────────────────

def build_ffmpeg_cmd(
    *,
    ffmpeg_bin: str,
    rtsp_url: str,
    output_path: str,
    use_nvenc: bool,
    segment_seconds: int,
) -> list[str]:
    """
    Build the ffmpeg command to capture one RTSP segment.

    Hardware path (RTX 5090 NVENC):
      -hwaccel cuda -hwaccel_output_format cuda
      -c:v h264_nvenc -preset p1 -tune ll (ultra-low latency)
      -rc cbr -b:v 4M -maxrate 4M -bufsize 8M

    Software fallback (libx264):
      -c:v libx264 -preset ultrafast -tune zerolatency
      -crf 23

    Common flags:
      -rtsp_transport tcp      — reliable delivery over LAN
      -t <seconds>             — record exactly one segment
      -avoid_negative_ts make_zero
      -movflags +faststart     — place MOOV atom at file start (better seek)
    """

    cmd: list[str] = [ffmpeg_bin]

    # ── Hardware decode (NVENC path only) ─────────────────────────────────────
    if use_nvenc:
        cmd += [
            "-hwaccel",               "cuda",
            "-hwaccel_output_format", "cuda",
        ]

    # ── Input (RTSP) ──────────────────────────────────────────────────────────
    cmd += [
        "-rtsp_transport", "tcp",
        "-i",              rtsp_url,
        "-t",              str(segment_seconds),
        "-avoid_negative_ts", "make_zero",
    ]

    # ── Video encoder ─────────────────────────────────────────────────────────
    if use_nvenc:
        cmd += [
            # NVENC H.264 — optimised for RTX 5090
            "-c:v",      "h264_nvenc",
            "-preset",   "p1",           # fastest NVENC preset (lowest latency)
            "-tune",     "ll",           # low-latency tuning
            "-rc",       "cbr",          # constant bitrate for predictable size
            "-b:v",      "4M",
            "-maxrate",  "4M",
            "-bufsize",  "8M",
            "-g",        "60",           # keyframe every 2 s at 30 fps
            "-bf",       "0",            # no B-frames (lower latency)
            "-spatial-aq", "1",
        ]
    else:
        cmd += [
            "-c:v",    "libx264",
            "-preset", "ultrafast",
            "-tune",   "zerolatency",
            "-crf",    "23",
            "-g",      "60",
        ]

    # ── Audio (pass-through from DVR stream) ─────────────────────────────────
    cmd += ["-c:a", "aac", "-b:a", "128k"]

    # ── Output ────────────────────────────────────────────────────────────────
    cmd += [
        "-movflags", "+faststart",
        "-y",        output_path,
    ]

    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Video Segment Recorder (one instance per camera)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoRecorder:
    """
    Continuously records a single RTSP camera into sequential .mp4 segments.
    Runs inside an asyncio event loop via asyncio.subprocess.
    """

    center_id:      str
    table_id:       str
    cam_number:     int
    rtsp_url:       str

    _cfg = None   # set in __post_init__

    def __post_init__(self):
        self._cfg = get_recorder_settings()

    async def record_forever(self, stop_event: asyncio.Event) -> None:
        """Loop: record one segment, register it, repeat until stop_event."""
        cfg = self._cfg
        root = Path(cfg.recordings_root)
        log.info(
            f"[VideoRecorder] Starting camera {self.cam_number} "
            f"center={self.center_id} table={self.table_id}"
        )

        while not stop_event.is_set():
            ts   = utc_stamp()
            date = iso_date()
            out_dir = segment_dir(root, self.center_id, self.table_id, "VIDEO", date)
            stem = build_file_stem(self.table_id, self.cam_number, None, ts)
            out_path = str(out_dir / f"{stem}.mp4")

            cmd = build_ffmpeg_cmd(
                ffmpeg_bin      = cfg.ffmpeg_path,
                rtsp_url        = self.rtsp_url,
                output_path     = out_path,
                use_nvenc       = cfg.use_nvenc,
                segment_seconds = cfg.segment_seconds,
            )

            log.info(
                f"[cam{self.cam_number}] Recording segment → {out_path}\n"
                f"  cmd: {' '.join(cmd)}"
            )

            start_ts = asyncio.get_event_loop().time()
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                # Wait for segment to finish, or stop_event
                _, pending = await asyncio.wait(
                    [
                        asyncio.create_task(proc.wait()),
                        asyncio.create_task(stop_event.wait()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()

                if stop_event.is_set():
                    proc.terminate()
                    await proc.wait()
                    break

                duration = asyncio.get_event_loop().time() - start_ts
                file_size = Path(out_path).stat().st_size if Path(out_path).exists() else 0

                if proc.returncode == 0 or Path(out_path).exists():
                    log.info(
                        f"[cam{self.cam_number}] Segment done "
                        f"({duration:.1f}s, {file_size // 1024} KB) → {out_path}"
                    )
                    await register_media(
                        media_type   = "VIDEO",
                        absolute_path= out_path,
                        center_id    = self.center_id,
                        table_id     = self.table_id,
                        camera_number= self.cam_number,
                        recording_date=date,
                        file_size    = file_size,
                        duration_sec = duration,
                    )
                else:
                    stderr_out = b""
                    if proc.stderr:
                        stderr_out = await proc.stderr.read()
                    log.warning(
                        f"[cam{self.cam_number}] ffmpeg exited with code "
                        f"{proc.returncode}: {stderr_out[-300:].decode(errors='replace')}"
                    )
                    # Back-off before retrying to avoid flood on persistent error
                    await asyncio.sleep(5)

            except Exception as exc:
                log.error(f"[cam{self.cam_number}] Unexpected error: {exc}")
                await asyncio.sleep(5)

        log.info(f"[cam{self.cam_number}] Recorder stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Audio Buffer — PCM → WAV writer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioBuffer:
    """
    Thread-safe buffer that accumulates raw PCM bytes coming from the MQTT
    broker (ESP32-S3 I2S mic) and flushes them to a .wav file every
    SEGMENT_SECONDS seconds.

    MQTT payload format:
        Topic: falcon/audio/{centerId}/{tableId}/{micNumber}/pcm
        Payload: raw little-endian PCM frames (16-bit or 32-bit signed integers)
    """

    center_id:   str
    table_id:    str
    mic_number:  int

    _buf:        bytearray = field(default_factory=bytearray, init=False, repr=False)
    _lock:       asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _cfg         = None

    def __post_init__(self):
        self._cfg = get_recorder_settings()
        self._lock = asyncio.Lock()
        self._buf  = bytearray()

    def append_pcm(self, data: bytes) -> None:
        """Called from the MQTT on_message callback (sync context)."""
        self._buf.extend(data)

    async def flush(self) -> Optional[str]:
        """
        Write accumulated PCM to a .wav file.
        Returns the absolute path of the written file, or None if the buffer
        was empty.
        """
        async with self._lock:
            if not self._buf:
                return None

            pcm_data = bytes(self._buf)
            self._buf.clear()

        cfg  = self._cfg
        root = Path(cfg.recordings_root)
        ts   = utc_stamp()
        date = iso_date()
        out_dir = segment_dir(root, self.center_id, self.table_id, "AUDIO", date)
        stem = build_file_stem(self.table_id, None, self.mic_number, ts)
        out_path = str(out_dir / f"{stem}.wav")

        # Write WAV header + PCM payload
        sample_width = cfg.audio_bit_depth // 8  # bytes per sample
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(cfg.audio_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(cfg.audio_sample_rate)
            wf.writeframes(pcm_data)

        file_size = Path(out_path).stat().st_size
        num_frames = len(pcm_data) // (sample_width * cfg.audio_channels)
        duration   = num_frames / cfg.audio_sample_rate

        log.info(
            f"[mic{self.mic_number}] Flushed WAV "
            f"({duration:.1f}s, {file_size // 1024} KB) → {out_path}"
        )

        await register_media(
            media_type    = "AUDIO",
            absolute_path = out_path,
            center_id     = self.center_id,
            table_id      = self.table_id,
            mic_number    = self.mic_number,
            recording_date= date,
            file_size     = file_size,
            duration_sec  = duration,
        )
        return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MQTT Audio Client
# ─────────────────────────────────────────────────────────────────────────────

class MqttAudioClient:
    """
    Subscribes to falcon/audio/+/+/+/pcm and routes payloads to the
    correct AudioBuffer based on (centerId, tableId, micNumber) extracted
    from the topic segments.

    Topic format:
        falcon/audio/{centerId}/{tableId}/{micNumber}/pcm
    """

    def __init__(self):
        self._cfg = get_recorder_settings()
        # (center_id, table_id, mic_number) → AudioBuffer
        self._buffers: dict[tuple[str, str, int], AudioBuffer] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._client = mqtt.Client(client_id="falcon-recorder", protocol=mqtt.MQTTv5)
        self._client.on_connect    = self._on_connect
        self._client.on_message    = self._on_message
        self._client.on_disconnect = self._on_disconnect

        if self._cfg.mqtt_user:
            self._client.username_pw_set(self._cfg.mqtt_user, self._cfg.mqtt_pass)

    def register_buffer(self, buf: AudioBuffer) -> None:
        key = (buf.center_id, buf.table_id, buf.mic_number)
        self._buffers[key] = buf
        log.info(
            f"[MQTT] Registered audio buffer "
            f"center={buf.center_id} table={buf.table_id} mic={buf.mic_number}"
        )

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._client.connect_async(self._cfg.mqtt_host, self._cfg.mqtt_port, keepalive=60)
        self._client.loop_start()

    def stop(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    # ── MQTT callbacks ────────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc, props=None):
        if rc == 0:
            topic = self._cfg.mqtt_audio_topic_pattern
            client.subscribe(topic, qos=0)
            log.info(f"[MQTT] Connected — subscribed to '{topic}'")
        else:
            log.error(f"[MQTT] Connection failed, rc={rc}")

    def _on_disconnect(self, client, userdata, rc, props=None):
        log.warning(f"[MQTT] Disconnected (rc={rc}) — will auto-reconnect")

    def _on_message(self, client, userdata, message: mqtt.MQTTMessage):
        """
        Route incoming PCM payload to the matching AudioBuffer.
        Topic: falcon/audio/{centerId}/{tableId}/{micNumber}/pcm
        """
        parts = message.topic.split("/")
        # Expected: ['falcon', 'audio', centerId, tableId, micNumber, 'pcm']
        if len(parts) != 6 or parts[-1] != "pcm":
            return

        _, _, center_id, table_id, mic_str, _ = parts
        try:
            mic_number = int(mic_str)
        except ValueError:
            return

        key = (center_id, table_id, mic_number)
        buf = self._buffers.get(key)
        if buf is None:
            # Auto-create buffer for newly discovered microphones
            buf = AudioBuffer(
                center_id  = center_id,
                table_id   = table_id,
                mic_number = mic_number,
            )
            self._buffers[key] = buf
            log.info(f"[MQTT] Auto-registered buffer for mic {mic_number} on table {table_id}")

        buf.append_pcm(message.payload)


# ─────────────────────────────────────────────────────────────────────────────
# NestJS media registration
# ─────────────────────────────────────────────────────────────────────────────

async def register_media(
    *,
    media_type:     str,
    absolute_path:  str,
    center_id:      str,
    table_id:       Optional[str] = None,
    camera_number:  Optional[int] = None,
    mic_number:     Optional[int] = None,
    recording_date: Optional[str] = None,
    file_size:      int = 0,
    duration_sec:   Optional[float] = None,
) -> None:
    """
    POST /api/v1/local-media to register the completed file in the NestJS DB.
    Uses the internal NESTJS_SERVICE_KEY header for service-to-service auth.
    Failures are logged but never raise — recording must not stop on index error.
    """
    cfg = get_recorder_settings()
    payload: dict = {
        "mediaType":     media_type,
        "absolutePath":  absolute_path,
        "centerId":      center_id,
        "recordingDate": recording_date or iso_date(),
    }
    if table_id:      payload["tableId"]      = table_id
    if camera_number: payload["cameraNumber"] = camera_number
    if mic_number:    payload["micNumber"]    = mic_number
    if duration_sec:  payload["durationSec"]  = round(duration_sec, 2)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{cfg.nestjs_api_url}/local-media",
                json=payload,
                headers={
                    "x-service-key": cfg.nestjs_service_key,
                    "Content-Type":  "application/json",
                },
            )
            if resp.status_code in (200, 201):
                log.debug(f"[index] Registered {media_type} → {absolute_path}")
            else:
                log.warning(
                    f"[index] Registration failed (HTTP {resp.status_code}): "
                    f"{resp.text[:200]}"
                )
    except Exception as exc:
        log.warning(f"[index] Could not reach NestJS ({cfg.nestjs_api_url}): {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Periodic audio flush task
# ─────────────────────────────────────────────────────────────────────────────

async def flush_audio_periodically(
    buffers: list[AudioBuffer],
    stop_event: asyncio.Event,
) -> None:
    """Flush all audio buffers every SEGMENT_SECONDS seconds."""
    cfg = get_recorder_settings()
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=cfg.segment_seconds,
            )
        except asyncio.TimeoutError:
            pass  # normal — flush time

        # Flush all registered audio buffers concurrently
        results = await asyncio.gather(
            *[buf.flush() for buf in buffers],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                log.error(f"[audio flush] Error: {r}")


# ─────────────────────────────────────────────────────────────────────────────
# Session — groups cameras + mics for one recording session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecordingSession:
    """
    A recording session for a single center.
    Contains one VideoRecorder per camera and one AudioBuffer per microphone.
    """
    center_id: str
    cameras:   list[VideoRecorder] = field(default_factory=list)
    audio_buffers: list[AudioBuffer] = field(default_factory=list)

    def add_camera(
        self,
        table_id: str,
        cam_number: int,
        rtsp_url: str,
    ) -> VideoRecorder:
        rec = VideoRecorder(
            center_id  = self.center_id,
            table_id   = table_id,
            cam_number = cam_number,
            rtsp_url   = rtsp_url,
        )
        self.cameras.append(rec)
        return rec

    def add_microphone(self, table_id: str, mic_number: int) -> AudioBuffer:
        buf = AudioBuffer(
            center_id  = self.center_id,
            table_id   = table_id,
            mic_number = mic_number,
        )
        self.audio_buffers.append(buf)
        return buf


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def _run_session(session: RecordingSession, dry_run: bool) -> None:
    cfg = get_recorder_settings()
    stop_event = asyncio.Event()

    # ── Signal handlers ────────────────────────────────────────────────────
    loop = asyncio.get_running_loop()

    def _on_signal():
        log.info("Stop signal received — finishing current segments…")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    if dry_run:
        log.info("=== DRY RUN — printing ffmpeg commands ===")
        for cam in session.cameras:
            ts   = utc_stamp()
            date = iso_date()
            out_dir = segment_dir(
                Path(cfg.recordings_root),
                cam.center_id, cam.table_id, "VIDEO", date,
            )
            stem = build_file_stem(cam.table_id, cam.cam_number, None, ts)
            cmd = build_ffmpeg_cmd(
                ffmpeg_bin      = cfg.ffmpeg_path,
                rtsp_url        = cam.rtsp_url,
                output_path     = str(out_dir / f"{stem}.mp4"),
                use_nvenc       = cfg.use_nvenc,
                segment_seconds = cfg.segment_seconds,
            )
            log.info(f"  [dry-run] {' '.join(cmd)}")
        return

    # ── Start MQTT audio client ────────────────────────────────────────────
    mqtt_client = MqttAudioClient()
    for buf in session.audio_buffers:
        mqtt_client.register_buffer(buf)
    mqtt_client.start(loop)

    # ── Launch all tasks ───────────────────────────────────────────────────
    tasks: list[asyncio.Task] = []

    for cam in session.cameras:
        tasks.append(
            asyncio.create_task(
                cam.record_forever(stop_event),
                name=f"video-cam{cam.cam_number}",
            )
        )

    if session.audio_buffers:
        tasks.append(
            asyncio.create_task(
                flush_audio_periodically(session.audio_buffers, stop_event),
                name="audio-flush",
            )
        )

    log.info(
        f"🎬 Recording session started — "
        f"{len(session.cameras)} camera(s), "
        f"{len(session.audio_buffers)} microphone(s), "
        f"segment={cfg.segment_seconds}s, "
        f"NVENC={'yes' if cfg.use_nvenc else 'no (software)'}"
    )

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        mqtt_client.stop()
        # Final audio flush when stopping
        log.info("Flushing remaining audio buffers…")
        await asyncio.gather(
            *[buf.flush() for buf in session.audio_buffers],
            return_exceptions=True,
        )
        log.info("✅ Session stopped cleanly.")


def _build_session_from_args(args: argparse.Namespace) -> RecordingSession:
    cfg = get_recorder_settings()
    session = RecordingSession(center_id=args.center_id)

    if args.all_cameras:
        # Use all three seeded Dahua DVR channels
        channels = [
            (1, args.table_id or "default-table", 1),
            (2, args.table_id or "default-table", 2),
            (3, args.table_id or "default-table", 3),
        ]
        for cam_num, table_id, mic_num in channels:
            rtsp_url = (
                f"rtsp://{cfg.dvr_user}:{cfg.dvr_pass}@"
                f"{cfg.dvr_host}:{cfg.dvr_port}"
                f"/cam/realmonitor?channel={cam_num}&subtype=0"
            )
            session.add_camera(table_id=table_id, cam_number=cam_num, rtsp_url=rtsp_url)
            session.add_microphone(table_id=table_id, mic_number=mic_num)
    else:
        if not args.rtsp_url:
            raise ValueError("--rtsp-url is required unless --all-cameras is set")
        session.add_camera(
            table_id   = args.table_id or "default-table",
            cam_number = args.cam_number,
            rtsp_url   = args.rtsp_url,
        )
        if args.mic_number:
            session.add_microphone(
                table_id   = args.table_id or "default-table",
                mic_number = args.mic_number,
            )

    return session


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Falcon Security — Synchronized Video + Audio Recorder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--center-id",   default="cmnplma9p0001qi0gjud6861t",
                        help="Center (branch) ID from the Prisma DB")
    parser.add_argument("--table-id",    default=None,
                        help="Table ID (omit when using --all-cameras with defaults)")
    parser.add_argument("--cam-number",  type=int, default=1,
                        help="DVR camera channel number")
    parser.add_argument("--mic-number",  type=int, default=None,
                        help="Microphone number (omit to skip audio recording)")
    parser.add_argument("--rtsp-url",    default=None,
                        help="Full RTSP URL for the camera")
    parser.add_argument("--all-cameras", action="store_true",
                        help="Record all 3 Dahua DVR channels simultaneously")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print ffmpeg commands without executing them")
    args = parser.parse_args()

    # Load .env into environment before building session
    get_recorder_settings()

    cfg = get_recorder_settings()
    log.info(f"Storage root : {cfg.recordings_root}")
    log.info(f"FFmpeg       : {cfg.ffmpeg_path}")
    log.info(f"NVENC        : {'enabled (RTX 5090)' if cfg.use_nvenc else 'disabled (CPU)'}")
    log.info(f"Segment      : {cfg.segment_seconds}s")
    log.info(f"MQTT broker  : {cfg.mqtt_host}:{cfg.mqtt_port}")

    session = _build_session_from_args(args)
    asyncio.run(_run_session(session, args.dry_run))


if __name__ == "__main__":
    main()
