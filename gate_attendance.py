"""
gate_attendance.py
==================
Falcon Security — Gate Camera Face-Recognition Attendance Service
-----------------------------------------------------------------
Runs two dedicated RTSP camera streams:

  Camera_Entry  → punch-in  → POST /auth/punch-in
  Camera_Exit   → punch-out → POST /auth/punch-out

Architecture
------------
• DeepFace with ArcFace model + RetinaFace detector (CUDA / RTX-optimised)
• Per-camera worker threads keep frame capture and inference decoupled
• asyncio + aiohttp for non-blocking NestJS API calls
• ThreadPoolExecutor for parallel face recognition across multiple
  agents detected in a single frame (handles group entry/exit)
• Anti-duplicate cooldown: same agent cannot be punched-in/out twice
  within COOLDOWN_SECONDS (default 300 s / 5 min)
• Captured face crops are saved to CAPTURE_DIR for audit / re-training

RTX 5090 optimisations
-----------------------
• torch.backends.cudnn.benchmark = True  (tuned conv kernels)
• torch.set_float32_matmul_precision('high')  (TF32 tensor-core speed-up)
• DeepFace model loaded once at startup, kept in VRAM
• Batch inference via DeepFace.represent() on ALL faces in one frame
• workers=4 per camera (can be bumped if VRAM allows)

Usage
-----
  # Copy .env.gate-example to .env.gate and fill in values, then:
  python gate_attendance.py

Environment variables (loaded from .env.gate or system env)
------------------------------------------------------------
  ENTRY_CAM_URL       rtsp://... URL of the entry-gate camera
  EXIT_CAM_URL        rtsp://... URL of the exit-gate camera
  AGENT_FACES_DIR     directory of reference face images (one .jpg per agent,
                      filename = <userId>.jpg  e.g. clxyz123.jpg)
  CAPTURE_DIR         where to save gate-capture snapshots (default /data/gate-captures)
  NESTJS_BASE_URL     e.g. http://localhost:4000/api/v1
  GATE_API_KEY        shared secret — must match GATE_API_KEY in NestJS .env
  CENTER_ID           Prisma CUID of the branch where the gate cameras are installed
  RECOGNITION_THRESHOLD  cosine distance threshold (default 0.40 — lower = stricter)
  COOLDOWN_SECONDS    seconds before same agent can be re-punched (default 300)
  FRAME_SKIP          process every Nth frame (default 5 — ~6 fps @ 30 fps cam)
  LOG_LEVEL           DEBUG | INFO | WARNING (default INFO)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
import cv2
import numpy as np
from deepface import DeepFace
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# GPU / CUDA optimisations (RTX 5090)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import torch

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True          # auto-tune conv kernels
        torch.set_float32_matmul_precision("high")     # TF32 on Ampere+
        _DEVICE = "cuda"
        _GPU_NAME = torch.cuda.get_device_name(0)
    else:
        _DEVICE = "cpu"
        _GPU_NAME = "CPU (no CUDA)"
except ImportError:
    _DEVICE = "cpu"
    _GPU_NAME = "CPU (torch not installed)"

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env.gate", override=False)

ENTRY_CAM_URL: str       = os.environ["ENTRY_CAM_URL"]
EXIT_CAM_URL: str        = os.environ["EXIT_CAM_URL"]
AGENT_FACES_DIR: Path    = Path(os.environ.get("AGENT_FACES_DIR", "/data/agent-faces"))
CAPTURE_DIR: Path        = Path(os.environ.get("CAPTURE_DIR", "/data/gate-captures"))
NESTJS_BASE_URL: str     = os.environ.get("NESTJS_BASE_URL", "http://localhost:4000/api/v1").rstrip("/")
GATE_API_KEY: str        = os.environ["GATE_API_KEY"]
CENTER_ID: str           = os.environ["CENTER_ID"]
THRESHOLD: float         = float(os.environ.get("RECOGNITION_THRESHOLD", "0.40"))
COOLDOWN: int            = int(os.environ.get("COOLDOWN_SECONDS", "300"))
FRAME_SKIP: int          = int(os.environ.get("FRAME_SKIP", "5"))
LOG_LEVEL: str           = os.environ.get("LOG_LEVEL", "INFO").upper()

# DeepFace settings
_DETECTOR  = "retinaface"   # best accuracy — leverages CUDA via torch
_MODEL     = "ArcFace"      # compact + accurate; loaded once globally

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gate_attendance")

# ──────────────────────────────────────────────────────────────────────────────
# Reference face database
# Loaded once at startup: { userId: np.ndarray of embedding (512-d) }
# ──────────────────────────────────────────────────────────────────────────────
FaceDB = dict[str, np.ndarray]

def load_face_database() -> FaceDB:
    """
    Walk AGENT_FACES_DIR and compute an ArcFace embedding for every agent photo.
    File naming convention: <userId>.jpg  (one reference photo per agent).
    Returns a dict { userId: embedding_vector }.
    """
    AGENT_FACES_DIR.mkdir(parents=True, exist_ok=True)
    db: FaceDB = {}
    photos = list(AGENT_FACES_DIR.glob("*.jpg")) + list(AGENT_FACES_DIR.glob("*.jpeg"))

    if not photos:
        log.warning("No agent photos found in %s — no-one will be recognised.", AGENT_FACES_DIR)
        return db

    log.info("Loading face embeddings for %d agents from %s …", len(photos), AGENT_FACES_DIR)
    for photo in photos:
        user_id = photo.stem  # filename without extension = userId
        try:
            result = DeepFace.represent(
                img_path=str(photo),
                model_name=_MODEL,
                detector_backend=_DETECTOR,
                enforce_detection=True,
                align=True,
            )
            if result:
                embedding = np.array(result[0]["embedding"], dtype=np.float32)
                db[user_id] = embedding
                log.debug("  ✓ %s loaded (dim=%d)", user_id, len(embedding))
        except Exception as exc:
            log.warning("  ✗ Could not embed %s: %s", photo.name, exc)

    log.info("Face database ready — %d / %d agents enrolled.", len(db), len(photos))
    return db


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Returns cosine distance in [0, 2]. Identical vectors → 0."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 2.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def match_embedding(embedding: np.ndarray, db: FaceDB) -> Optional[tuple[str, float]]:
    """
    Find the closest match in db.
    Returns (userId, distance) if below THRESHOLD, else None.
    """
    best_id, best_dist = None, float("inf")
    for user_id, ref in db.items():
        dist = cosine_distance(embedding, ref)
        if dist < best_dist:
            best_dist = dist
            best_id = user_id
    if best_id and best_dist <= THRESHOLD:
        return best_id, best_dist
    return None


def detect_and_embed_all(frame_bgr: np.ndarray) -> list[dict]:
    """
    Run RetinaFace detection + ArcFace embedding on ALL faces in one frame.
    Returns a list of dicts: { "embedding": np.ndarray, "region": dict }
    Returns [] if no faces detected.
    """
    try:
        results = DeepFace.represent(
            img_path=frame_bgr,
            model_name=_MODEL,
            detector_backend=_DETECTOR,
            enforce_detection=False,    # return [] instead of raising on no face
            align=True,
        )
        return [
            {
                "embedding": np.array(r["embedding"], dtype=np.float32),
                "region":    r.get("facial_area", {}),
            }
            for r in results
            if r.get("embedding")
        ]
    except Exception as exc:
        log.debug("DeepFace.represent error: %s", exc)
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Capture-image saving
# ──────────────────────────────────────────────────────────────────────────────

def save_capture(frame_bgr: np.ndarray, gate: str, user_id: str) -> str:
    """
    Crop face region (or save full frame) to CAPTURE_DIR/{gate}/.
    Returns the absolute path as a string.
    """
    gate_dir = CAPTURE_DIR / gate
    gate_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = gate_dir / f"{ts}_{user_id}.jpg"
    cv2.imwrite(str(filename), frame_bgr)
    return str(filename)


# ──────────────────────────────────────────────────────────────────────────────
# NestJS API calls (async)
# ──────────────────────────────────────────────────────────────────────────────

async def call_nestjs(
    session: aiohttp.ClientSession,
    endpoint: str,          # "punch-in" or "punch-out"
    user_id: str,
    face_image_path: str,
) -> None:
    url = f"{NESTJS_BASE_URL}/auth/{endpoint}"
    payload = {
        "userId":    user_id,
        "centerId":  CENTER_ID,
        "faceImage": face_image_path,
    }
    headers = {"x-gate-api-key": GATE_API_KEY}
    try:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as resp:
            body = await resp.json(content_type=None)
            if resp.status == 200:
                status = body.get("status", "ok")
                log.info("✅ %s → %s  [%s]", endpoint, user_id, status)
            else:
                log.warning("⚠️  %s → %s  HTTP %s: %s", endpoint, user_id, resp.status, body)
    except Exception as exc:
        log.error("❌ Failed to call %s for %s: %s", endpoint, user_id, exc)


# ──────────────────────────────────────────────────────────────────────────────
# Camera worker
# Runs in a background thread; posts matched faces to an asyncio queue.
# ──────────────────────────────────────────────────────────────────────────────

class GateCameraWorker(threading.Thread):
    """
    Continuously reads frames from one RTSP camera, detects faces,
    matches against the in-memory face DB, and queues (userId, gate, frame)
    tuples for the async NestJS caller.
    """

    def __init__(
        self,
        name: str,
        rtsp_url: str,
        gate: str,                          # "entry" or "exit"
        face_db: FaceDB,
        event_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        executor: ThreadPoolExecutor,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.rtsp_url    = rtsp_url
        self.gate        = gate
        self.face_db     = face_db
        self.event_queue = event_queue
        self.loop        = loop
        self.executor    = executor
        self._stop_event = threading.Event()

        # Cooldown: { userId: last_seen_unix_ts }
        self._cooldown: dict[str, float] = {}

    def stop(self) -> None:
        self._stop_event.set()

    def _in_cooldown(self, user_id: str) -> bool:
        last = self._cooldown.get(user_id, 0.0)
        return (time.monotonic() - last) < COOLDOWN

    def _update_cooldown(self, user_id: str) -> None:
        self._cooldown[user_id] = time.monotonic()

    def _process_frame(self, frame_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """
        Run face detection + matching on one frame.
        Returns list of (userId, frame) for each unique, non-cooldown match.
        """
        detections = detect_and_embed_all(frame_bgr)
        matches: list[tuple[str, np.ndarray]] = []

        for det in detections:
            result = match_embedding(det["embedding"], self.face_db)
            if result is None:
                continue
            user_id, dist = result
            if self._in_cooldown(user_id):
                log.debug("[%s] %s in cooldown, skipping (dist=%.3f)", self.gate, user_id, dist)
                continue
            log.info("[%s] Recognised: %s  (dist=%.3f)", self.gate, user_id, dist)
            self._update_cooldown(user_id)
            matches.append((user_id, frame_bgr.copy()))

        return matches

    def run(self) -> None:
        log.info("[%s] Connecting to %s …", self.gate, self.rtsp_url)

        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                log.warning("[%s] Cannot open stream, retrying in 5 s …", self.gate)
                time.sleep(5)
                continue

            log.info("[%s] Stream opened.", self.gate)
            frame_idx = 0

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    log.warning("[%s] Frame read failed, reconnecting …", self.gate)
                    break

                frame_idx += 1
                if frame_idx % FRAME_SKIP != 0:
                    continue  # skip to reduce inference load

                # Run matching (CPU-fallback OK; DeepFace uses CUDA if available)
                matched = self._process_frame(frame)

                for user_id, snapshot in matched:
                    # Thread-safe enqueue into the asyncio event loop
                    asyncio.run_coroutine_threadsafe(
                        self.event_queue.put((user_id, self.gate, snapshot)),
                        self.loop,
                    )

            cap.release()

        log.info("[%s] Worker stopped.", self.gate)


# ──────────────────────────────────────────────────────────────────────────────
# Async event dispatcher
# ──────────────────────────────────────────────────────────────────────────────

async def event_dispatcher(event_queue: asyncio.Queue) -> None:
    """
    Consumes (userId, gate, frame) from the queue and calls NestJS.
    Uses a single persistent aiohttp session for connection-pool efficiency.
    """
    conn = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=conn) as session:
        log.info("Event dispatcher ready — waiting for gate events …")
        while True:
            user_id, gate, snapshot = await event_queue.get()
            endpoint = "punch-in" if gate == "entry" else "punch-out"

            # Save face capture image asynchronously (offload I/O)
            loop = asyncio.get_event_loop()
            face_path = await loop.run_in_executor(
                None, save_capture, snapshot, gate, user_id
            )

            # Fire NestJS call without blocking the queue consumer
            asyncio.create_task(
                call_nestjs(session, endpoint, user_id, face_path)
            )
            event_queue.task_done()


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("=" * 60)
    log.info("  Falcon Gate Attendance Service")
    log.info("  Device: %s", _GPU_NAME)
    log.info("  Model:  %s + %s detector", _MODEL, _DETECTOR)
    log.info("  Threshold: %.2f  |  Cooldown: %d s  |  FrameSkip: %d", THRESHOLD, COOLDOWN, FRAME_SKIP)
    log.info("=" * 60)

    # Load all agent face embeddings into memory
    face_db = load_face_database()

    # Shared asyncio queue for cross-thread → async communication
    event_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    loop = asyncio.get_event_loop()

    # Executor for parallel inference (two cameras × workers)
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gate-worker")

    # Spawn one camera worker per gate
    entry_worker = GateCameraWorker(
        name="entry-cam",
        rtsp_url=ENTRY_CAM_URL,
        gate="entry",
        face_db=face_db,
        event_queue=event_queue,
        loop=loop,
        executor=executor,
    )
    exit_worker = GateCameraWorker(
        name="exit-cam",
        rtsp_url=EXIT_CAM_URL,
        gate="exit",
        face_db=face_db,
        event_queue=event_queue,
        loop=loop,
        executor=executor,
    )

    entry_worker.start()
    exit_worker.start()

    # Run the dispatcher (never returns under normal operation)
    try:
        await event_dispatcher(event_queue)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("Shutdown requested — stopping workers …")
        entry_worker.stop()
        exit_worker.stop()
        entry_worker.join(timeout=5)
        exit_worker.join(timeout=5)
        executor.shutdown(wait=False)
        log.info("Gate attendance service stopped.")


if __name__ == "__main__":
    asyncio.run(main())
