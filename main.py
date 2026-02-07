from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uuid
import os
import pickle
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io
import cv2
import tempfile
import shutil
from contextlib import asynccontextmanager
import threading
import logging
import time
import re
import json
import asyncio
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# GPU / CPU AUTO-DETECTION
# ============================================================================
def _detect_device() -> dict:
    """Detect whether a CUDA GPU is available via ONNX Runtime providers."""
    available = ort.get_available_providers()
    
    # Check if NVIDIA drivers are actually available (prevents loading CUDA libs on CPU-only)
    has_nvidia_driver = shutil.which("nvidia-smi") is not None
    
    use_gpu = "CUDAExecutionProvider" in available and has_nvidia_driver
    
    if use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ctx_id = 0  # GPU device id
    else:
        providers = ['CPUExecutionProvider']
        ctx_id = -1  # CPU
    return {
        "use_gpu": use_gpu,
        "providers": providers,
        "ctx_id": ctx_id,
        "available_providers": available,
    }


DEVICE_INFO = _detect_device()
logger.info(f"Device: {'GPU (CUDA)' if DEVICE_INFO['use_gpu'] else 'CPU'}")
logger.info(f"Available ONNX Runtime providers: {DEVICE_INFO['available_providers']}")

# ============================================================================
# CONFIGURATION - Auto-tuned for GPU when available, CPU fallback
# ============================================================================
DB_FILE = "data/embeddings_insightface.pkl"  # DB file for InsightFace embeddings
RECOGNITION_THRESHOLD = 0.5  # Cosine distance threshold (lower = stricter, higher = more lenient)
MODEL_NAME = "buffalo_l"  # buffalo_sc (fast) | buffalo_s | buffalo_l (most accurate, best for angles)

if DEVICE_INFO["use_gpu"]:
    # GPU can handle heavier workloads
    VIDEO_PROCESS_EVERY_N_FRAMES = 2   # Process nearly every frame on GPU
    STREAM_PROCESS_EVERY_N_FRAMES = 2  # Near real-time stream processing
    DETECTION_SIZE = 640               # Full resolution detection
else:
    # Conservative settings for CPU
    VIDEO_PROCESS_EVERY_N_FRAMES = 5
    STREAM_PROCESS_EVERY_N_FRAMES = 5
    DETECTION_SIZE = 640               # Detection input size (320/480/640 - lower = faster)

# Thread-safe locks
db_lock = threading.Lock()
model_lock = threading.Lock()

# ============================================================================
# GLOBAL STATE
# ============================================================================
# Structure: uuid -> {"name": str, "embeddings": List[np.array]}
known_embeddings: Dict[str, dict] = {}

# Cached flattened embeddings for fast recognition
_cached_embeddings: Optional[Tuple[List[str], np.ndarray, Dict[str, str]]] = None
_cache_valid = False

# InsightFace model (lazy loaded)
_face_analyzer: Optional[FaceAnalysis] = None

# Video analysis jobs - for SSE progress tracking
# job_id -> {"status": str, "progress": float, "detections": list, "result": dict}
video_jobs: Dict[str, dict] = {}
jobs_lock = threading.Lock()


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
def get_face_analyzer() -> FaceAnalysis:
    """
    Lazy load the InsightFace model with GPU acceleration when available.
    Falls back to CPU automatically. Uses thread-safe singleton pattern.
    """
    global _face_analyzer
    
    if _face_analyzer is not None:
        return _face_analyzer
    
    with model_lock:
        if _face_analyzer is not None:
            return _face_analyzer
        
        device_label = "GPU (CUDA)" if DEVICE_INFO["use_gpu"] else "CPU"
        logger.info(f"Loading InsightFace model: {MODEL_NAME} on {device_label}")
        start_time = time.time()
        
        # Initialize with detected providers (CUDA preferred, CPU fallback)
        app = FaceAnalysis(
            name=MODEL_NAME,
            providers=DEVICE_INFO["providers"],
            allowed_modules=['detection', 'recognition']  # Skip age/gender for speed
        )
        
        # ctx_id=0 for GPU, ctx_id=-1 for CPU
        app.prepare(
            ctx_id=DEVICE_INFO["ctx_id"],
            det_size=(DETECTION_SIZE, DETECTION_SIZE)
        )
        
        _face_analyzer = app
        logger.info(f"Model loaded on {device_label} in {time.time() - start_time:.2f}s")
        
        return _face_analyzer


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================
def invalidate_cache():
    """Mark the embedding cache as invalid."""
    global _cache_valid
    _cache_valid = False


def get_cached_embeddings() -> Tuple[List[str], np.ndarray, Dict[str, str]]:
    """
    Get flattened embeddings array for vectorized similarity computation.
    Returns: (uuids_list, embeddings_matrix, uuid_to_name_map)
    """
    global _cached_embeddings, _cache_valid
    
    if _cache_valid and _cached_embeddings is not None:
        return _cached_embeddings
    
    uuids: List[str] = []
    embeddings: List[np.ndarray] = []
    names: Dict[str, str] = {}
    
    for uid, data in known_embeddings.items():
        names[uid] = data["name"]
        for emb in data["embeddings"]:
            uuids.append(uid)
            embeddings.append(emb)
    
    # Stack into matrix for vectorized operations
    emb_matrix = np.array(embeddings) if embeddings else np.array([]).reshape(0, 512)
    
    _cached_embeddings = (uuids, emb_matrix, names)
    _cache_valid = True
    
    return _cached_embeddings


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================
def load_db():
    """Load embeddings database from disk."""
    global known_embeddings
    
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "rb") as f:
                known_embeddings = pickle.load(f)
            invalidate_cache()
            logger.info(f"Loaded {len(known_embeddings)} users from database")
        except Exception as e:
            logger.error(f"Error loading DB: {e}")
            known_embeddings = {}
    else:
        logger.info("No existing database found, starting fresh")


def save_db():
    """Save embeddings database to disk (thread-safe)."""
    with db_lock:
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
        with open(DB_FILE, "wb") as f:
            pickle.dump(known_embeddings, f)
    invalidate_cache()


# ============================================================================
# FACE RECOGNITION UTILITIES
# ============================================================================
def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    Returns value between -1 and 1 (higher = more similar).
    """
    return float(np.dot(embedding1, embedding2) / 
                 (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))


def find_best_match(
    face_embedding: np.ndarray,
    known_uuids: List[str],
    known_embeddings_matrix: np.ndarray,
    known_names: Dict[str, str],
    threshold: float = RECOGNITION_THRESHOLD
) -> Tuple[bool, str, str, float]:
    """
    Find the best matching face using vectorized cosine similarity.
    
    Returns: (is_match, uuid, name, confidence)
    """
    if len(known_embeddings_matrix) == 0:
        return False, "", "Unknown", 0.0
    
    # Vectorized cosine similarity computation
    # Normalize the query embedding
    query_norm = face_embedding / np.linalg.norm(face_embedding)
    
    # Normalize all known embeddings (row-wise)
    norms = np.linalg.norm(known_embeddings_matrix, axis=1, keepdims=True)
    known_norm = known_embeddings_matrix / norms
    
    # Compute all similarities at once
    similarities = np.dot(known_norm, query_norm)
    
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    
    # Convert similarity to distance (for threshold comparison)
    # distance = 1 - similarity (0 = identical, 2 = opposite)
    distance = 1 - best_similarity
    
    if distance < threshold:
        match_uuid = known_uuids[best_idx]
        confidence = float(best_similarity)  # Use similarity as confidence
        return True, match_uuid, known_names[match_uuid], confidence
    
    return False, "", "Unknown", 0.0


def process_image_bytes(contents: bytes) -> np.ndarray:
    """Convert image bytes to BGR numpy array for InsightFace."""
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


# ============================================================================
# VIDEO ANALYSIS WITH PROGRESS TRACKING
# ============================================================================
def process_video_with_progress(job_id: str, temp_path: str):
    """
    Process video in background thread with progress updates.
    Updates video_jobs[job_id] with progress, detections, and final result.
    """
    try:
        detections_dir = "static/detections"
        os.makedirs(detections_dir, exist_ok=True)
        
        video_capture = cv2.VideoCapture(temp_path)
        
        if not video_capture.isOpened():
            with jobs_lock:
                video_jobs[job_id]["status"] = "error"
                video_jobs[job_id]["error"] = "Failed to open video file"
            return
        
        # Get video properties
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS) or 30
        
        with jobs_lock:
            video_jobs[job_id]["total_frames"] = total_frames
            video_jobs[job_id]["fps"] = fps
            video_jobs[job_id]["status"] = "processing"
        
        analyzer = get_face_analyzer()
        known_uuids, known_matrix, known_names = get_cached_embeddings()
        
        if len(known_matrix) == 0:
            video_capture.release()
            with jobs_lock:
                video_jobs[job_id]["status"] = "complete"
                video_jobs[job_id]["progress"] = 100
                video_jobs[job_id]["result"] = {"message": "Video analysis complete", "found_people": []}
            return
        
        # Track best detection of each person
        found_people = {}
        frame_count = 0
        frames_processed = 0
        start_time = time.time()
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            elapsed = time.time() - start_time
            frames_per_sec = frames_processed / elapsed if elapsed > 0 else 0
            remaining_frames = total_frames - frame_count
            eta_seconds = remaining_frames / (fps * (frames_processed / frame_count)) if frames_processed > 0 and frame_count > 0 else 0
            
            with jobs_lock:
                video_jobs[job_id]["progress"] = progress
                video_jobs[job_id]["current_frame"] = frame_count
                video_jobs[job_id]["processing_fps"] = round(frames_per_sec, 1)
                video_jobs[job_id]["eta_seconds"] = round(eta_seconds, 1)
            
            if frame_count % VIDEO_PROCESS_EVERY_N_FRAMES != 0:
                continue
            
            frames_processed += 1
            faces = analyzer.get(frame)
            
            for face in faces:
                embedding = face.embedding / np.linalg.norm(face.embedding)
                bbox = face.bbox.astype(int)
                
                is_match, _, name, confidence = find_best_match(
                    embedding, known_uuids, known_matrix, known_names
                )
                
                if is_match:
                    # Check if this is a new detection or better confidence
                    is_new = name not in found_people
                    is_better = not is_new and confidence > found_people[name]["confidence"]
                    
                    if is_new or is_better:
                        found_people[name] = {
                            "frame": frame.copy(),
                            "confidence": confidence,
                            "frame_number": frame_count,
                            "box": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                        }
                        
                        # Send detection event
                        with jobs_lock:
                            detection_event = {
                                "name": name,
                                "confidence": confidence,
                                "frame_number": frame_count,
                                "is_new": is_new
                            }
                            video_jobs[job_id]["detections"].append(detection_event)
                            video_jobs[job_id]["latest_detection"] = detection_event
        
        video_capture.release()
        
        # Save best frame for each person
        result_people = []
        for name, data in found_people.items():
            frame = data["frame"]
            left, top, right, bottom = data["box"]
            confidence = data["confidence"]
            
            # Draw bounding box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.rectangle(frame, (left, bottom - 40), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                frame, f"{name} ({confidence:.1%})",
                (left + 6, bottom - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2
            )
            
            # Save frame
            timestamp = int(time.time() * 1000)
            safe_name = re.sub(r'[^a-z0-9]+', '_', name.lower())
            filename = f"{timestamp}_{safe_name}.jpg"
            filepath = os.path.join(detections_dir, filename)
            cv2.imwrite(filepath, frame)
            
            result_people.append({
                "name": name,
                "frame_url": f"/detections/{filename}",
                "confidence": confidence,
                "frame_number": data["frame_number"]
            })
        
        with jobs_lock:
            video_jobs[job_id]["status"] = "complete"
            video_jobs[job_id]["progress"] = 100
            video_jobs[job_id]["result"] = {
                "message": "Video analysis complete",
                "found_people": result_people,
                "total_frames_analyzed": frame_count,
                "processing_time": round(time.time() - start_time, 2)
            }
            
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        with jobs_lock:
            video_jobs[job_id]["status"] = "error"
            video_jobs[job_id]["error"] = str(e)
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    os.makedirs("static", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    load_db()
    
    # Pre-load model during startup
    logger.info("Pre-loading InsightFace model...")
    get_face_analyzer()
    
    logger.info("Application started")
    yield
    # Shutdown
    logger.info("Application shutting down")


app = FastAPI(
    title="God's Eye - Face Recognition API",
    description="Fast face recognition powered by InsightFace",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/users")
async def get_users():
    """Get list of all registered users with their embedding counts."""
    return [
        {
            "uuid": uid,
            "name": data["name"],
            "count": len(data["embeddings"])
        }
        for uid, data in known_embeddings.items()
    ]


@app.post("/register")
async def register(name: str = Form(...), files: List[UploadFile] = File(...)):
    """
    Register a new user or add more photos to an existing user.
    InsightFace extracts high-quality embeddings without needing jittering.
    """
    try:
        analyzer = get_face_analyzer()
        
        # Check if name already exists (case-insensitive)
        user_uuid = None
        name_lower = name.lower()
        for uid, data in known_embeddings.items():
            if data["name"].lower() == name_lower:
                user_uuid = uid
                break
        
        if not user_uuid:
            user_uuid = str(uuid.uuid4())
            known_embeddings[user_uuid] = {
                "name": name,
                "embeddings": []
            }
        
        processed_count = 0
        
        for file in files:
            contents = await file.read()
            try:
                image = process_image_bytes(contents)
            except ValueError as e:
                logger.warning(f"Skipping invalid image: {e}")
                continue
            
            # Detect and get embeddings
            faces = analyzer.get(image)
            
            if faces:
                # Use the largest face (most prominent) if multiple detected
                largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embedding = largest_face.embedding
                
                # Normalize embedding before storing
                embedding = embedding / np.linalg.norm(embedding)
                known_embeddings[user_uuid]["embeddings"].append(embedding)
                processed_count += 1
        
        if processed_count == 0:
            # Clean up if no faces found and user was just created
            if len(known_embeddings[user_uuid]["embeddings"]) == 0:
                del known_embeddings[user_uuid]
            return JSONResponse(
                status_code=400,
                content={"message": "No faces found in any of the uploaded images"}
            )
        
        save_db()
        
        return {
            "message": f"Registered {name} successfully. Processed {processed_count} images.",
            "uuid": user_uuid,
            "total_embeddings": len(known_embeddings[user_uuid]["embeddings"])
        }
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    """Recognize faces in an uploaded image."""
    try:
        if not known_embeddings:
            return JSONResponse(
                status_code=400,
                content={"message": "No registered faces in database"}
            )
        
        contents = await file.read()
        image = process_image_bytes(contents)
        
        analyzer = get_face_analyzer()
        faces = analyzer.get(image)
        
        if not faces:
            return JSONResponse(
                status_code=400,
                content={"message": "No face found in image"}
            )
        
        # Get cached recognition data
        known_uuids, known_matrix, known_names = get_cached_embeddings()
        
        found_faces = []
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding / np.linalg.norm(face.embedding)
            
            is_match, match_uuid, name, confidence = find_best_match(
                embedding, known_uuids, known_matrix, known_names
            )
            
            face_data = {
                "box": {
                    "top": int(bbox[1]),
                    "right": int(bbox[2]),
                    "bottom": int(bbox[3]),
                    "left": int(bbox[0])
                },
                "match": is_match,
                "name": name,
                "confidence": confidence
            }
            
            if is_match:
                face_data["uuid"] = match_uuid
            
            found_faces.append(face_data)
        
        return {
            "message": f"Found {len(found_faces)} faces",
            "faces": found_faces
        }
        
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/recognize_video")
async def recognize_video(file: UploadFile = File(...)):
    """
    Start video analysis job and return job ID for progress tracking.
    Use /video_progress/{job_id} SSE endpoint to track progress.
    """
    try:
        if not known_embeddings:
            return JSONResponse(
                status_code=400,
                content={"message": "No registered faces in database"}
            )
        
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
        
        # Initialize job state
        with jobs_lock:
            video_jobs[job_id] = {
                "status": "starting",
                "progress": 0,
                "current_frame": 0,
                "total_frames": 0,
                "fps": 0,
                "processing_fps": 0,
                "eta_seconds": 0,
                "detections": [],
                "latest_detection": None,
                "result": None,
                "error": None
            }
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_video_with_progress,
            args=(job_id, temp_path),
            daemon=True
        )
        thread.start()
        
        return {"job_id": job_id, "message": "Video analysis started"}
        
    except Exception as e:
        logger.error(f"Video recognition error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.get("/video_progress/{job_id}")
async def video_progress(job_id: str):
    """
    Server-Sent Events endpoint for real-time video analysis progress.
    Streams progress updates, detection events, and final results.
    """
    async def event_generator():
        last_detection_count = 0
        
        while True:
            with jobs_lock:
                job = video_jobs.get(job_id)
            
            if job is None:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break
            
            # Send progress update
            event_data = {
                "type": "progress",
                "status": job["status"],
                "progress": round(job["progress"], 1),
                "current_frame": job["current_frame"],
                "total_frames": job["total_frames"],
                "processing_fps": job["processing_fps"],
                "eta_seconds": job["eta_seconds"],
                "detections_count": len(job["detections"])
            }
            
            # Include new detections
            if len(job["detections"]) > last_detection_count:
                new_detections = job["detections"][last_detection_count:]
                event_data["new_detections"] = new_detections
                last_detection_count = len(job["detections"])
            
            yield f"data: {json.dumps(event_data)}\n\n"
            
            # Check if complete or error
            if job["status"] == "complete":
                yield f"data: {json.dumps({'type': 'complete', 'result': job['result']})}\n\n"
                break
            elif job["status"] == "error":
                yield f"data: {json.dumps({'type': 'error', 'error': job['error']})}\n\n"
                break
            
            await asyncio.sleep(0.3)  # Update every 300ms
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/video_job/{job_id}")
async def get_video_job(job_id: str):
    """Get the current state of a video analysis job."""
    with jobs_lock:
        job = video_jobs.get(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job


# ============================================================================
# RTSP STREAMING
# ============================================================================
camera_config = {"url": None}


def generate_frames(rtsp_url: str):
    """
    Generator that yields MJPEG frames from an RTSP stream with face recognition overlays.
    """
    camera = cv2.VideoCapture(rtsp_url)
    if not camera.isOpened():
        logger.error(f"Cannot open camera: {rtsp_url}")
        return
    
    analyzer = get_face_analyzer()
    known_uuids, known_matrix, known_names = get_cached_embeddings()
    
    frame_count = 0
    detected_faces: List[Tuple[int, int, int, int, str, Tuple[int, int, int]]] = []
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            frame_count += 1
            
            # Process faces periodically
            if frame_count % STREAM_PROCESS_EVERY_N_FRAMES == 0 and len(known_matrix) > 0:
                faces = analyzer.get(frame)
                
                new_faces = []
                for face in faces:
                    bbox = face.bbox.astype(int)
                    embedding = face.embedding / np.linalg.norm(face.embedding)
                    
                    is_match, _, name, _ = find_best_match(
                        embedding, known_uuids, known_matrix, known_names
                    )
                    
                    color = (0, 255, 0) if is_match else (0, 0, 255)
                    new_faces.append((
                        int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[0]),
                        name, color
                    ))
                
                detected_faces = new_faces
            
            # Draw detected faces
            for (top, right, bottom, left, name, color) in detected_faces:
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
            )
    finally:
        camera.release()


@app.post("/set_camera_url")
async def set_camera_url(url: str = Form(...)):
    """Set the RTSP camera URL for live streaming."""
    camera_config["url"] = url
    return {"message": "Camera URL updated successfully"}


@app.get("/video_feed")
async def video_feed():
    """Stream live video with face recognition overlays."""
    url = camera_config.get("url")
    if not url:
        raise HTTPException(
            status_code=400,
            detail="No camera URL set. Please configure it first."
        )
    
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise HTTPException(
            status_code=400,
            detail="Failed to connect to RTSP stream"
        )
    cap.release()
    
    return StreamingResponse(
        generate_frames(url),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.delete("/users/{user_uuid}")
async def delete_user(user_uuid: str):
    """Delete a registered user."""
    if user_uuid not in known_embeddings:
        raise HTTPException(status_code=404, detail="User not found")
    
    name = known_embeddings[user_uuid]["name"]
    del known_embeddings[user_uuid]
    save_db()
    
    return {"message": f"Deleted user: {name}"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": "GPU (CUDA)" if DEVICE_INFO["use_gpu"] else "CPU",
        "onnxruntime_providers": DEVICE_INFO["available_providers"],
        "detection_size": DETECTION_SIZE,
        "video_process_every_n": VIDEO_PROCESS_EVERY_N_FRAMES,
        "stream_process_every_n": STREAM_PROCESS_EVERY_N_FRAMES,
        "registered_users": len(known_embeddings),
        "total_embeddings": sum(len(d["embeddings"]) for d in known_embeddings.values())
    }


# Mount static files last
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
