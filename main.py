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
import insightface
from insightface.app import FaceAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Tune these for your CPU
# ============================================================================
DB_FILE = "data/embeddings_insightface.pkl"  # New DB file for InsightFace embeddings
RECOGNITION_THRESHOLD = 0.5  # Cosine distance threshold (lower = stricter, higher = more lenient)
VIDEO_PROCESS_EVERY_N_FRAMES = 5  # Process more frames since InsightFace is faster
STREAM_PROCESS_EVERY_N_FRAMES = 5  # More frequent updates for streams
DETECTION_SIZE = 640  # Detection input size (320/480/640 - lower = faster)
MODEL_NAME = "buffalo_l"  # buffalo_sc (fast) | buffalo_s | buffalo_l (most accurate, best for angles)

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


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
def get_face_analyzer() -> FaceAnalysis:
    """
    Lazy load the InsightFace model with CPU optimizations.
    Uses thread-safe singleton pattern.
    """
    global _face_analyzer
    
    if _face_analyzer is not None:
        return _face_analyzer
    
    with model_lock:
        if _face_analyzer is not None:
            return _face_analyzer
        
        logger.info(f"Loading InsightFace model: {MODEL_NAME}")
        start_time = time.time()
        
        # Initialize with CPU provider
        app = FaceAnalysis(
            name=MODEL_NAME,
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition']  # Skip age/gender for speed
        )
        
        # Prepare with detection size (smaller = faster)
        app.prepare(ctx_id=-1, det_size=(DETECTION_SIZE, DETECTION_SIZE))
        
        _face_analyzer = app
        logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
        
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
    """Analyze a video file and identify all recognized faces."""
    temp_path = None
    try:
        if not known_embeddings:
            return JSONResponse(
                status_code=400,
                content={"message": "No registered faces in database"}
            )
        
        # Ensure detections directory exists
        detections_dir = "static/detections"
        os.makedirs(detections_dir, exist_ok=True)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
        
        video_capture = cv2.VideoCapture(temp_path)
        
        if not video_capture.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video file")
        
        analyzer = get_face_analyzer()
        known_uuids, known_matrix, known_names = get_cached_embeddings()
        
        if len(known_matrix) == 0:
            video_capture.release()
            return {"message": "Video analysis complete", "found_people": []}
        
        # Track best detection of each person
        # name -> {"frame": ndarray, "confidence": float, "frame_number": int, "box": tuple}
        found_people = {}
        frame_count = 0
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % VIDEO_PROCESS_EVERY_N_FRAMES != 0:
                continue
            
            faces = analyzer.get(frame)
            
            for face in faces:
                embedding = face.embedding / np.linalg.norm(face.embedding)
                bbox = face.bbox.astype(int)
                
                is_match, _, name, confidence = find_best_match(
                    embedding, known_uuids, known_matrix, known_names
                )
                
                if is_match:
                    if name not in found_people or confidence > found_people[name]["confidence"]:
                        found_people[name] = {
                            "frame": frame.copy(),
                            "confidence": confidence,
                            "frame_number": frame_count,
                            "box": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                        }
        
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
        
        return {
            "message": "Video analysis complete",
            "found_people": result_people
        }
        
    except Exception as e:
        logger.error(f"Video recognition error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


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
        "registered_users": len(known_embeddings),
        "total_embeddings": sum(len(d["embeddings"]) for d in known_embeddings.values())
    }


# Mount static files last
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
