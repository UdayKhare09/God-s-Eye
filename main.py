from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import face_recognition
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
from functools import lru_cache
import threading
import logging

from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DB_FILE = "data/embeddings.pkl"
RECOGNITION_TOLERANCE = 0.5
FRAME_SCALE_FACTOR = 0.5
VIDEO_PROCESS_EVERY_N_FRAMES = 10
STREAM_PROCESS_EVERY_N_FRAMES = 30
REGISTRATION_NUM_JITTERS = 50  # Higher = more accurate but slower

# Thread-safe database lock
db_lock = threading.Lock()

# Structure: uuid -> {"name": str, "embeddings": List[np.array]}
known_embeddings: Dict[str, dict] = {}

# Cached flattened embeddings for fast recognition
# This avoids rebuilding the list on every recognition call
_cached_db_lists: Optional[Tuple[List[str], List[np.ndarray], Dict[str, str]]] = None
_cache_valid = False


def invalidate_cache():
    """Mark the cache as invalid when embeddings change."""
    global _cache_valid
    _cache_valid = False


def get_db_lists() -> Tuple[List[str], List[np.ndarray], Dict[str, str]]:
    """
    Get flattened lists from known_embeddings for recognition.
    Uses caching to avoid rebuilding on every call.
    """
    global _cached_db_lists, _cache_valid
    
    if _cache_valid and _cached_db_lists is not None:
        return _cached_db_lists
    
    k_uuids: List[str] = []
    k_encs: List[np.ndarray] = []
    k_names: Dict[str, str] = {}
    
    for uid, data in known_embeddings.items():
        k_names[uid] = data["name"]
        for emb in data["embeddings"]:
            k_uuids.append(uid)
            k_encs.append(emb)
    
    _cached_db_lists = (k_uuids, k_encs, k_names)
    _cache_valid = True
    return _cached_db_lists


def load_db():
    """Load embeddings database from disk."""
    global known_embeddings
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "rb") as f:
                data = pickle.load(f)
                # Migration check: if old format (single embedding), convert to list
                if data and isinstance(list(data.values())[0], dict) and "embedding" in list(data.values())[0]:
                    logger.info("Migrating DB to new format...")
                    new_data = {}
                    for uid, info in data.items():
                        new_data[uid] = {
                            "name": info["name"],
                            "embeddings": [info["embedding"]]
                        }
                    known_embeddings = new_data
                    save_db()
                else:
                    known_embeddings = data
            invalidate_cache()
            logger.info(f"Loaded {len(known_embeddings)} users from database")
        except Exception as e:
            logger.error(f"Error loading DB: {e}")
            known_embeddings = {}


def save_db():
    """Save embeddings database to disk (thread-safe)."""
    with db_lock:
        with open(DB_FILE, "wb") as f:
            pickle.dump(known_embeddings, f)
    invalidate_cache()


def process_image_bytes(contents: bytes) -> np.ndarray:
    """Convert image bytes to numpy array for face_recognition."""
    return face_recognition.load_image_file(io.BytesIO(contents))


def find_best_match(
    face_encoding: np.ndarray,
    known_encs: List[np.ndarray],
    known_uuids: List[str],
    known_names: Dict[str, str],
    tolerance: float = RECOGNITION_TOLERANCE
) -> Tuple[bool, str, str, float]:
    """
    Find the best matching face from known encodings.
    
    Returns:
        (is_match, uuid, name, confidence)
    """
    if not known_encs:
        return False, "", "Unknown", 0.0
    
    matches = face_recognition.compare_faces(known_encs, face_encoding, tolerance=tolerance)
    face_distances = face_recognition.face_distance(known_encs, face_encoding)
    
    best_match_index = np.argmin(face_distances)
    
    if matches[best_match_index]:
        match_uuid = known_uuids[best_match_index]
        return True, match_uuid, known_names[match_uuid], float(1 - face_distances[best_match_index])
    
    return False, "", "Unknown", 0.0


def process_frame_for_recognition(
    frame: np.ndarray,
    scale: float = FRAME_SCALE_FACTOR
) -> Tuple[np.ndarray, List, float]:
    """
    Prepare a frame for face recognition processing.
    
    Returns:
        (rgb_frame, face_locations, scale_used)
    """
    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    face_locations = face_recognition.face_locations(rgb_frame)
    return rgb_frame, face_locations, scale


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.makedirs("static", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    load_db()
    logger.info("Application started")
    yield
    # Shutdown
    logger.info("Application shutting down")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    Uses high num_jitters for better encoding accuracy.
    """
    try:
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
            image = process_image_bytes(contents)
            
            # Use higher num_jitters for registration accuracy
            encodings = face_recognition.face_encodings(image, num_jitters=REGISTRATION_NUM_JITTERS)
            
            if encodings:
                # Assume one person per training image
                known_embeddings[user_uuid]["embeddings"].append(encodings[0])
                processed_count += 1
        
        if processed_count == 0:
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
        
        # Get face locations and encodings
        face_locations = face_recognition.face_locations(image)
        unknown_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not unknown_encodings:
            return JSONResponse(
                status_code=400,
                content={"message": "No face found in image"}
            )
        
        # Get cached recognition data
        known_uuids, known_encs, known_names = get_db_lists()
        
        found_faces = []
        for (top, right, bottom, left), unknown_encoding in zip(face_locations, unknown_encodings):
            is_match, match_uuid, name, confidence = find_best_match(
                unknown_encoding, known_encs, known_uuids, known_names
            )
            
            face_data = {
                "box": {"top": top, "right": right, "bottom": bottom, "left": left},
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

        # Save to temp file (cv2.VideoCapture needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        video_capture = cv2.VideoCapture(temp_path)
        
        if not video_capture.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video file")
        
        found_people = set()
        frame_count = 0
        
        # Get cached recognition data
        known_uuids, known_encs, known_names = get_db_lists()
        
        if not known_encs:
            video_capture.release()
            return {
                "message": "Video analysis complete",
                "found_names": []
            }

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % VIDEO_PROCESS_EVERY_N_FRAMES != 0:
                continue

            rgb_frame, face_locations, _ = process_frame_for_recognition(frame)
            
            if not face_locations:
                continue
                
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                is_match, _, name, _ = find_best_match(
                    face_encoding, known_encs, known_uuids, known_names
                )
                if is_match:
                    found_people.add(name)

        video_capture.release()

        return {
            "message": "Video analysis complete",
            "found_names": list(found_people)
        }

    except Exception as e:
        logger.error(f"Video recognition error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# Global state for RTSP streaming
camera_config = {
    "url": None
}


def generate_frames(rtsp_url: str):
    """
    Generator that yields MJPEG frames from an RTSP stream with face recognition overlays.
    Processes faces every N frames to maintain smooth streaming.
    """
    camera = cv2.VideoCapture(rtsp_url)
    if not camera.isOpened():
        logger.error(f"Cannot open camera: {rtsp_url}")
        return

    # Get cached recognition data
    k_uuids, k_encs, k_names = get_db_lists()
    
    frame_count = 0
    detected_faces: List[Tuple[int, int, int, int, str, Tuple[int, int, int]]] = []
    scale_inverse = int(1 / FRAME_SCALE_FACTOR)
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            frame_count += 1

            # Process faces periodically to prevent lag
            if frame_count % STREAM_PROCESS_EVERY_N_FRAMES == 0 and k_encs:
                rgb_frame, face_locations, _ = process_frame_for_recognition(frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                new_faces = []
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Scale coordinates back to original size
                    top *= scale_inverse
                    right *= scale_inverse
                    bottom *= scale_inverse
                    left *= scale_inverse

                    is_match, _, name, _ = find_best_match(
                        face_encoding, k_encs, k_uuids, k_names
                    )
                    
                    color = (0, 255, 0) if is_match else (0, 0, 255)  # Green/Red in BGR
                    new_faces.append((top, right, bottom, left, name, color))
                
                detected_faces = new_faces

            # Draw detected faces on every frame for smooth visualization
            for (top, right, bottom, left, name, color) in detected_faces:
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

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
    
    # Quick connection check
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


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "registered_users": len(known_embeddings),
        "total_embeddings": sum(len(d["embeddings"]) for d in known_embeddings.values())
    }


# Mount static files last to avoid conflicts with API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
