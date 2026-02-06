from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import face_recognition
import numpy as np
import uuid
import os
import pickle
from typing import Dict, List
from PIL import Image
import io
import cv2
import tempfile
import shutil

app = FastAPI()

# Create static directory if it doesn't exist (just in case)
os.makedirs("static", exist_ok=True)
os.makedirs("data", exist_ok=True)



# Simple storage
DB_FILE = "data/embeddings.pkl"
# Structure: uuid -> {"name": str, "embeddings": List[np.array]}
known_embeddings: Dict[str, dict] = {}

def load_db():
    global known_embeddings
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "rb") as f:
                data = pickle.load(f)
                # Migration check: if old format, convert
                if data and isinstance(list(data.values())[0], dict) and "embedding" in list(data.values())[0]:
                    print("Migrating DB to new format...")
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
        except Exception as e:
            print(f"Error loading DB: {e}")
            known_embeddings = {}

def save_db():
    with open(DB_FILE, "wb") as f:
        pickle.dump(known_embeddings, f)

# Load DB on startup
load_db()

@app.post("/register")
async def register(name: str = Form(...), files: List[UploadFile] = File(...)):
    try:
        # Check if name already exists to append to
        user_uuid = None
        for uid, data in known_embeddings.items():
            if data["name"].lower() == name.lower():
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
            # Read image
            contents = await file.read()
            image = face_recognition.load_image_file(io.BytesIO(contents))
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image,num_jitters=50)
            
            if encodings:
                # Add all faces found or just the first? Usually one per training image is safer
                # lets assume one person per image for registration
                known_embeddings[user_uuid]["embeddings"].append(encodings[0])
                processed_count += 1
        
        if processed_count == 0:
             # If it was a new user and no faces found, maybe cleanup? 
             # But keeping the ID is fine, just empty embeddings.
             return JSONResponse(status_code=400, content={"message": "No faces found in any of the uploaded images"})

        save_db()
        
        return {
            "message": f"Registered {name} successfully. processed {processed_count} images.", 
            "uuid": user_uuid,
            "total_embeddings": len(known_embeddings[user_uuid]["embeddings"])
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        if not known_embeddings:
             return JSONResponse(status_code=400, content={"message": "No registered faces in database"})

        # Read image
        contents = await file.read()
        image = face_recognition.load_image_file(io.BytesIO(contents))
        
        # Get face locations and encodings
        # We need locations to draw boxes later
        face_locations = face_recognition.face_locations(image)
        unknown_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not unknown_encodings:
            return JSONResponse(status_code=400, content={"message": "No face found in image"})
        
        # Prepare flattened lists for batch comparison
        known_uuids = [] # map index back to uuid
        known_encs = []
        known_names = {}
        
        for uid, data in known_embeddings.items():
            known_names[uid] = data["name"]
            for emb in data["embeddings"]:
                known_uuids.append(uid)
                known_encs.append(emb)
        
        found_faces = []

        # Iterate through each face found in the uploaded image
        for (top, right, bottom, left), unknown_encoding in zip(face_locations, unknown_encodings):
            matches = face_recognition.compare_faces(known_encs, unknown_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encs, unknown_encoding)
            
            best_match_index = np.argmin(face_distances)
            
            face_data = {
                "box": {"top": top, "right": right, "bottom": bottom, "left": left},
                "match": False,
                "name": "Unknown",
                "confidence": 0.0
            }

            if len(matches) > 0 and matches[best_match_index]:
                match_uuid = known_uuids[best_match_index]
                face_data["match"] = True
                face_data["name"] = known_names[match_uuid]
                face_data["uuid"] = match_uuid
                face_data["confidence"] = float(1 - face_distances[best_match_index])
            
            found_faces.append(face_data)
            
        return {
            "message": f"Found {len(found_faces)} faces",
            "faces": found_faces
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/recognize_video")
async def recognize_video(file: UploadFile = File(...)):
    try:
        if not known_embeddings:
            return JSONResponse(status_code=400, content={"message": "No registered faces in database"})

        # Save to temp file because cv2.VideoCapture needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        # Process video
        video_capture = cv2.VideoCapture(temp_path)
        
        found_people = set()
        frame_count = 0
        process_every_n_frames = 10 # Optimization: Process 1 frame every ~1 second (assuming 30fps)
        
        # Prepare comparison data
        known_uuids = [] 
        known_encs = []
        known_names = {}
        for uid, data in known_embeddings.items():
            known_names[uid] = data["name"]
            for emb in data["embeddings"]:
                known_uuids.append(uid)
                known_encs.append(emb)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % process_every_n_frames != 0:
                continue

            # Optimization: Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if not face_locations:
                continue
                
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_encs, face_encoding, tolerance=0.5)
                
                if True in matches:
                    face_distances = face_recognition.face_distance(known_encs, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        uuid = known_uuids[best_match_index]
                        name = known_names[uuid]
                        found_people.add(name)

        video_capture.release()
        os.remove(temp_path)

        return {
            "message": "Video analysis complete",
            "found_names": list(found_people)
        }

    except Exception as e:
        # cleanup if error changes flow
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(status_code=500, content={"message": str(e)})


# Mount static files at the end to avoid conflicts with API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
