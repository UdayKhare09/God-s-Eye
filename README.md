# 👁️ God's Eye 

**Advanced Facial Recognition & Surveillance System**

God's Eye is an industry-grade, visually stunning web application designed for comprehensive facial recognition, live surveillance, and video forensics. It merges an immersive, 3D-accelerated interactive landing page with a robust, highly functional analytical dashboard.

---

## ✨ Features

### 1. Immersive Landing Page (`/`)
- **3D Visualizations & Post-Processing**: Powered by `Three.js` and `@react-three/fiber` to create an engaging visual experience.
- **Smooth Animations & Scroll Dynamics**: Built with `Framer Motion`, `GSAP` (ScrollTrigger), and `Lenis` for seamless, native-feeling scroll behavior.
- **Component Highlights**: Features modular aesthetic sections including *Hero*, *Showcase*, *Vision*, *PhotoCarousel*, *Deploy*, and *Team*.

### 2. Analytical Dashboard (`/dashboard`)
- **Live Surveillance**: Connect to an RTSP/HTTP stream and watch a live camera feed with real-time tracking overlay.
- **Scan Image (Facial Recognition)**: Upload or drag & drop photographs to instantly identify individuals within the image with high confidence scoring.
- **Video Forensics**: Upload archival video footage. The system processes frames sequentially, extracting and classifying registered facial signatures in bulk.
- **Subject Management**: Register new individuals or append additional images to existing subjects for higher identification accuracy. 

---

## 🛠️ Technology Stack

**Frontend:**
- **Framework**: React 19, Vite, Tailwind CSS v4
- **Routing**: React Router DOM v7
- **3D & Canvas**: Three.js, React Three Fiber, React Three Drei, Postprocessing
- **Animation & Scrolling**: Framer Motion, GSAP, Lenis Studio

**Backend:**
- **Framework**: FastAPI, Uvicorn
- **Computer Vision**: OpenCV (cv2)
- **Facial Recognition**: ONNX Runtime (CPU/GPU acceleration) & specialized embedding models

---

## 📂 Project Structure

```text
God-s-Eye/
├── main.py                       # Python FastAPI Backend Server
├── requirements.txt              # Python dependencies
├── static/                       # Compiled frontend assets and server media
├── frontend/                     # Main Web Application Directory
│   ├── package.json              # App configuration and dependencies
│   ├── vite.config.js            # Build configuration, chunk splitting
│   ├── src/
│   │   ├── main.jsx              # React mounting root
│   │   ├── App.jsx               # Application Router (Landing vs Dashboard)
│   │   ├── pages/                # Contains layout-level components
│   │   ├── components/           # Reusable generic UI elements 
│   │   └── sections/             # Marketing sections for the Landing Page
```

---

## 🚀 Getting Started (Full Stack)

This project requires both the Python Backend and the React Frontend to be built and hosted locally.

### 1. Frontend Build Setup
First, we compile the frontend application and inject it into the backend's static rendering folder.

```bash
# Navigate to the frontend directory
cd frontend

# Install JavaScript dependencies
npm install

# Build the React app (this automatically copies the output to the backend's static folder)
npm run build

# Return to the root folder
cd ..
```

### 2. Backend Setup & Virtual Environment
Ensure you have **Python 3.9+** and `pip` installed. We highly recommend using a Virtual Environment.

```bash
# 1. Create a virtual environment named .venv in the root directory
python3 -m venv .venv

# 2. Activate the virtual environment
# On Linux/MacOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 3. Install the required Python packages
pip install -r requirements.txt
```

### 3. Running the Application
Once the frontend is built and the Python packages are installed, you just need a single command to run the whole project. 

Make sure your virtual environment is still activated (`(.venv)` should be visible in your terminal prompt) and you are in the root directory (`/God-s-Eye`):

```bash
python3 main.py
```
*Wait a few seconds for the ML Models to load.*

You will see a success message indicating the server is running. Navigate to **http://localhost:8000/** in your web browser. This serves the completely unified application (React + FastAPI).

---

## 🕹️ Navigating the Application Features

1. **Deploy System (Landing Page)**: On the main homepage layer, scroll down and click **Initialize System** (or click the top left `Back to Home` arrow returning from the dashboard) to transition contexts.
2. **Live Surveillance (`/dashboard`)**: Paste any active camera stream URL (e.g. your local IP Camera or DroidCam app stream such as `http://192.168.1.100:8080/video`) to perform active AI detection over the frame buffer.
3. **Register Subjects (`/dashboard`)**: Go here first! Type in a name (like "John Doe") and upload multiple face shots. The embedding models will instantly generate a reference vector of their face to store.
4. **Scan Image (`/dashboard`)**: Upload a multi-person photo. The system will detect all faces and compare them against your previously registered subjects, providing accurate matching percentages.
5. **Video Forensics (`/dashboard`)**: Upload an archived `.mp4` surveillance feed. The backend will unpack the video frame by frame, isolating all instances of registered targets for bulk retrieval.
