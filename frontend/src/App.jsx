import { useState, useRef, useEffect } from 'react';
import { Upload, Camera, Video, UserPlus, Scan, Eye, Play, Square, Loader2, AlertCircle, CheckCircle2, Users, Clock, Zap } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('surveillance');

  return (
    <div className="min-h-screen w-full bg-slate-950 text-slate-100 font-sans">
      <div className="px-4 py-8">
        <header className="mb-12 text-center">
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent mb-4">
            God's Eye
          </h1>
          <p className="text-slate-400 text-lg">Advanced Facial Recognition System</p>
        </header>

        <nav className="flex flex-wrap justify-center gap-2 mb-12">
          <NavButton
            active={activeTab === 'surveillance'}
            onClick={() => setActiveTab('surveillance')}
            icon={<Eye size={18} />}
            label="Live Surveillance"
          />
          <NavButton
            active={activeTab === 'scan'}
            onClick={() => setActiveTab('scan')}
            icon={<Scan size={18} />}
            label="Scan Image"
          />
          <NavButton
            active={activeTab === 'video'}
            onClick={() => setActiveTab('video')}
            icon={<Video size={18} />}
            label="Video Analysis"
          />
          <NavButton
            active={activeTab === 'register'}
            onClick={() => setActiveTab('register')}
            icon={<UserPlus size={18} />}
            label="Register"
          />
        </nav>

        <main className="w-full">
          <div className="transition-all duration-300 ease-in-out">
            {activeTab === 'surveillance' && <SurveillancePanel />}
            {activeTab === 'scan' && <ScanPanel />}
            {activeTab === 'video' && <VideoPanel />}
            {activeTab === 'register' && <RegisterPanel />}
          </div>
        </main>
      </div>
    </div>
  );
}

function NavButton({ active, onClick, icon, label }) {
  return (
    <button
      onClick={onClick}
      className={`
        flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all duration-200
        ${active
          ? 'bg-indigo-600/20 text-indigo-300 ring-1 ring-indigo-500/50 shadow-[0_0_15px_-3px_rgba(99,102,241,0.2)]'
          : 'bg-slate-900/50 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-slate-800/50'}
      `}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

function SurveillancePanel() {
  const [url, setUrl] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const imgRef = useRef(null);

  const startStream = async () => {
    if (!url) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('url', url);
      const res = await fetch('/set_camera_url', { method: 'POST', body: formData });
      if (!res.ok) throw new Error('Failed to connect to camera');

      setIsStreaming(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const stopStream = () => {
    setIsStreaming(false);
  };

  return (
    <div className="space-y-6">
      <Card>
        <div className="flex flex-col gap-6">
          <div className="space-y-4">
            <label className="text-sm font-medium text-slate-400 ml-1">RTSP Stream URL</label>
            <div className="flex gap-3">
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="rtsp://user:pass@ip:port/stream"
                disabled={isStreaming}
                className="flex-1 bg-slate-950/50 border border-slate-800 rounded-xl px-4 py-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all font-mono text-sm"
              />
              {!isStreaming ? (
                <button
                  onClick={startStream}
                  disabled={loading || !url}
                  className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-white px-6 rounded-xl font-medium transition-colors flex items-center gap-2"
                >
                  {loading ? <Loader2 className="animate-spin" size={20} /> : <Play size={20} />}
                  Start
                </button>
              ) : (
                <button
                  onClick={stopStream}
                  className="bg-rose-600 hover:bg-rose-500 text-white px-6 rounded-xl font-medium transition-colors flex items-center gap-2"
                >
                  <Square size={20} />
                  Stop
                </button>
              )}
            </div>
          </div>

          {error && (
            <div className="bg-rose-950/30 border border-rose-900/50 text-rose-300 p-4 rounded-xl flex items-center gap-3">
              <AlertCircle size={20} />
              {error}
            </div>
          )}

          <div className="relative aspect-video bg-slate-950 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center group">
            {isStreaming ? (
              <img
                ref={imgRef}
                src={`/video_feed?t=${Date.now()}`}
                alt="Live Stream"
                className="w-full h-full object-contain"
                onError={() => {
                  setIsStreaming(false);
                  setError("Stream connection lost");
                }}
              />
            ) : (
              <div className="text-center p-6">
                <div className="w-16 h-16 bg-slate-900 rounded-full flex items-center justify-center mx-auto mb-4 text-slate-600 group-hover:text-indigo-500 transition-colors">
                  <Camera size={32} />
                </div>
                <p className="text-slate-500">Camera feed is offline</p>
              </div>
            )}

            {isStreaming && (
              <div className="absolute top-4 right-4 flex items-center gap-2 bg-black/60 backdrop-blur pointer-events-none px-3 py-1.5 rounded-full border border-white/10">
                <div className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
                <span className="text-xs font-medium text-white/90">LIVE</span>
              </div>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}

function ScanPanel() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  const handleFile = (e) => {
    if (e.target.files[0]) {
      setImage(URL.createObjectURL(e.target.files[0]));
      setResult(null);
    }
  };

  const scan = async () => {
    const fileInput = document.getElementById('scan-file');
    if (!fileInput.files[0]) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
      const res = await fetch('/recognize', { method: 'POST', body: formData });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (result && result.faces && imgRef.current && canvasRef.current) {
      const img = imgRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;

      const scaleX = img.clientWidth / img.naturalWidth;
      const scaleY = img.clientHeight / img.naturalHeight;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      result.faces.forEach(face => {
        const { top, right, bottom, left } = face.box;
        const x = left * scaleX;
        const y = top * scaleY;
        const w = (right - left) * scaleX;
        const h = (bottom - top) * scaleY;

        const color = face.match ? '#22c55e' : '#f43f5e';

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = color;
        const text = face.match ? `${face.name} ${(face.confidence * 100).toFixed(0)}%` : face.name;
        ctx.font = 'bold 14px Inter, sans-serif';
        const textWidth = ctx.measureText(text).width;

        ctx.fillRect(x, y - 28, textWidth + 16, 28);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, x + 8, y - 8);
      });
    }
  }, [result]);

  return (
    <div className="max-w-4xl mx-auto">
      <Card>
        <div className="space-y-6">
          <div className="grid place-items-center border-2 border-dashed border-slate-800 rounded-2xl p-8 hover:border-slate-700 transition-colors bg-slate-950/30">
            <input
              type="file"
              id="scan-file"
              accept="image/*"
              onChange={handleFile}
              className="hidden"
            />
            <label htmlFor="scan-file" className="cursor-pointer flex flex-col items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-indigo-500/10 text-indigo-400 flex items-center justify-center">
                <Upload size={28} />
              </div>
              <span className="text-slate-400 font-medium">Click to upload photo or drag & drop</span>
            </label>
          </div>

          {image && (
            <div className="relative rounded-xl overflow-hidden bg-black/50 border border-slate-800">
              <img ref={imgRef} src={image} className="w-full h-auto block" />
              <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />
            </div>
          )}

          <button
            onClick={scan}
            disabled={!image || loading}
            className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white py-4 rounded-xl font-bold text-lg transition-all shadow-[0_4px_20px_-4px_rgba(79,70,229,0.3)] hover:translate-y-[-2px]"
          >
            {loading ? <span className="flex items-center justify-center gap-2"><Loader2 className="animate-spin" /> Scanning...</span> : 'Identify Faces'}
          </button>

          {result && (
            <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-800">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <CheckCircle2 size={20} className={result.faces?.length > 0 ? "text-emerald-500" : "text-slate-500"} />
                Analysis Results
              </h3>
              {result.faces && result.faces.length > 0 ? (
                <div className="space-y-2">
                  {result.faces.map((face, i) => (
                    <div key={i} className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg border border-slate-700/50">
                      <span className="font-medium text-slate-200">{face.name}</span>
                      {face.match && (
                        <span className="text-emerald-400 text-sm font-mono bg-emerald-500/10 px-2 py-1 rounded">
                          {(face.confidence * 100).toFixed(1)}% Match
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-slate-500">No faces detected.</p>
              )}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}

function VideoPanel() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Progress tracking state
  const [progress, setProgress] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [processingFps, setProcessingFps] = useState(0);
  const [etaSeconds, setEtaSeconds] = useState(0);
  const [liveDetections, setLiveDetections] = useState([]);
  const [status, setStatus] = useState('idle');

  const eventSourceRef = useRef(null);

  const formatEta = (seconds) => {
    if (seconds <= 0) return '--:--';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const analyze = async () => {
    if (!file) return;

    // Reset state
    setLoading(true);
    setResult(null);
    setError(null);
    setProgress(0);
    setCurrentFrame(0);
    setTotalFrames(0);
    setProcessingFps(0);
    setEtaSeconds(0);
    setLiveDetections([]);
    setStatus('uploading');

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Start the video analysis job
      const res = await fetch('/recognize_video', { method: 'POST', body: formData });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.message || 'Failed to start video analysis');
      }

      const jobId = data.job_id;
      setStatus('processing');

      // Connect to SSE for progress updates
      const eventSource = new EventSource(`/video_progress/${jobId}`);
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        const eventData = JSON.parse(event.data);

        if (eventData.error) {
          setError(eventData.error);
          setLoading(false);
          setStatus('error');
          eventSource.close();
          return;
        }

        if (eventData.type === 'progress') {
          setProgress(eventData.progress);
          setCurrentFrame(eventData.current_frame);
          setTotalFrames(eventData.total_frames);
          setProcessingFps(eventData.processing_fps);
          setEtaSeconds(eventData.eta_seconds);

          // Handle new detections
          if (eventData.new_detections && eventData.new_detections.length > 0) {
            setLiveDetections(prev => [...prev, ...eventData.new_detections]);
          }
        }

        if (eventData.type === 'complete') {
          setResult(eventData.result);
          setLoading(false);
          setStatus('complete');
          setProgress(100);
          eventSource.close();
        }

        if (eventData.type === 'error') {
          setError(eventData.error);
          setLoading(false);
          setStatus('error');
          eventSource.close();
        }
      };

      eventSource.onerror = () => {
        setError('Lost connection to server');
        setLoading(false);
        setStatus('error');
        eventSource.close();
      };

    } catch (e) {
      console.error(e);
      setError(e.message);
      setLoading(false);
      setStatus('error');
    }
  };

  // Cleanup EventSource on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  return (
    <div className="max-w-4xl mx-auto">
      <Card>
        <h2 className="text-2xl font-bold mb-6">Video Forensics</h2>

        <div className="space-y-6">
          <div className="block">
            <label className="block text-sm font-medium text-slate-400 mb-2">Target Video File</label>
            <input
              type="file"
              accept="video/*"
              onChange={e => {
                setFile(e.target.files[0]);
                setResult(null);
                setLiveDetections([]);
                setProgress(0);
                setStatus('idle');
              }}
              disabled={loading}
              className="w-full text-slate-400 file:mr-4 file:py-2.5 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-500/10 file:text-indigo-400 hover:file:bg-indigo-500/20 transition-all cursor-pointer bg-slate-900/50 rounded-xl p-1 disabled:opacity-50"
            />
          </div>

          <button
            onClick={analyze}
            disabled={!file || loading}
            className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white py-3 rounded-xl font-bold transition-all flex items-center justify-center gap-2"
          >
            {loading ? <Loader2 className="animate-spin" /> : <Video size={20} />}
            {loading ? 'Processing Video...' : 'Start Analysis'}
          </button>

          {/* Progress Section */}
          {loading && (
            <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2">
              {/* Progress Bar */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">
                    {status === 'uploading' ? 'Uploading...' : 'Analyzing frames...'}
                  </span>
                  <span className="text-indigo-400 font-mono">{progress.toFixed(1)}%</span>
                </div>
                <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-indigo-600 to-purple-500 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>

              {/* Stats Row */}
              {status === 'processing' && totalFrames > 0 && (
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-slate-900/50 rounded-xl p-3 border border-slate-800/50">
                    <div className="flex items-center gap-2 text-slate-500 text-xs mb-1">
                      <Video size={14} />
                      Frames
                    </div>
                    <div className="text-lg font-semibold text-slate-200">
                      {currentFrame.toLocaleString()} <span className="text-slate-500 text-sm">/ {totalFrames.toLocaleString()}</span>
                    </div>
                  </div>
                  <div className="bg-slate-900/50 rounded-xl p-3 border border-slate-800/50">
                    <div className="flex items-center gap-2 text-slate-500 text-xs mb-1">
                      <Zap size={14} />
                      Speed
                    </div>
                    <div className="text-lg font-semibold text-slate-200">
                      {processingFps} <span className="text-slate-500 text-sm">fps</span>
                    </div>
                  </div>
                  <div className="bg-slate-900/50 rounded-xl p-3 border border-slate-800/50">
                    <div className="flex items-center gap-2 text-slate-500 text-xs mb-1">
                      <Clock size={14} />
                      ETA
                    </div>
                    <div className="text-lg font-semibold text-slate-200 font-mono">
                      {formatEta(etaSeconds)}
                    </div>
                  </div>
                </div>
              )}

              {/* Live Detections */}
              {liveDetections.length > 0 && (
                <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-800/50">
                  <div className="flex items-center gap-2 mb-3">
                    <Users size={16} className="text-emerald-400" />
                    <span className="text-sm font-medium text-slate-300">Live Detections</span>
                    <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full">
                      {liveDetections.filter(d => d.is_new).length} found
                    </span>
                  </div>
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {liveDetections.map((detection, idx) => (
                      <div
                        key={idx}
                        className={`flex items-center justify-between p-2 rounded-lg border transition-all ${detection.is_new
                            ? 'bg-emerald-500/10 border-emerald-500/30 animate-in fade-in slide-in-from-left-2'
                            : 'bg-slate-800/30 border-slate-700/30'
                          }`}
                      >
                        <div className="flex items-center gap-2">
                          {detection.is_new && (
                            <span className="text-xs bg-emerald-500 text-white px-1.5 py-0.5 rounded font-medium">NEW</span>
                          )}
                          <span className="text-slate-200 font-medium">{detection.name}</span>
                        </div>
                        <div className="flex items-center gap-3 text-xs">
                          <span className="text-emerald-400 font-mono">
                            {(detection.confidence * 100).toFixed(1)}%
                          </span>
                          <span className="text-slate-500">
                            Frame #{detection.frame_number}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="bg-rose-950/30 border border-rose-900/50 text-rose-300 p-4 rounded-xl flex items-center gap-3">
              <AlertCircle size={20} />
              {error}
            </div>
          )}

          {/* Final Results */}
          {result && (
            <div className="bg-slate-900/50 p-6 rounded-xl border border-slate-800 animate-in fade-in slide-in-from-bottom-4">
              <h3 className="text-lg font-semibold text-emerald-400 mb-4 flex items-center gap-2">
                <CheckCircle2 size={20} /> Analysis Complete
              </h3>

              {/* Summary Stats */}
              {result.processing_time && (
                <div className="flex gap-4 mb-4 text-sm text-slate-400">
                  <span>Processed in <strong className="text-slate-200">{result.processing_time}s</strong></span>
                  <span>•</span>
                  <span><strong className="text-slate-200">{result.total_frames_analyzed}</strong> frames analyzed</span>
                </div>
              )}

              {result.found_people && result.found_people.length > 0 ? (
                <div>
                  <p className="text-slate-400 text-sm mb-4">Identified Persons (Best Detection Frame):</p>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {result.found_people.map((person, idx) => (
                      <div key={idx} className="relative group rounded-xl overflow-hidden border border-slate-700 hover:border-indigo-500 transition-all bg-slate-800/50">
                        <div className="aspect-video overflow-hidden bg-black">
                          <img
                            src={person.frame_url}
                            alt={person.name}
                            className="w-full h-full object-contain group-hover:scale-105 transition-transform duration-300"
                          />
                        </div>
                        <div className="p-3 space-y-1">
                          <p className="font-semibold text-slate-200">{person.name}</p>
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-emerald-400 font-mono bg-emerald-500/10 px-2 py-1 rounded">
                              {(person.confidence * 100).toFixed(1)}% match
                            </span>
                            <span className="text-slate-500">
                              Frame #{person.frame_number}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="text-slate-500">No registered faces found in the video.</p>
              )}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}

function RegisterPanel() {
  const [mode, setMode] = useState('new'); // 'new' or 'existing'
  const [name, setName] = useState('');
  const [selectedUser, setSelectedUser] = useState('');
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      const res = await fetch('/users');
      if (res.ok) {
        const data = await res.json();
        setUsers(data);
      }
    } catch (e) {
      console.error("Failed to fetch users", e);
    }
  };

  const register = async () => {
    const files = fileInputRef.current.files;
    const targetName = mode === 'new' ? name : selectedUser;

    if (!targetName || files.length === 0) return;

    setLoading(true);
    setMessage(null);

    const formData = new FormData();
    formData.append('name', targetName);
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    try {
      const res = await fetch('/register', { method: 'POST', body: formData });
      const data = await res.json();
      if (res.ok) {
        setMessage({ type: 'success', text: data.message });
        setName('');
        setSelectedUser('');
        fileInputRef.current.value = '';
        fetchUsers(); // Refresh list to update counts
      } else {
        setMessage({ type: 'error', text: data.message });
      }
    } catch {
      setMessage({ type: 'error', text: 'Network error occurred' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto w-full">
      <Card>
        <div className="flex flex-col md:flex-row items-center gap-4 mb-6 text-center md:text-left">
          <div className="w-12 h-12 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center shrink-0">
            <UserPlus size={24} />
          </div>
          <div>
            <h2 className="text-2xl font-bold">Manage Subjects</h2>
            <p className="text-slate-500 text-sm">Register new people or add photos to existing ones</p>
          </div>
        </div>

        <div className="space-y-6">
          {/* Toggle */}
          <div className="flex bg-slate-950/50 p-1 rounded-xl border border-slate-800/50">
            <button
              onClick={() => setMode('new')}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${mode === 'new' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200'}`}
            >
              New Subject
            </button>
            <button
              onClick={() => setMode('existing')}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${mode === 'existing' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200'}`}
            >
              Existing Subject
            </button>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1.5">
              {mode === 'new' ? 'Full Name' : 'Select Subject'}
            </label>

            {mode === 'new' ? (
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full bg-slate-950/50 border border-slate-800 rounded-xl px-4 py-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all placeholder:text-slate-600"
                placeholder="e.g. John Doe"
              />
            ) : (
              <div className="relative">
                <select
                  value={selectedUser}
                  onChange={(e) => setSelectedUser(e.target.value)}
                  className="w-full bg-slate-950/50 border border-slate-800 rounded-xl px-4 py-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all appearance-none cursor-pointer"
                >
                  <option value="">-- Choose a subject --</option>
                  {users.map(u => (
                    <option key={u.uuid} value={u.name}>
                      {u.name} ({u.count} photos)
                    </option>
                  ))}
                </select>
                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-slate-500">
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2.5 4.5L6 8L9.5 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                </div>
              </div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1.5">Reference Photos</label>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*"
              className="w-full text-slate-400 file:mr-4 file:py-2.5 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-slate-800 file:text-slate-300 hover:file:bg-slate-700 transition-all cursor-pointer bg-slate-900/50 rounded-xl p-1"
            />
            <p className="text-xs text-slate-500 mt-2">Upload multiple angles for better accuracy.</p>
          </div>

          <button
            onClick={register}
            disabled={loading || (mode === 'new' ? !name : !selectedUser)}
            className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white py-3.5 rounded-xl font-bold transition-all shadow-[0_4px_20px_-4px_rgba(79,70,229,0.3)] mt-2"
          >
            {loading ? <Loader2 className="animate-spin mx-auto" /> : (mode === 'new' ? 'Register Subject' : 'Update Subject')}
          </button>

          {message && (
            <div className={`p-4 rounded-xl text-sm font-medium flex items-center gap-3 animate-in fade-in slide-in-from-bottom-2 ${message.type === 'success'
              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
              : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
              }`}>
              {message.type === 'success' ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
              {message.text}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}

function Card({ children }) {
  return (
    <div className="bg-slate-900/40 backdrop-blur-sm border border-slate-800/60 rounded-3xl p-6 md:p-8 shadow-xl">
      {children}
    </div>
  )
}

export default App;
