from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
from harness.orchestrator import VoiceRAGOrchestrator
from retrieval.vector_store import get_vector_store
from config.logger import logger
import shutil
import tempfile
import os

orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    col = get_vector_store()
    logger.info(f"Loaded ChromaDB vector index with {col.count()} passages.")
    orchestrator = VoiceRAGOrchestrator()
    yield

app = FastAPI(title="HH Goa Voice-RAG Service", lifespan=lifespan)

@app.post("/api/query")
def handle_query(
    audio: UploadFile = File(None),
    text_override: str = Form(None)
):
    temp_path = None
    if audio and audio.filename:
        suffix = os.path.splitext(audio.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            temp_path = tmp.name

    try:
        result = orchestrator.process(audio_path=temp_path, text_override=text_override)
        return JSONResponse(content=result)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.get("/api/metrics")
def get_metrics():
    return orchestrator.metrics_db.compute_percentiles()

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HH Goa 2026 | Voice-Enabled RAG</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {
                font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            .font-mono {
                font-family: 'JetBrains Mono', monospace;
            }
            /* Custom glassmorphism & gradients */
            .glass-panel {
                background: rgba(15, 23, 42, 0.75);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            }
            .glass-input {
                background: rgba(2, 6, 23, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.2s ease-in-out;
            }
            .glass-input:focus {
                border-color: #10b981;
                box-shadow: 0 0 15px rgba(16, 185, 129, 0.25);
            }
            .glow-emerald {
                box-shadow: 0 0 25px -5px rgba(16, 185, 129, 0.3);
            }
            .glow-rose {
                box-shadow: 0 0 25px -5px rgba(225, 29, 72, 0.4);
            }
            .bg-mesh {
                background-color: #020617;
                background-image: 
                    radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.12) 0px, transparent 50%),
                    radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
                    radial-gradient(at 50% 100%, rgba(15, 23, 42, 0.8) 0px, transparent 50%);
            }
            /* Custom styled scrollbars */
            ::-webkit-scrollbar {
                width: 6px;
                height: 6px;
            }
            ::-webkit-scrollbar-track {
                background: #020617;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb {
                background: #334155;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #10b981;
            }
        </style>
    </head>
    <body class="bg-mesh text-slate-100 min-h-screen p-6 md:p-12 relative overflow-x-hidden">
        <!-- Ambient background glows -->
        <div class="fixed top-0 left-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none"></div>
        <div class="fixed bottom-0 right-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none"></div>

        <div class="max-w-3xl mx-auto space-y-6 relative z-10">
            <header class="border-b border-slate-800/80 pb-5">
                <div class="flex items-center justify-between">
                    <span class="text-xs uppercase tracking-widest text-emerald-400 font-mono font-bold bg-emerald-950/60 border border-emerald-800/50 px-2.5 py-1 rounded-md flex items-center gap-1.5">
                        <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                        HH Goa 2026 · Task #2
                    </span>
                    <div class="flex items-center gap-3">
                        <span class="text-xs text-indigo-300 font-mono font-semibold bg-indigo-950/60 border border-indigo-800/50 px-2.5 py-1 rounded-md flex items-center gap-1.5">
                            <svg class="w-3.5 h-3.5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"/></svg>
                            Team: No More Tokens
                        </span>
                        <span class="text-xs text-slate-500 font-mono hidden sm:flex items-center gap-1">
                            <svg class="w-3.5 h-3.5 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14"/></svg>
                            RAGInGoa
                        </span>
                    </div>
                </div>
                <h1 class="text-3xl md:text-4xl font-extrabold mt-3 tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent flex items-center gap-3">
                    <svg class="w-8 h-8 text-emerald-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/>
                    </svg>
                    Voice-Enabled Grounded RAG
                </h1>
                <p class="text-slate-400 text-sm mt-2 leading-relaxed">Built by <span class="text-slate-200 font-semibold">Team No More Tokens</span> · ElevenLabs & Sarvam STT · Multi-Strategy Chunking · Latency Analytics · Refusal Guardrail</p>
            </header>

            <!-- Input Controls -->
            <div class="glass-panel rounded-2xl p-6 md:p-7 space-y-6 border border-slate-800/80">
                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-2">
                        <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg>
                        Voice Input (Live Microphone or Audio Upload)
                    </label>
                    <div class="flex flex-wrap items-center gap-3">
                        <button id="recordBtn" onclick="toggleRecording()" class="flex items-center gap-2.5 bg-gradient-to-r from-rose-600 to-rose-700 hover:from-rose-500 hover:to-rose-600 text-white font-semibold px-6 py-3 rounded-xl text-sm transition-all duration-200 shadow-lg glow-rose active:scale-95">
                            <span id="recordIcon" class="w-2.5 h-2.5 rounded-full bg-white animate-pulse"></span>
                            <span id="recordLabel">Start Live Mic</span>
                        </button>

                        <label class="cursor-pointer flex items-center gap-2 bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 border border-slate-700/80 font-medium px-5 py-3 rounded-xl text-sm transition-all duration-200 hover:border-slate-600 active:scale-95">
                            <svg class="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg>
                            <span>Upload .WAV</span>
                            <input id="audioFileInput" type="file" accept="audio/*" class="hidden" onchange="uploadAudioFile(this)">
                        </label>
                        <span id="recordStatus" class="text-xs text-slate-400 font-mono w-full md:w-auto flex items-center gap-1.5"></span>
                    </div>
                </div>

                <div class="relative flex items-center">
                    <div class="flex-grow border-t border-slate-800/80"></div>
                    <span class="flex-shrink mx-4 text-xs uppercase tracking-wider text-slate-500 font-mono font-medium">or test with query text</span>
                    <div class="flex-grow border-t border-slate-800/80"></div>
                </div>

                <div class="flex gap-3">
                    <input id="textInput" type="text" placeholder="e.g. What is the definition of honesty and integrity?" class="flex-1 glass-input rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none">
                    <button onclick="sendTextQuery()" class="bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-500 hover:to-emerald-600 font-semibold px-6 py-3 rounded-xl text-sm transition-all duration-200 shadow-lg glow-emerald active:scale-95 flex items-center gap-2">
                        <span>Submit</span>
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
                    </button>
                </div>
            </div>

            <!-- Loading Spinner -->
            <div id="loader" class="hidden text-center py-8 glass-panel rounded-2xl border border-slate-800/80">
                <div class="relative inline-flex items-center justify-center">
                    <div class="w-10 h-10 border-4 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin"></div>
                    <svg class="w-4 h-4 text-emerald-400 absolute" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                </div>
                <div class="text-xs text-slate-400 mt-3 font-mono">Transcribing Speech, Retrieving Context & Formulating Answer...</div>
            </div>

            <!-- Results Card -->
            <div id="outputCard" class="hidden glass-panel rounded-2xl p-6 md:p-7 space-y-5 border border-slate-800/80">
                <div class="flex justify-between items-center border-b border-slate-800/80 pb-4">
                    <h3 class="text-base font-bold text-white flex items-center gap-2">
                        <svg class="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                        Pipeline Execution Result
                    </h3>
                    <span id="statusBadge" class="px-3.5 py-1.5 text-xs rounded-full font-mono font-semibold flex items-center gap-1.5"></span>
                </div>
                
                <div class="space-y-2">
                    <div id="sttEngineLabel" class="text-xs text-slate-400 font-mono uppercase tracking-wider flex items-center gap-2 px-1">
                        <svg class="w-4 h-4 text-slate-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg>
                        <span>Speech Transcript</span>
                    </div>
                    <div id="transcriptText" class="text-slate-100 font-medium bg-slate-950/90 px-5 py-4 rounded-xl border border-slate-800/80 text-sm leading-relaxed shadow-inner"></div>
                </div>

                <!-- Increased Height & Scrollable Answer Box -->
                <div class="space-y-2 pt-1">
                    <div class="text-xs text-slate-400 font-mono uppercase tracking-wider flex items-center gap-2 px-1">
                        <svg class="w-4 h-4 text-emerald-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>
                        <span>Grounded Answer (Groq Meta LLaMA-3.1)</span>
                    </div>
                    <div id="answerText" class="text-emerald-300 font-normal leading-relaxed bg-slate-950/90 px-5 py-4 rounded-xl border border-slate-800/80 min-h-[140px] max-h-[300px] overflow-y-auto whitespace-pre-wrap text-sm shadow-inner"></div>
                </div>

                <div class="space-y-2 pt-1">
                    <div class="text-xs text-slate-400 font-mono uppercase tracking-wider flex items-center gap-2 px-1">
                        <svg class="w-4 h-4 text-slate-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                        <span>Empirical Latency Breakdown</span>
                    </div>
                    <div id="timingsBreakdown" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 font-mono text-xs text-center"></div>
                </div>
            </div>
        </div>

        <script>
            let mediaRecorder = null;
            let audioChunks = [];
            let isRecording = false;

            // MediaRecorder emits webm/opus in every browser; ElevenLabs wants a real WAV.
            // Decode the recorded blob and re-encode as 16-bit mono PCM WAV.
            function audioBufferToWav(buffer) {
                const numChannels = 1;
                const sampleRate = buffer.sampleRate;
                const length = buffer.length * numChannels * 2;
                const ab = new ArrayBuffer(44 + length);
                const view = new DataView(ab);
                const writeStr = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
                writeStr(0, 'RIFF'); view.setUint32(4, 36 + length, true); writeStr(8, 'WAVE');
                writeStr(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
                view.setUint16(22, numChannels, true); view.setUint32(24, sampleRate, true);
                view.setUint32(28, sampleRate * numChannels * 2, true);
                view.setUint16(32, numChannels * 2, true); view.setUint16(34, 16, true);
                writeStr(36, 'data'); view.setUint32(40, length, true);
                const channel = buffer.getChannelData(0);
                let offset = 44;
                for (let i = 0; i < buffer.length; i++) {
                    const s = Math.max(-1, Math.min(1, channel[i]));
                    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                    offset += 2;
                }
                return new Blob([ab], { type: 'audio/wav' });
            }

            async function blobToWav(blob) {
                const arrayBuffer = await blob.arrayBuffer();
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
                const wav = audioBufferToWav(audioBuffer);
                ctx.close();
                return wav;
            }

            // SVG icon helpers for status badge
            const SVG_REFUSAL = `<svg class="w-3.5 h-3.5 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>`;
            const SVG_GROUNDED = `<svg class="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`;
            const SVG_ERROR = `<svg class="w-3.5 h-3.5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`;

            async function toggleRecording() {
                if (!isRecording) {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        mediaRecorder = new MediaRecorder(stream);
                        audioChunks = [];
                        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
                        mediaRecorder.onstop = sendRecordedAudio;
                        mediaRecorder.start();
                        isRecording = true;
                        document.getElementById('recordLabel').innerText = "Stop & Run";
                        document.getElementById('recordBtn').className = "flex items-center gap-2.5 bg-gradient-to-r from-amber-600 to-amber-700 hover:from-amber-500 hover:to-amber-600 text-white font-semibold px-6 py-3 rounded-xl text-sm transition-all duration-200 active:scale-95 shadow-lg shadow-amber-950/40";
                        document.getElementById('recordStatus').innerHTML = `<svg class="w-3.5 h-3.5 text-rose-500 animate-pulse" fill="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/></svg> Microphone active. Speaking...`;
                    } catch (err) {
                        alert("Microphone permission error: " + err.message);
                    }
                } else {
                    if (mediaRecorder && mediaRecorder.state !== "inactive") {
                        mediaRecorder.stop();
                    }
                    isRecording = false;
                    document.getElementById('recordLabel').innerText = "Start Live Mic";
                    document.getElementById('recordBtn').className = "flex items-center gap-2.5 bg-gradient-to-r from-rose-600 to-rose-700 hover:from-rose-500 hover:to-rose-600 text-white font-semibold px-6 py-3 rounded-xl text-sm transition-all duration-200 shadow-lg glow-rose active:scale-95";
                    document.getElementById('recordStatus').innerText = "Transcribing audio...";
                }
            }

            async function sendRecordedAudio() {
                document.getElementById('loader').classList.remove('hidden');
                document.getElementById('outputCard').classList.add('hidden');

                const rawBlob = new Blob(audioChunks, { type: 'audio/webm' });
                let audioBlob = rawBlob;
                try {
                    audioBlob = await blobToWav(rawBlob);
                } catch (err) {
                    console.warn('WAV conversion failed, sending original bytes:', err);
                }
                const formData = new FormData();
                formData.append('audio', audioBlob, 'mic_voice.wav');

                try {
                    const res = await fetch('/api/query', { method: 'POST', body: formData });
                    const data = await res.json();
                    renderResult(data);
                } catch (e) {
                    alert("Query failed: " + e.message);
                    document.getElementById('loader').classList.add('hidden');
                }
            }

            async function uploadAudioFile(input) {
                if (!input.files || !input.files[0]) return;
                const file = input.files[0];
                document.getElementById('recordStatus').innerText = `Uploaded: ${file.name}`;
                document.getElementById('loader').classList.remove('hidden');
                document.getElementById('outputCard').classList.add('hidden');

                const formData = new FormData();
                formData.append('audio', file, file.name);

                try {
                    const res = await fetch('/api/query', { method: 'POST', body: formData });
                    const data = await res.json();
                    renderResult(data);
                } catch (e) {
                    alert("Upload failed: " + e.message);
                    document.getElementById('loader').classList.add('hidden');
                }
            }

            async function sendTextQuery() {
                const text = document.getElementById('textInput').value.trim();
                if (!text) return;

                document.getElementById('loader').classList.remove('hidden');
                document.getElementById('outputCard').classList.add('hidden');

                const formData = new FormData();
                formData.append('text_override', text);

                try {
                    const res = await fetch('/api/query', { method: 'POST', body: formData });
                    const data = await res.json();
                    renderResult(data);
                } catch (e) {
                    alert("Query failed: " + e.message);
                    document.getElementById('loader').classList.add('hidden');
                }
            }

            function renderResult(data) {
                document.getElementById('loader').classList.add('hidden');
                document.getElementById('outputCard').classList.remove('hidden');
                document.getElementById('recordStatus').innerText = "";
                
                // Dynamic STT Engine label
                const engineLabel = document.getElementById('sttEngineLabel');
                if (data.stt_engine && data.stt_engine.includes("elevenlabs")) {
                    engineLabel.innerHTML = `<svg class="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg><span>Speech Transcript (ElevenLabs Scribe v2)</span>`;
                } else if (data.stt_engine && data.stt_engine.includes("sarvam")) {
                    engineLabel.innerHTML = `<svg class="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg><span>Speech Transcript (Sarvam AI saaras:v3)</span>`;
                } else {
                    engineLabel.innerHTML = `<svg class="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/></svg><span>Query Input (Text Input)</span>`;
                }

                if (data.error) {
                    document.getElementById('transcriptText').innerText = "Error: " + data.error;
                    document.getElementById('answerText').innerText = "Pipeline halted due to error.";
                    const badge = document.getElementById('statusBadge');
                    badge.className = 'px-3.5 py-1.5 text-xs rounded-full font-mono bg-red-950/80 text-red-400 border border-red-800/80 flex items-center gap-1.5';
                    badge.innerHTML = `${SVG_ERROR}<span>STT / PIPELINE ERROR</span>`;
                    return;
                }

                // Clean answer string (remove any lingering Passage artifacts)
                let cleanAns = (data.answer || "No response").replace(/Passage \\d+:/g, '').trim();

                document.getElementById('transcriptText').innerText = data.transcript || "N/A";
                document.getElementById('answerText').innerText = cleanAns;

                const badge = document.getElementById('statusBadge');
                if (data.refused) {
                    badge.className = 'px-3.5 py-1.5 text-xs rounded-full font-mono bg-rose-950/80 text-rose-300 border border-rose-800/80 flex items-center gap-1.5';
                    const refusalText = (data.refusal_reason && data.refusal_reason.includes("grounding check")) 
                        ? "GUARDRAIL REFUSAL (UNFAITHFUL / HALLUCINATION)" 
                        : "GUARDRAIL REFUSAL (OUT OF DOMAIN)";
                    badge.innerHTML = `${SVG_REFUSAL}<span>${refusalText}</span>`;
                } else {
                    badge.className = 'px-3.5 py-1.5 text-xs rounded-full font-mono bg-emerald-950/80 text-emerald-300 border border-emerald-800/80 flex items-center gap-1.5';
                    badge.innerHTML = `${SVG_GROUNDED}<span>GROUNDED ANSWER</span>`;
                }

                const timings = data.timings || {};
                let html = `
                    <div class="bg-slate-950/90 p-3 rounded-xl border border-slate-800/80"><div class="text-slate-400 text-[11px]">STT</div><div class="text-emerald-400 font-bold mt-1 text-sm">${timings.stt ? timings.stt.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950/90 p-3 rounded-xl border border-slate-800/80"><div class="text-slate-400 text-[11px]">Retrieval</div><div class="text-emerald-400 font-bold mt-1 text-sm">${timings.retrieval ? timings.retrieval.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950/90 p-3 rounded-xl border border-slate-800/80"><div class="text-slate-400 text-[11px]">Guardrail</div><div class="text-emerald-400 font-bold mt-1 text-sm">${timings.guardrail ? timings.guardrail.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950/90 p-3 rounded-xl border border-slate-800/80"><div class="text-slate-400 text-[11px]">LLM</div><div class="text-emerald-400 font-bold mt-1 text-sm">${timings.generation ? timings.generation.toFixed(1) : 0}ms</div></div>
                `;
                if (timings.hallucination_check !== undefined) {
                    html += `<div class="bg-slate-950/90 p-3 rounded-xl border border-slate-800/80"><div class="text-slate-400 text-[11px]">Hallucination Check</div><div class="text-emerald-400 font-bold mt-1 text-sm">${timings.hallucination_check.toFixed(1)}ms</div></div>`;
                }
                html += `<div class="bg-slate-950/90 p-3 rounded-xl border border-emerald-500/30 glow-emerald"><div class="text-slate-300 text-[11px]">Total</div><div class="text-emerald-300 font-bold mt-1 text-sm">${timings.total ? timings.total.toFixed(1) : 0}ms</div></div>`;
                document.getElementById('timingsBreakdown').innerHTML = html;
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
