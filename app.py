from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
from harness.orchestrator import VoiceRAGOrchestrator
from dataset.loader import ingest_corpus
import shutil
import tempfile
import os

orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    ingest_corpus()
    orchestrator = VoiceRAGOrchestrator()
    yield

app = FastAPI(title="HH Goa Voice-RAG Service", lifespan=lifespan)

@app.post("/api/query")
async def handle_query(
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
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen p-6 md:p-12">
        <div class="max-w-3xl mx-auto space-y-6">
            <header class="border-b border-slate-800 pb-4">
                <div class="flex items-center justify-between">
                    <span class="text-xs uppercase tracking-widest text-emerald-400 font-mono font-semibold">HH Goa 2026 · Task #2</span>
                    <span class="text-xs text-slate-500 font-mono">#RAGInGoa</span>
                </div>
                <h1 class="text-3xl font-black mt-1">🎙️ Voice-Enabled Grounded RAG</h1>
                <p class="text-slate-400 text-sm mt-1">Sarvam STT (saarika:v2.5) · Multi-Strategy Chunking · Latency Analytics · Refusal Guardrail</p>
            </header>

            <!-- Input Controls -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5">
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Voice Input (Microphone or Audio File)</label>
                    <div class="flex flex-wrap items-center gap-3">
                        <button id="recordBtn" onclick="toggleRecording()" class="flex items-center gap-2 bg-rose-600 hover:bg-rose-500 text-white font-semibold px-5 py-2.5 rounded-lg text-sm transition shadow-lg shadow-rose-950/50">
                            <span id="recordIcon" class="w-2.5 h-2.5 rounded-full bg-white animate-pulse"></span>
                            <span id="recordLabel">Start Live Mic</span>
                        </button>

                        <label class="cursor-pointer flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium px-4 py-2.5 rounded-lg text-sm transition">
                            <span>📁 Upload .WAV / .MP3</span>
                            <input id="audioFileInput" type="file" accept="audio/*" class="hidden" onchange="uploadAudioFile(this)">
                        </label>
                        <span id="recordStatus" class="text-xs text-slate-400 font-mono w-full md:w-auto"></span>
                    </div>
                </div>

                <div class="relative flex items-center">
                    <div class="flex-grow border-t border-slate-800"></div>
                    <span class="flex-shrink mx-4 text-xs uppercase text-slate-500 font-mono">or test with text</span>
                    <div class="flex-grow border-t border-slate-800"></div>
                </div>

                <div class="flex gap-3">
                    <input id="textInput" type="text" placeholder="e.g. What is the official state language of Goa?" class="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-emerald-500">
                    <button onclick="sendTextQuery()" class="bg-emerald-600 hover:bg-emerald-500 font-semibold px-6 py-2.5 rounded-lg text-sm transition shadow-lg shadow-emerald-950/50">Submit</button>
                </div>
            </div>

            <!-- Loading Spinner -->
            <div id="loader" class="hidden text-center py-6">
                <div class="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-emerald-500"></div>
                <div class="text-xs text-slate-400 mt-2 font-mono">Transcribing with Sarvam & Retrieving Grounded Context...</div>
            </div>

            <!-- Results Card -->
            <div id="outputCard" class="hidden bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
                <div class="flex justify-between items-center border-b border-slate-800 pb-3">
                    <h3 class="text-base font-bold text-white">Pipeline Execution Result</h3>
                    <span id="statusBadge" class="px-3 py-1 text-xs rounded-full font-mono font-semibold"></span>
                </div>
                
                <div>
                    <div class="text-xs text-slate-400 font-mono uppercase">Speech Transcript (Sarvam STT)</div>
                    <div id="transcriptText" class="text-slate-100 font-medium mt-1 bg-slate-950 p-3 rounded border border-slate-800"></div>
                </div>

                <div>
                    <div class="text-xs text-slate-400 font-mono uppercase">Grounded Answer</div>
                    <div id="answerText" class="text-emerald-300 font-medium mt-1 bg-slate-950 p-3 rounded border border-slate-800"></div>
                </div>

                <div>
                    <div class="text-xs text-slate-400 font-mono uppercase mb-2">Empirical Latency Breakdown</div>
                    <div id="timingsBreakdown" class="grid grid-cols-5 gap-2 font-mono text-xs text-center"></div>
                </div>
            </div>
        </div>

        <script>
            let mediaRecorder = null;
            let audioChunks = [];
            let isRecording = false;

            async function toggleRecording() {
                if (!isRecording) {
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        alert("⚠️ Microphone access requires opening this page at http://localhost:8000 or http://127.0.0.1:8000. You can also use the 'Upload .WAV' button.");
                        return;
                    }
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        mediaRecorder = new MediaRecorder(stream);
                        audioChunks = [];
                        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
                        mediaRecorder.onstop = sendRecordedAudio;
                        mediaRecorder.start();
                        isRecording = true;
                        document.getElementById('recordLabel').innerText = "Stop & Run";
                        document.getElementById('recordBtn').className = "flex items-center gap-2 bg-amber-600 hover:bg-amber-500 text-white font-semibold px-5 py-2.5 rounded-lg text-sm transition";
                        document.getElementById('recordStatus').innerText = "🔴 Recording microphone...";
                    } catch (err) {
                        alert("Microphone permission error: " + err.message);
                    }
                } else {
                    if (mediaRecorder && mediaRecorder.state !== "inactive") {
                        mediaRecorder.stop();
                    }
                    isRecording = false;
                    document.getElementById('recordLabel').innerText = "Start Live Mic";
                    document.getElementById('recordBtn').className = "flex items-center gap-2 bg-rose-600 hover:bg-rose-500 text-white font-semibold px-5 py-2.5 rounded-lg text-sm transition shadow-lg shadow-rose-950/50";
                    document.getElementById('recordStatus').innerText = "Processing audio...";
                }
            }

            async function sendRecordedAudio() {
                document.getElementById('loader').classList.remove('hidden');
                document.getElementById('outputCard').classList.add('hidden');

                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const formData = new FormData();
                formData.append('audio', audioBlob, 'mic_input.wav');

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
                    alert("Upload processing failed: " + e.message);
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
                
                if (data.error) {
                    document.getElementById('transcriptText').innerText = "Error: " + data.error;
                    document.getElementById('answerText').innerText = "Pipeline stopped due to error.";
                    const badge = document.getElementById('statusBadge');
                    badge.className = 'px-3 py-1 text-xs rounded-full font-mono bg-red-950 text-red-400 border border-red-800';
                    badge.innerText = '❌ ERROR';
                    return;
                }

                document.getElementById('transcriptText').innerText = data.transcript || "N/A";
                document.getElementById('answerText').innerText = data.answer || "No response";

                const badge = document.getElementById('statusBadge');
                if (data.refused) {
                    badge.className = 'px-3 py-1 text-xs rounded-full font-mono bg-rose-950 text-rose-400 border border-rose-800';
                    badge.innerText = '🛡️ GUARDRAIL REFUSAL (OUT OF DOMAIN)';
                } else {
                    badge.className = 'px-3 py-1 text-xs rounded-full font-mono bg-emerald-950 text-emerald-400 border border-emerald-800';
                    badge.innerText = '✅ GROUNDED ANSWER';
                }

                const timings = data.timings || {};
                document.getElementById('timingsBreakdown').innerHTML = `
                    <div class="bg-slate-950 p-2.5 rounded border border-slate-800"><div class="text-slate-400">STT</div><div class="text-emerald-400 font-bold mt-1">${timings.stt ? timings.stt.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950 p-2.5 rounded border border-slate-800"><div class="text-slate-400">Retrieval</div><div class="text-emerald-400 font-bold mt-1">${timings.retrieval ? timings.retrieval.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950 p-2.5 rounded border border-slate-800"><div class="text-slate-400">Guardrail</div><div class="text-emerald-400 font-bold mt-1">${timings.guardrail ? timings.guardrail.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950 p-2.5 rounded border border-slate-800"><div class="text-slate-400">LLM</div><div class="text-emerald-400 font-bold mt-1">${timings.generation ? timings.generation.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950 p-2.5 rounded border border-emerald-900"><div class="text-slate-300">Total</div><div class="text-emerald-300 font-bold mt-1">${timings.total ? timings.total.toFixed(1) : 0}ms</div></div>
                `;
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
