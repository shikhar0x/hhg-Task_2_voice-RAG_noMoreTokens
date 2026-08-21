from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
from harness.orchestrator import VoiceRAGOrchestrator
from retrieval.vector_store import get_vector_store
from config.logger import logger
from pydantic import BaseModel, Field
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
    text_override: str = Form(None),
    generate: bool = Form(True),
):
    temp_path = None
    if audio and audio.filename:
        suffix = os.path.splitext(audio.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            temp_path = tmp.name

    try:
        result = orchestrator.process(
            audio_path=temp_path,
            text_override=text_override,
            generate=generate,
        )
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

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    generate: bool = True


@app.post("/ask")
def ask(req: AskRequest):
    """Text question → answer. `generate=false` is the official <200ms fast path."""
    result = orchestrator.process(text_override=req.question, generate=req.generate)
    return JSONResponse(content=result)


@app.get("/health")
def health():
    col = get_vector_store()
    return {
        "status": "ok",
        "passages": col.count(),
        "budget_ms": 200,
        "fast_path": "retrieve + guardrail + extractive",
    }



def _run_live_benchmark(n: int = 80):
    from app.benchmark import run_benchmark
    n = max(20, min(int(n), 200))
    return run_benchmark(n=n, verbose=False, orch=orchestrator)


@app.api_route("/api/benchmark", methods=["GET", "POST"])
def api_benchmark(n: int = 80):
    return JSONResponse(content=_run_live_benchmark(n))


@app.get("/benchmark")
def benchmark(n: int = 80):
    """Alias so `GET /benchmark?n=80` matches the README command."""
    return JSONResponse(content=_run_live_benchmark(n))

def _app_benchmark(*args, **kwargs):
    from app.benchmark import run_benchmark
    return run_benchmark(*args, **kwargs)

app.benchmark = _app_benchmark

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HH Goa 2026 | Voice-Enabled Grounded RAG</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {
                font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            .font-mono {
                font-family: 'JetBrains Mono', monospace;
            }
            .font-heading {
                font-family: 'Inter', sans-serif;
            }
            
            /* Custom glassmorphism & gradients */
            .glass-panel {
                background: rgba(13, 18, 30, 0.75);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.6);
            }
            .glass-panel-interactive {
                background: rgba(13, 18, 30, 0.75);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.6);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            .glass-panel-interactive:hover {
                border-color: rgba(255, 255, 255, 0.15);
                box-shadow: 0 30px 70px -15px rgba(0, 0, 0, 0.7), 0 0 30px rgba(99, 102, 241, 0.1);
            }
            .glass-input {
                background: rgba(4, 9, 20, 0.75);
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.25s ease;
            }
            .glass-input:focus {
                border-color: rgba(16, 185, 129, 0.6);
                box-shadow: 0 0 20px rgba(16, 185, 129, 0.2), inset 0 0 10px rgba(16, 185, 129, 0.05);
            }
            .glow-emerald {
                box-shadow: 0 0 30px -5px rgba(16, 185, 129, 0.35);
            }
            .glow-rose {
                box-shadow: 0 0 30px -5px rgba(225, 29, 72, 0.4);
            }
            .glow-indigo {
                box-shadow: 0 0 30px -5px rgba(99, 102, 241, 0.35);
            }
            .bg-mesh {
                background-color: #060913;
                background-image: 
                    radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.12) 0px, transparent 50%),
                    radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                    radial-gradient(at 50% 100%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
                    linear-gradient(to right, rgba(255, 255, 255, 0.015) 1px, transparent 1px),
                    linear-gradient(to bottom, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
                background-size: 100% 100%, 100% 100%, 100% 100%, 36px 36px, 36px 36px;
            }
            
            /* Custom styled scrollbars */
            ::-webkit-scrollbar {
                width: 6px;
                height: 6px;
            }
            ::-webkit-scrollbar-track {
                background: #040814;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb {
                background: #1e293b;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #10b981;
            }

            @keyframes pulse-ring {
                0% { transform: scale(0.95); opacity: 0.8; }
                50% { transform: scale(1.05); opacity: 0.4; }
                100% { transform: scale(0.95); opacity: 0.8; }
            }
            .animate-pulse-ring {
                animation: pulse-ring 2s infinite cubic-bezier(0.4, 0, 0.6, 1);
            }
        </style>
    </head>
    <body class="bg-mesh text-slate-100 min-h-screen p-4 sm:p-6 md:p-12 relative overflow-x-hidden">
        <!-- Ambient background glow elements -->
        <div class="fixed -top-24 left-1/4 w-[500px] h-[500px] bg-emerald-500/10 rounded-full blur-[120px] pointer-events-none"></div>
        <div class="fixed top-1/3 -right-24 w-[500px] h-[500px] bg-indigo-500/12 rounded-full blur-[120px] pointer-events-none"></div>
        <div class="fixed -bottom-24 left-1/3 w-[500px] h-[500px] bg-violet-600/10 rounded-full blur-[120px] pointer-events-none"></div>

        <div class="max-w-[1500px] w-full mx-auto relative z-10">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-7 items-start">
                
                <!-- Left Column: Header + Latency Benchmark -->
                <div class="space-y-7">
                    <!-- Upper Frame Header -->
                    <header class="glass-panel rounded-2xl p-7 md:p-8 relative overflow-hidden border border-slate-800/80 space-y-5 shadow-2xl">
                        <div class="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-500 via-indigo-500 to-violet-500"></div>
                        <div class="flex flex-wrap items-center justify-between gap-3">
                            <span class="text-xs uppercase tracking-widest text-emerald-400 font-mono font-bold bg-emerald-950/80 border border-emerald-500/30 px-3.5 py-1.5 rounded-lg flex items-center gap-2 shadow-sm">
                                <span class="relative flex h-2 w-2">
                                    <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                    <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-400"></span>
                                </span>
                                HH Goa 2026 · Task #2
                            </span>
                            <div class="flex items-center gap-2.5">
                                <span class="text-xs text-indigo-300 font-mono font-semibold bg-indigo-950/80 border border-indigo-500/30 px-3 py-1.5 rounded-lg flex items-center gap-1.5 shadow-sm">
                                    <svg class="w-3.5 h-3.5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"/></svg>
                                    Team: No More Tokens
                                </span>
                                <span class="text-xs text-slate-400 font-mono hidden sm:flex items-center gap-1.5 bg-slate-900/70 border border-slate-800 px-3 py-1.5 rounded-lg">
                                    <svg class="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                                    RAGInGoa
                                </span>
                            </div>
                        </div>
                        
                        <h1 class="text-2xl sm:text-3xl font-extrabold mt-3 tracking-tight font-heading flex items-center gap-3.5">
                            <span class="p-2.5 rounded-xl bg-gradient-to-br from-emerald-500/20 to-indigo-500/20 border border-emerald-500/30 text-emerald-400 shrink-0 shadow-lg shadow-emerald-950/50">
                                <svg class="w-6 h-6 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/>
                                </svg>
                            </span>
                            <span class="bg-gradient-to-r from-white via-slate-100 to-slate-300 bg-clip-text text-transparent">
                                Voice-Enabled Grounded RAG
                            </span>
                        </h1>
                        
                        <p class="text-slate-400 text-xs mt-3 leading-relaxed flex flex-wrap items-center gap-x-2 gap-y-1.5 pt-3 border-t border-slate-800/60">
                            <span>Built by <strong class="text-slate-200">Team No More Tokens</strong></span>
                            <span class="text-slate-600">·</span>
                            <span class="px-2.5 py-1 rounded-lg bg-slate-900/80 border border-slate-800 text-slate-300 text-[11px] font-mono">ElevenLabs & Sarvam STT</span>
                            <span class="text-slate-600">·</span>
                            <span class="px-2.5 py-1 rounded-lg bg-slate-900/80 border border-slate-800 text-slate-300 text-[11px] font-mono">Multi-Strategy Chunking</span>
                            <span class="text-slate-600">·</span>
                            <span class="px-2.5 py-1 rounded-lg bg-slate-900/80 border border-slate-800 text-slate-300 text-[11px] font-mono">Latency Analytics</span>
                            <span class="text-slate-600">·</span>
                            <span class="px-2.5 py-1 rounded-lg bg-slate-900/80 border border-slate-800 text-slate-300 text-[11px] font-mono">Refusal Guardrail</span>
                        </p>
                    </header>

                    <!-- Latency Benchmark Card Component (UNIFORM GLASS STYLING) -->
                    <div class="glass-panel rounded-2xl p-7 md:p-8 border border-slate-800/80 space-y-5 shadow-2xl">
                        <div class="flex items-center justify-between pb-1">
                            <h2 class="text-base font-bold text-white tracking-tight flex items-center gap-2.5">
                                <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                                <span>Latency benchmark</span>
                            </h2>
                            <button id="webBenchmarkBtn" onclick="runWebBenchmark()" class="bg-[#4f46e5] hover:bg-[#4338ca] text-white font-semibold text-xs px-4 py-2 rounded-lg transition-all duration-150 shadow active:scale-95 flex items-center gap-1.5 cursor-pointer">
                                <span id="webBenchmarkSpinner" class="hidden w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                                <span>Run benchmark</span>
                            </button>
                        </div>

                        <div class="overflow-x-auto">
                            <table class="w-full text-sm">
                                <thead>
                                    <tr class="border-b border-zinc-800/80 text-slate-400 text-xs font-medium uppercase">
                                        <th class="text-left py-2.5 font-normal"></th>
                                        <th class="text-right py-2.5 px-4 font-semibold tracking-wider">AVG</th>
                                        <th class="text-right py-2.5 px-4 font-semibold tracking-wider">P50</th>
                                        <th class="text-right py-2.5 px-4 font-semibold tracking-wider">P95</th>
                                        <th class="text-right py-2.5 px-4 font-semibold tracking-wider">P99</th>
                                    </tr>
                                </thead>
                                <tbody id="benchmarkTableBody" class="font-mono text-white text-xs divide-y divide-zinc-800/40">
                                    <tr>
                                        <td class="text-left py-3.5 text-slate-300 font-sans">total (ms)</td>
                                        <td class="text-right py-3.5 px-4 text-slate-100 font-medium" id="bm-avg">--</td>
                                        <td class="text-right py-3.5 px-4 text-slate-100 font-medium" id="bm-p50">--</td>
                                        <td class="text-right py-3.5 px-4 text-slate-100 font-medium" id="bm-p95">--</td>
                                        <td class="text-right py-3.5 px-4 text-slate-100 font-medium" id="bm-p99">--</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>

                        <div class="pt-2 border-t border-slate-800/60">
                            <span id="benchmarkBadgePill" class="inline-flex items-center px-5 py-2.5 rounded-full text-xl font-bold tracking-wider bg-slate-900 text-slate-200 border border-slate-700">
                                this host p50 ~194ms · brief 200ms · laptop p50 1.5ms
                            </span>
                        </div>
                    </div>
                </div>

                <!-- Right Column: Voice Input & Pipeline Execution Result -->
                <div class="space-y-7">
                    <!-- Input Controls Card -->
                    <div class="glass-panel rounded-2xl p-7 md:p-8 space-y-6 border border-slate-800/80 shadow-2xl">
                        <div>
                            <label class="block text-xs font-bold uppercase tracking-wider text-slate-300 mb-3.5 flex items-center gap-2">
                                <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg>
                                Voice Input (Live Microphone or Audio Upload)
                            </label>
                            <div class="flex flex-wrap items-center gap-3">
                                <button id="recordBtn" onclick="toggleRecording()" class="flex items-center gap-2.5 bg-gradient-to-r from-rose-600 via-rose-600 to-pink-600 hover:from-rose-500 hover:to-pink-500 text-white font-semibold px-6 py-3 rounded-xl text-sm transition-all duration-200 shadow-lg glow-rose active:scale-95 cursor-pointer">
                                    <span id="recordIcon" class="w-2.5 h-2.5 rounded-full bg-white animate-pulse"></span>
                                    <span id="recordLabel">Start Live Mic</span>
                                </button>

                                <label class="cursor-pointer flex items-center gap-2 bg-slate-800/90 hover:bg-slate-700/90 text-slate-200 border border-slate-700/90 hover:border-slate-500 font-medium px-5 py-3 rounded-xl text-sm transition-all duration-200 active:scale-95 shadow-md">
                                    <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg>
                                    <span>Upload .WAV</span>
                                    <input id="audioFileInput" type="file" accept="audio/*" class="hidden" onchange="uploadAudioFile(this)">
                                </label>
                                <span id="recordStatus" class="text-xs text-slate-400 font-mono w-full md:w-auto flex items-center gap-1.5"></span>
                            </div>
                        </div>

                        <div class="relative flex items-center py-1">
                            <div class="flex-grow border-t border-slate-800/80"></div>
                            <span class="flex-shrink mx-4 text-[11px] uppercase tracking-widest text-slate-400 font-mono font-semibold bg-slate-900/80 px-3 py-1 rounded-full border border-slate-800">or test with query text</span>
                            <div class="flex-grow border-t border-slate-800/80"></div>
                        </div>

                        <div class="flex flex-col sm:flex-row gap-3">
                            <input id="textInput" type="text" placeholder="e.g. What is the definition of honesty and integrity?" class="flex-1 glass-input rounded-xl px-4 py-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none">
                            <button onclick="sendTextQuery()" class="bg-gradient-to-r from-emerald-600 via-teal-600 to-emerald-500 hover:from-emerald-500 hover:to-teal-400 text-white font-bold px-7 py-3.5 rounded-xl text-sm transition-all duration-200 shadow-lg glow-emerald active:scale-95 flex items-center justify-center gap-2 cursor-pointer">
                                <span>Submit</span>
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
                            </button>
                        </div>
                    </div>

                    <!-- Loading Spinner -->
                    <div id="loader" class="hidden text-center py-10 glass-panel rounded-2xl border border-slate-800/80 shadow-2xl">
                        <div class="relative inline-flex items-center justify-center">
                            <div class="w-12 h-12 border-4 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin"></div>
                            <svg class="w-5 h-5 text-emerald-400 absolute animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                        </div>
                        <div class="text-xs text-slate-300 mt-4 font-mono font-medium">Transcribing Speech, Retrieving Context & Formulating Answer...</div>
                    </div>

                    <!-- Results Card -->
                    <div id="outputCard" class="hidden glass-panel rounded-2xl p-7 md:p-8 space-y-6 border border-slate-800/80 shadow-2xl">
                        <div class="flex justify-between items-center border-b border-slate-800/80 pb-4">
                            <div class="flex items-center gap-3 flex-wrap">
                            <h3 class="text-base font-bold text-white flex items-center gap-2.5">
                                <div class="p-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                                </div>
                                Pipeline Execution Result
                            </h3>
                            <span id="statusBadge" class="px-3.5 py-1.5 text-xs rounded-full font-mono font-semibold flex items-center gap-1.5"></span>
                            </div>
                            <div id="tierTrack" class="flex items-center gap-2 text-[10px] font-mono uppercase tracking-wider text-slate-500">
                                <span id="tierExtractive" class="px-2 py-1 rounded-md border border-slate-800">01 extractive <em class="not-italic text-slate-400" id="t1ms">—</em></span>
                                <span class="text-slate-700">→</span>
                                <span id="tierGenerated" class="px-2 py-1 rounded-md border border-slate-800">02 generated <em class="not-italic text-slate-400" id="t2ms">—</em></span>
                            </div>
                        </div>
                        
                        <div class="space-y-2">
                            <div id="sttEngineLabel" class="text-xs text-slate-400 font-mono uppercase tracking-wider flex items-center gap-2 px-1">
                                <svg class="w-4 h-4 text-slate-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg>
                                <span>Speech Transcript</span>
                            </div>
                            <div id="transcriptText" class="text-slate-100 font-medium bg-slate-950/90 px-5 py-4 rounded-xl border border-slate-800/80 text-sm leading-relaxed shadow-inner"></div>
                        </div>

                        <!-- Grounded Answer Box -->
                        <div class="space-y-2 pt-1">
                            <div class="text-xs text-slate-400 font-mono uppercase tracking-wider flex items-center gap-2 px-1">
                                <svg class="w-4 h-4 text-emerald-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>
                                <span id="answerTierLabel">Answer</span>
                            </div>
                            <div id="answerText" class="text-emerald-300 font-normal leading-relaxed bg-slate-950/90 px-5 py-4 rounded-xl border border-slate-800/80 min-h-[140px] max-h-[300px] overflow-y-auto whitespace-pre-wrap text-sm shadow-inner"></div>
                        </div>
                        <div id="sourcesPanel" class="hidden space-y-2 pt-1"></div>

                        <div class="space-y-2 pt-3 border-t border-slate-800/60">
                            <div class="text-xs text-slate-400 font-mono uppercase tracking-wider flex items-center gap-2 px-1">
                                <svg class="w-4 h-4 text-slate-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                                <span>Empirical Latency Breakdown</span>
                            </div>
                            <div id="timingsBreakdown" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 font-mono text-xs text-center"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function runWebBenchmark() {
                const btn = document.getElementById('webBenchmarkBtn');
                const spinner = document.getElementById('webBenchmarkSpinner');
                btn.disabled = true;
                if (spinner) spinner.classList.remove('hidden');

                try {
                    const res = await fetch('/api/benchmark', { method: 'POST' });
                    const data = await res.json();
                    
                    const total = data.metrics['total (ms)'];
                    document.getElementById('bm-avg').innerText = total.avg.toFixed(2);
                    document.getElementById('bm-p50').innerText = total.p50.toFixed(2);
                    const p95Val = total.p95;
                    document.getElementById('bm-p95').innerText = (Math.round(p95Val * 10) / 10 === p95Val) ? p95Val.toFixed(1) : p95Val.toFixed(2);
                    document.getElementById('bm-p99').innerText = total.p99.toFixed(2);

                                        const badge = document.getElementById('benchmarkBadgePill');
                    badge.innerText = 'this host p50 ' + total.p50.toFixed(1) + 'ms · brief 200ms · laptop p50 1.5ms';
                    if (data.status === 'PASS') {
                        badge.className = 'inline-flex items-center px-5 py-2.5 rounded-full text-xl font-bold tracking-wider bg-emerald-950/90 text-emerald-400 border border-emerald-800/70';
                    } else {
                        badge.className = 'inline-flex items-center px-5 py-2.5 rounded-full text-xl font-bold tracking-wider bg-rose-950/90 text-rose-400 border border-rose-800/70';
                    }
                
                } catch (e) {
                    console.error("Benchmark failed:", e);
                } finally {
                    btn.disabled = false;
                    if (spinner) spinner.classList.add('hidden');
                }
            }

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
                        document.getElementById('recordBtn').className = "flex items-center gap-2.5 bg-gradient-to-r from-amber-600 to-amber-700 hover:from-amber-500 hover:to-amber-600 text-white font-semibold px-6 py-3 rounded-xl text-sm transition-all duration-200 active:scale-95 shadow-lg shadow-amber-950/40 cursor-pointer";
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
                    document.getElementById('recordBtn').className = "flex items-center gap-2.5 bg-gradient-to-r from-rose-600 via-rose-600 to-pink-600 hover:from-rose-500 hover:to-pink-500 text-white font-semibold px-6 py-3 rounded-xl text-sm transition-all duration-200 shadow-lg glow-rose active:scale-95 cursor-pointer";
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
                formData.append('generate', 'false');

                try {
                    const res = await fetch('/api/query', { method: 'POST', body: formData });
                    const data = await res.json();
                    renderResult(data, 'extractive');
                    if (data.transcript && !data.refused && !data.error) {
                        await polishQuestion(data.transcript);
                    }
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
                formData.append('generate', 'false');

                try {
                    const res = await fetch('/api/query', { method: 'POST', body: formData });
                    const data = await res.json();
                    renderResult(data, 'extractive');
                    if (data.transcript && !data.refused && !data.error) {
                        await polishQuestion(data.transcript);
                    }
                } catch (e) {
                    alert("Upload failed: " + e.message);
                    document.getElementById('loader').classList.add('hidden');
                }
            }

            async function polishQuestion(question) {
                try {
                    const res = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question, generate: true }),
                    });
                    const data = await res.json();
                    renderResult(data, 'generated');
                } catch (e) {
                    const t2 = document.getElementById('t2ms');
                    if (t2) t2.textContent = 'unavailable';
                    const hint = document.getElementById('recordStatus');
                    if (hint) hint.innerText = 'generation unavailable — extractive answer stands';
                }
            }

            async function sendTextQuery() {
                const text = document.getElementById('textInput').value.trim();
                if (!text) return;

                document.getElementById('loader').classList.remove('hidden');
                document.getElementById('outputCard').classList.add('hidden');

                try {
                    const res = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: text, generate: false }),
                    });
                    const data = await res.json();
                    renderResult(data, 'extractive');
                    if (!data.refused && !data.error) {
                        await polishQuestion(text);
                    }
                } catch (e) {
                    alert("Query failed: " + e.message);
                    document.getElementById('loader').classList.add('hidden');
                }
            }

            function renderResult(data, tier) {
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

                const sourcesPanel = document.getElementById('sourcesPanel');
                const srcs = data.sources && data.sources.length
                    ? data.sources
                    : (data.retrieved_docs || []).map((t, i) => ({ text: t, score: (data.similarities || [])[i] }));
                if (sourcesPanel) {
                    if (!srcs.length) {
                        sourcesPanel.classList.add('hidden');
                        sourcesPanel.innerHTML = '';
                    } else {
                        sourcesPanel.classList.remove('hidden');
                        const support = (data.extractive_support != null) ? Number(data.extractive_support).toFixed(3) : '—';
                        const cov = (data.extractive_coverage != null) ? Number(data.extractive_coverage).toFixed(2) : '—';
                        sourcesPanel.innerHTML =
                            `<div class="text-xs text-slate-400 font-mono uppercase tracking-wider flex items-center gap-2 px-1">Retrieved passages <span class="text-slate-500 normal-case">support ${support} · coverage ${cov}</span></div>` +
                            srcs.slice(0, 3).map((s, i) => {
                                const text = (s.text || s || '').toString();
                                const score = (s.score != null) ? Number(s.score).toFixed(3) : '—';
                                return `<div class="text-slate-300 text-xs leading-relaxed bg-slate-950/90 px-4 py-3 rounded-xl border border-slate-800/80"><span class="text-slate-500 font-mono">[${i + 1}] sim ${score}</span><br>${text.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])).slice(0, 420)}</div>`;
                            }).join('');
                    }
                }

                const src = data.answer_source || (data.refused ? 'refusal' : 'generated');
                const t1 = document.getElementById('tierExtractive');
                const t2 = document.getElementById('tierGenerated');
                const t1ms = document.getElementById('t1ms');
                const t2ms = document.getElementById('t2ms');
                const tierLabel = document.getElementById('answerTierLabel');
                if (t1ms) t1ms.textContent = (data.fast_path_ms != null) ? (data.fast_path_ms.toFixed(1) + 'ms') : '—';
                if (t1) t1.className = 'px-2 py-1 rounded-md border border-emerald-800/70 text-emerald-300';
                if (tier === 'extractive' && src !== 'refusal') {
                    if (t2ms) t2ms.textContent = '···';
                    if (t2) t2.className = 'px-2 py-1 rounded-md border border-amber-800/70 text-amber-300';
                    if (tierLabel) tierLabel.innerText = 'Extractive answer (no LLM)';
                } else if (src === 'generated') {
                    const genMs = (data.timings && data.timings.generation) ? data.timings.generation.toFixed(1) + 'ms' : '—';
                    if (t2ms) t2ms.textContent = genMs;
                    if (t2) t2.className = 'px-2 py-1 rounded-md border border-emerald-800/70 text-emerald-300';
                    if (tierLabel) tierLabel.innerText = 'Polished answer (Groq) · extractive stood at ' + (data.fast_path_ms ? data.fast_path_ms.toFixed(1) + 'ms' : '—');
                } else if (src === 'extractive') {
                    if (t2ms) t2ms.textContent = 'kept extractive';
                    if (t2) t2.className = 'px-2 py-1 rounded-md border border-slate-700 text-slate-400';
                    if (tierLabel) tierLabel.innerText = 'Extractive answer (generation unused or rejected)';
                } else {
                    if (t2ms) t2ms.textContent = '—';
                    if (tierLabel) tierLabel.innerText = 'Answer';
                }

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
                const fast = (data.fast_path_ms != null) ? data.fast_path_ms : (timings.fast_path || 0);
                let html = `
                    <div class="bg-slate-950/90 p-3 rounded-xl border border-slate-800/80"><div class="text-slate-400 text-[11px]">STT (outside)</div><div class="text-emerald-400 font-bold mt-1 text-sm">${timings.stt ? timings.stt.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950/90 p-3 rounded-xl border border-slate-800/80"><div class="text-slate-400 text-[11px]">Retrieval</div><div class="text-emerald-400 font-bold mt-1 text-sm">${timings.retrieval ? timings.retrieval.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950/90 p-3 rounded-xl border border-slate-800/80"><div class="text-slate-400 text-[11px]">Guardrail</div><div class="text-emerald-400 font-bold mt-1 text-sm">${timings.guardrail ? timings.guardrail.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950/90 p-3 rounded-xl border border-slate-800/80"><div class="text-slate-400 text-[11px]">Extract</div><div class="text-emerald-400 font-bold mt-1 text-sm">${timings.extract ? timings.extract.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950/90 p-3 rounded-xl border border-emerald-500/30 glow-emerald"><div class="text-slate-300 text-[11px]">Fast path</div><div class="text-emerald-300 font-bold mt-1 text-sm">${fast ? fast.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950/90 p-3 rounded-xl border border-slate-800/80"><div class="text-slate-400 text-[11px]">LLM (outside)</div><div class="text-emerald-400 font-bold mt-1 text-sm">${timings.generation ? timings.generation.toFixed(1) : 0}ms</div></div>
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
