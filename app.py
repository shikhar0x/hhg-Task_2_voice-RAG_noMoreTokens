from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from harness.orchestrator import VoiceRAGOrchestrator
from dataset.loader import ingest_corpus
import shutil
import tempfile
import os

app = FastAPI(title="HH Goa Voice-RAG Service", version="1.0.0")
orchestrator = VoiceRAGOrchestrator()

# Initialize corpus on startup
@app.on_event("startup")
def startup_event():
    ingest_corpus()

@app.post("/api/query")
async def handle_query(
    audio: UploadFile = File(None),
    text_override: str = Form(None)
):
    temp_path = None
    if audio:
        suffix = os.path.splitext(audio.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            temp_path = tmp.name

    try:
        result = orchestrator.process(audio_path=temp_path, text_override=text_override)
        return JSONResponse(content=result)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

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
        <title>HH Goa 2026 | Voice-Enabled RAG</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen p-8">
        <div class="max-w-4xl mx-auto space-y-6">
            <header class="border-b border-slate-800 pb-4">
                <span class="text-xs uppercase tracking-widest text-emerald-400 font-mono">HH Goa 2026 · Task #2</span>
                <h1 class="text-3xl font-black mt-1">🎙️ Voice-Enabled Grounded RAG</h1>
                <p class="text-slate-400 text-sm mt-1">Real-time STT · Multi-Strategy Chunking · Latency Analytics · Refusal Guardrails</p>
            </header>

            <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
                <label class="block text-sm font-medium text-slate-300">Test Query (Voice or Text)</label>
                <div class="flex gap-3">
                    <input id="textInput" type="text" placeholder="e.g., What is the capital of Goa and its language?" class="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-emerald-500">
                    <button onclick="sendQuery()" class="bg-emerald-600 hover:bg-emerald-500 font-semibold px-6 py-2 rounded-lg text-sm transition">Run Pipeline</button>
                </div>
            </div>

            <div id="outputCard" class="hidden bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
                <div class="flex justify-between items-center">
                    <h3 class="text-lg font-bold text-white">Pipeline Result</h3>
                    <span id="statusBadge" class="px-3 py-1 text-xs rounded-full font-mono"></span>
                </div>
                <div>
                    <div class="text-xs text-slate-400 font-mono uppercase">Transcript</div>
                    <div id="transcriptText" class="text-slate-200 mt-1"></div>
                </div>
                <div>
                    <div class="text-xs text-slate-400 font-mono uppercase">Answer</div>
                    <div id="answerText" class="text-emerald-300 font-medium mt-1"></div>
                </div>
                <div>
                    <div class="text-xs text-slate-400 font-mono uppercase">Latency Breakdown</div>
                    <div id="timingsBreakdown" class="grid grid-cols-5 gap-2 mt-2 font-mono text-xs text-center"></div>
                </div>
            </div>
        </div>

        <script>
            async function sendQuery() {
                const text = document.getElementById('textInput').value;
                const formData = new FormData();
                formData.append('text_override', text);

                const res = await fetch('/api/query', { method: 'POST', body: formData });
                const data = await res.json();

                document.getElementById('outputCard').classList.remove('hidden');
                document.getElementById('transcriptText').innerText = data.transcript || text;
                document.getElementById('answerText').innerText = data.answer;

                const badge = document.getElementById('statusBadge');
                if (data.refused) {
                    badge.className = 'px-3 py-1 text-xs rounded-full font-mono bg-red-950 text-red-400 border border-red-800';
                    badge.innerText = 'GUARDRAIL REFUSED';
                } else {
                    badge.className = 'px-3 py-1 text-xs rounded-full font-mono bg-emerald-950 text-emerald-400 border border-emerald-800';
                    badge.innerText = 'GROUNDED ANSWER';
                }

                const timings = data.timings || {};
                document.getElementById('timingsBreakdown').innerHTML = `
                    <div class="bg-slate-950 p-2 rounded border border-slate-800"><div>STT</div><div class="text-emerald-400">${timings.stt ? timings.stt.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950 p-2 rounded border border-slate-800"><div>Retrieval</div><div class="text-emerald-400">${timings.retrieval ? timings.retrieval.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950 p-2 rounded border border-slate-800"><div>Guardrail</div><div class="text-emerald-400">${timings.guardrail ? timings.guardrail.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950 p-2 rounded border border-slate-800"><div>LLM</div><div class="text-emerald-400">${timings.generation ? timings.generation.toFixed(1) : 0}ms</div></div>
                    <div class="bg-slate-950 p-2 rounded border border-emerald-900 font-bold"><div>Total</div><div class="text-emerald-300">${timings.total ? timings.total.toFixed(1) : 0}ms</div></div>
                `;
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
