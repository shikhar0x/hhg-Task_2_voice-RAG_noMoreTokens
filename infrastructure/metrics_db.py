import sqlite3
import numpy as np
from datetime import datetime
from config.settings import settings
from config.logger import logger

class MetricsDB:
    """Latency metrics repository computing empirical P50, P70, P100 percentiles."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.metrics_db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS latency_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    transcript TEXT,
                    stt_ms REAL,
                    retrieval_ms REAL,
                    guardrail_ms REAL,
                    generation_ms REAL,
                    total_ms REAL,
                    refused INTEGER,
                    timestamp TEXT
                )
            """)
            conn.commit()

    def log(self, query_id: str, transcript: str, timings: dict[str, float], refused: bool = False):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO latency_logs (query_id, transcript, stt_ms, retrieval_ms, guardrail_ms, generation_ms, total_ms, refused, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_id,
                transcript,
                timings.get("stt", 0.0),
                timings.get("retrieval", 0.0),
                timings.get("guardrail", 0.0),
                timings.get("generation", 0.0),
                timings.get("total", 0.0),
                1 if refused else 0,
                datetime.now().isoformat()
            ))
            conn.commit()

    def compute_percentiles(self) -> dict[str, dict[str, str]]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT stt_ms, retrieval_ms, guardrail_ms, generation_ms, total_ms FROM latency_logs")
            rows = cur.fetchall()

        if not rows:
            return {}

        data = np.array(rows)
        stages = ["STT", "Vector Retrieval", "Guardrail Gate", "LLM Generation", "Total End-to-End"]
        metrics = {}

        for i, stage in enumerate(stages):
            vals = data[:, i]
            metrics[stage] = {
                "P50 (Median)": f"{float(np.percentile(vals, 50)):.2f} ms",
                "P70": f"{float(np.percentile(vals, 70)):.2f} ms",
                "P100 (Max)": f"{float(np.percentile(vals, 100)):.2f} ms"
            }
        return metrics
