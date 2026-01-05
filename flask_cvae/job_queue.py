"""
Async job queue for prediction requests.
Jobs are stored in SQLite and processed by a background worker.
"""
import sqlite3
import threading
import time
import uuid
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Job:
    job_id: str
    job_type: str  # "predict_all" or "predict"
    params: Dict[str, Any]  # {"inchi": "...", "property_token": ...}
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0

class JobQueue:
    # Track average processing time for ETA estimation
    AVG_PREDICT_ALL_TIME = 75.0  # seconds (updated dynamically)

    def __init__(self, db_path: str = "cache/jobs.sqlite"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

        # Track processing times for ETA
        self._processing_times: List[float] = []

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                params TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                result TEXT,
                error TEXT,
                progress REAL DEFAULT 0.0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON jobs(created_at)")
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def submit(self, job_type: str, params: Dict[str, Any]) -> Job:
        """Submit a new job to the queue."""
        job = Job(
            job_id=str(uuid.uuid4()),
            job_type=job_type,
            params=params,
            status=JobStatus.PENDING,
            created_at=time.time()
        )

        with self.lock:
            conn = self._get_conn()
            conn.execute("""
                INSERT INTO jobs (job_id, job_type, params, status, created_at, progress)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (job.job_id, job.job_type, json.dumps(params),
                  job.status.value, job.created_at, 0.0))
            conn.commit()
            conn.close()

        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        conn.close()

        if row is None:
            return None

        return Job(
            job_id=row['job_id'],
            job_type=row['job_type'],
            params=json.loads(row['params']),
            status=JobStatus(row['status']),
            created_at=row['created_at'],
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            result=json.loads(row['result']) if row['result'] else None,
            error=row['error'],
            progress=row['progress'] or 0.0
        )

    def get_next_pending(self) -> Optional[Job]:
        """Get the next pending job (FIFO order)."""
        with self.lock:
            conn = self._get_conn()
            row = conn.execute("""
                SELECT * FROM jobs
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT 1
            """, (JobStatus.PENDING.value,)).fetchone()

            if row is None:
                conn.close()
                return None

            # Mark as processing
            conn.execute("""
                UPDATE jobs SET status = ?, started_at = ?
                WHERE job_id = ?
            """, (JobStatus.PROCESSING.value, time.time(), row['job_id']))
            conn.commit()
            conn.close()

            return Job(
                job_id=row['job_id'],
                job_type=row['job_type'],
                params=json.loads(row['params']),
                status=JobStatus.PROCESSING,
                created_at=row['created_at'],
                started_at=time.time()
            )

    def update_progress(self, job_id: str, progress: float):
        """Update job progress (0.0 to 1.0)."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE jobs SET progress = ? WHERE job_id = ?",
            (progress, job_id)
        )
        conn.commit()
        conn.close()

    def complete(self, job_id: str, result: Any):
        """Mark job as completed with result."""
        completed_at = time.time()

        with self.lock:
            conn = self._get_conn()
            # Get started_at for timing
            row = conn.execute(
                "SELECT started_at FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()

            conn.execute("""
                UPDATE jobs SET status = ?, completed_at = ?, result = ?, progress = 1.0
                WHERE job_id = ?
            """, (JobStatus.COMPLETED.value, completed_at, json.dumps(result), job_id))
            conn.commit()
            conn.close()

            # Update average processing time
            if row and row['started_at']:
                processing_time = completed_at - row['started_at']
                self._processing_times.append(processing_time)
                if len(self._processing_times) > 100:
                    self._processing_times = self._processing_times[-100:]
                self.AVG_PREDICT_ALL_TIME = sum(self._processing_times) / len(self._processing_times)

    def fail(self, job_id: str, error: str):
        """Mark job as failed with error message."""
        conn = self._get_conn()
        conn.execute("""
            UPDATE jobs SET status = ?, completed_at = ?, error = ?
            WHERE job_id = ?
        """, (JobStatus.FAILED.value, time.time(), error, job_id))
        conn.commit()
        conn.close()

    def get_queue_position(self, job_id: str) -> int:
        """Get position in queue (0 = processing, 1 = first in queue, etc.)."""
        conn = self._get_conn()

        # Check if processing
        row = conn.execute(
            "SELECT status, created_at FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()

        if row is None:
            conn.close()
            return -1

        if row['status'] == JobStatus.PROCESSING.value:
            conn.close()
            return 0

        if row['status'] != JobStatus.PENDING.value:
            conn.close()
            return -1

        # Count jobs ahead in queue
        count = conn.execute("""
            SELECT COUNT(*) as cnt FROM jobs
            WHERE status IN (?, ?) AND created_at < ?
        """, (JobStatus.PENDING.value, JobStatus.PROCESSING.value,
              row['created_at'])).fetchone()['cnt']

        conn.close()
        return count + 1  # +1 because there might be one processing

    def estimate_completion(self, job_id: str) -> Optional[float]:
        """Estimate seconds until job completion."""
        job = self.get(job_id)
        if job is None:
            return None

        if job.status == JobStatus.COMPLETED:
            return 0.0
        elif job.status == JobStatus.FAILED:
            return None
        elif job.status == JobStatus.PROCESSING:
            # Estimate based on progress
            if job.progress > 0:
                elapsed = time.time() - (job.started_at or job.created_at)
                remaining = elapsed * (1.0 - job.progress) / job.progress
                return remaining
            else:
                return self.AVG_PREDICT_ALL_TIME
        else:
            # Pending - estimate based on queue position
            position = self.get_queue_position(job_id)
            return position * self.AVG_PREDICT_ALL_TIME

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        conn = self._get_conn()

        pending = conn.execute(
            "SELECT COUNT(*) as cnt FROM jobs WHERE status = ?",
            (JobStatus.PENDING.value,)
        ).fetchone()['cnt']

        processing = conn.execute(
            "SELECT COUNT(*) as cnt FROM jobs WHERE status = ?",
            (JobStatus.PROCESSING.value,)
        ).fetchone()['cnt']

        completed_24h = conn.execute("""
            SELECT COUNT(*) as cnt FROM jobs
            WHERE status = ? AND completed_at > ?
        """, (JobStatus.COMPLETED.value, time.time() - 86400)).fetchone()['cnt']

        conn.close()

        return {
            "pending": pending,
            "processing": processing,
            "completed_24h": completed_24h,
            "avg_processing_time": self.AVG_PREDICT_ALL_TIME
        }

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove completed/failed jobs older than max_age_hours."""
        cutoff = time.time() - (max_age_hours * 3600)
        conn = self._get_conn()
        conn.execute("""
            DELETE FROM jobs
            WHERE status IN (?, ?) AND completed_at < ?
        """, (JobStatus.COMPLETED.value, JobStatus.FAILED.value, cutoff))
        conn.commit()
        conn.close()


class JobWorker:
    """Background worker that processes jobs from the queue."""

    def __init__(self, queue: JobQueue, predictor):
        self.queue = queue
        self.predictor = predictor
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background worker."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logging.info("Job worker started")

    def stop(self):
        """Stop the background worker."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logging.info("Job worker stopped")

    def _run(self):
        """Main worker loop."""
        while self.running:
            try:
                job = self.queue.get_next_pending()

                if job is None:
                    # No jobs, sleep briefly
                    time.sleep(0.5)
                    continue

                logging.info(f"Processing job {job.job_id}: {job.job_type}")

                try:
                    if job.job_type == "predict_all":
                        result = self._process_predict_all(job)
                    elif job.job_type == "predict":
                        result = self._process_predict(job)
                    else:
                        raise ValueError(f"Unknown job type: {job.job_type}")

                    self.queue.complete(job.job_id, result)
                    logging.info(f"Job {job.job_id} completed")

                except Exception as e:
                    logging.error(f"Job {job.job_id} failed: {e}")
                    self.queue.fail(job.job_id, str(e))

            except Exception as e:
                logging.error(f"Worker error: {e}")
                time.sleep(1)

    def _process_predict_all(self, job: Job) -> List[Dict]:
        """Process a predict_all job with progress updates."""
        import dataclasses

        inchi = job.params.get("inchi")
        if not inchi:
            raise ValueError("Missing 'inchi' parameter")

        # Get predictions with progress callback
        predictions = self.predictor.predict_all_properties(
            inchi,
            progress_callback=lambda p: self.queue.update_progress(job.job_id, p)
        )

        return [dataclasses.asdict(p) for p in predictions]

    def _process_predict(self, job: Job) -> Dict:
        """Process a single predict job."""
        import dataclasses

        inchi = job.params.get("inchi")
        property_token = job.params.get("property_token")

        if not inchi or property_token is None:
            raise ValueError("Missing 'inchi' or 'property_token' parameter")

        prediction = self.predictor.predict_property(inchi, int(property_token))
        return dataclasses.asdict(prediction)
