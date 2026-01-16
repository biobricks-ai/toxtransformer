from flask import Flask, request, jsonify
from werkzeug.serving import WSGIRequestHandler
import sqlite3
import threading
import logging
import dataclasses
import time

import sys
sys.path.append('./flask_cvae')
from flask_cvae.predictor import Predictor, Prediction
from flask_cvae.job_queue import JobQueue, JobWorker, JobStatus

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[logging.StreamHandler()])


WSGIRequestHandler.protocol_version = "HTTP/1.1"

app = Flask(__name__)
app.config['TIMEOUT'] = 300  # 5 minutes in seconds

predict_lock = threading.Lock()
LOCK_TIMEOUT = 120  # seconds - prevent deadlock if request times out
predictor = Predictor()

# Initialize job queue and worker
job_queue = JobQueue()
job_worker = JobWorker(job_queue, predictor)
job_worker.start()

# ============== Synchronous Endpoints (existing) ==============

@app.route('/predict_all', methods=['GET'])
def predict_all():
    """Synchronous predict_all - blocks until complete."""
    logging.info(f"Predicting all properties for inchi: {request.args.get('inchi')}")
    inchi = request.args.get('inchi')

    if not predict_lock.acquire(timeout=LOCK_TIMEOUT):
        return jsonify({'error': 'Server busy - try again later or use async /jobs endpoint'}), 503

    try:
        property_predictions : list[Prediction] = predictor.predict_all_properties(inchi)
        json_predictions = [dataclasses.asdict(p) for p in property_predictions]
        return jsonify(json_predictions)
    finally:
        predict_lock.release()

@app.route('/predict', methods=['GET'])
def predict():
    """Synchronous predict - blocks until complete."""
    logging.info(f"Predicting property for inchi: {request.args.get('inchi')} and property token: {request.args.get('property_token')}")
    inchi = request.args.get('inchi')
    property_token = request.args.get('property_token', None)
    if inchi is None or property_token is None:
        return jsonify({'error': 'inchi and property token parameters are required'}), 400

    if not predict_lock.acquire(timeout=LOCK_TIMEOUT):
        return jsonify({'error': 'Server busy - try again later or use async /jobs endpoint'}), 503

    try:
        prediction : Prediction = predictor.predict_property(inchi, int(property_token))
        if prediction is None:
            return jsonify({'error': 'Prediction failed - invalid property token or molecule'}), 400
        return jsonify(dataclasses.asdict(prediction))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.exception(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        predict_lock.release()

# ============== Async Job Endpoints (new) ==============

@app.route('/jobs', methods=['POST'])
def submit_job():
    """
    Submit a prediction job to the queue.

    Request body (JSON):
        {
            "job_type": "predict_all" or "predict",
            "inchi": "InChI=1S/...",
            "property_token": 123  # only for job_type="predict"
        }

    Response:
        {
            "job_id": "uuid",
            "status": "pending",
            "queue_position": 3,
            "estimated_seconds": 225,
            "poll_url": "/jobs/{job_id}"
        }
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    job_type = data.get("job_type", "predict_all")
    inchi = data.get("inchi")

    if not inchi:
        return jsonify({"error": "Missing 'inchi' parameter"}), 400

    if job_type not in ["predict_all", "predict"]:
        return jsonify({"error": "job_type must be 'predict_all' or 'predict'"}), 400

    if job_type == "predict" and "property_token" not in data:
        return jsonify({"error": "property_token required for job_type='predict'"}), 400

    # Build params
    params = {"inchi": inchi}
    if job_type == "predict":
        params["property_token"] = data["property_token"]

    # Submit job
    job = job_queue.submit(job_type, params)

    # Get queue info
    position = job_queue.get_queue_position(job.job_id)
    eta = job_queue.estimate_completion(job.job_id)

    return jsonify({
        "job_id": job.job_id,
        "status": job.status.value,
        "queue_position": position,
        "estimated_seconds": round(eta, 1) if eta else None,
        "poll_url": f"/jobs/{job.job_id}"
    }), 202


@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """
    Get job status and result.

    Response (pending/processing):
        {
            "job_id": "uuid",
            "status": "processing",
            "progress": 0.45,
            "queue_position": 0,
            "estimated_seconds": 41.2
        }

    Response (completed):
        {
            "job_id": "uuid",
            "status": "completed",
            "progress": 1.0,
            "result": [...predictions...],
            "processing_time": 74.5
        }

    Response (failed):
        {
            "job_id": "uuid",
            "status": "failed",
            "error": "error message"
        }
    """
    job = job_queue.get(job_id)

    if job is None:
        return jsonify({"error": "Job not found"}), 404

    response = {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress
    }

    if job.status == JobStatus.PENDING:
        response["queue_position"] = job_queue.get_queue_position(job_id)
        response["estimated_seconds"] = round(job_queue.estimate_completion(job_id) or 0, 1)

    elif job.status == JobStatus.PROCESSING:
        response["queue_position"] = 0
        eta = job_queue.estimate_completion(job_id)
        response["estimated_seconds"] = round(eta, 1) if eta else None

    elif job.status == JobStatus.COMPLETED:
        response["result"] = job.result
        if job.started_at and job.completed_at:
            response["processing_time"] = round(job.completed_at - job.started_at, 1)

    elif job.status == JobStatus.FAILED:
        response["error"] = job.error

    return jsonify(response)


@app.route('/jobs', methods=['GET'])
def list_jobs():
    """
    Get queue statistics.

    Response:
        {
            "pending": 5,
            "processing": 1,
            "completed_24h": 127,
            "avg_processing_time": 74.5
        }
    """
    stats = job_queue.get_queue_stats()
    return jsonify(stats)


@app.route('/health', methods=['GET'])
def health_check():
    """Quick health check that returns immediately."""
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    app.run(debug=True)
