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
    except ValueError as e:
        # Handle RDKit conversion errors gracefully
        error_msg = str(e)
        logging.error(f"InChI conversion error: {error_msg}")
        return jsonify({'error': error_msg, 'inchi': inchi}), 400
    finally:
        predict_lock.release()

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Predict properties for chemical structures.

    GET (legacy): Single property prediction with ?inchi=...&property_token=...
    POST (tool registry): All properties prediction with {"smiles": [...]} or {"inchi": [...]}

    POST Request:
    {
        "smiles": ["CCO", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"],  # SMILES strings
        // OR
        "inchi": ["InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"]   # InChI strings
    }

    POST Response:
    {
        "predictions": [
            {
                "input": "CCO",
                "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
                "properties": [
                    {
                        "property_token": 0,
                        "property": {...},
                        "value": 0.123
                    },
                    ...
                ]
            }
        ],
        "count": 1,
        "total_properties": 6647
    }
    """
    if request.method == 'POST':
        # New POST endpoint for tool registry
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Accept either SMILES or InChI
        smiles_input = data.get('smiles', [])
        inchi_input = data.get('inchi', [])

        if not smiles_input and not inchi_input:
            return jsonify({"error": "No SMILES or InChI provided"}), 400

        # Normalize to list
        if isinstance(smiles_input, str):
            smiles_input = [smiles_input]
        if isinstance(inchi_input, str):
            inchi_input = [inchi_input]

        # Convert SMILES to InChI if needed
        try:
            from rdkit import Chem
            from rdkit.Chem import inchi as rdkit_inchi

            inchi_list = []
            input_list = []

            for smiles in smiles_input:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return jsonify({"error": f"Invalid SMILES: {smiles}"}), 400
                inchi_str = rdkit_inchi.MolToInchi(mol)
                if not inchi_str:
                    return jsonify({"error": f"Could not convert SMILES to InChI: {smiles}"}), 400
                inchi_list.append(inchi_str)
                input_list.append(smiles)

            for inchi_str in inchi_input:
                inchi_list.append(inchi_str)
                input_list.append(inchi_str)

        except Exception as e:
            logging.exception("SMILES/InChI conversion error")
            return jsonify({"error": f"Conversion error: {str(e)}"}), 500

        if not predict_lock.acquire(timeout=LOCK_TIMEOUT):
            return jsonify({'error': 'Server busy - try again later or use async /jobs endpoint'}), 503

        try:
            predictions = []

            for input_str, inchi_str in zip(input_list, inchi_list):
                logging.info(f"Predicting all properties for: {input_str}")

                property_predictions = predictor.predict_all_properties(inchi_str)

                predictions.append({
                    "input": input_str,
                    "inchi": inchi_str,
                    "properties": [dataclasses.asdict(p) for p in property_predictions]
                })

            return jsonify({
                "predictions": predictions,
                "count": len(predictions),
                "total_properties": len(property_predictions) if property_predictions else 0
            })

        except ValueError as e:
            # Handle RDKit conversion errors gracefully (invalid InChI/SMILES)
            logging.error(f"InChI conversion error: {str(e)}")
            return jsonify({
                "error": str(e),
                "message": "Invalid chemical structure - could not convert to SELFIES"
            }), 400
        except Exception as e:
            logging.exception("Prediction error")
            return jsonify({
                "error": str(e),
                "message": "Prediction failed"
            }), 500
        finally:
            predict_lock.release()

    else:
        # GET (legacy endpoint for single property)
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


@app.route('/properties', methods=['GET'])
def get_properties():
    """
    Get property definitions.

    Optional query parameters:
        - property_token: Get specific property by token (int)
        - search: Search properties by title (string)
        - limit: Maximum number of results (default 100)

    Response:
        [
            {
                "property_token": 123,
                "title": "Estrogen Receptor Alpha Binding",
                "source": "ToxCast",
                "categories": [...]
            },
            ...
        ]
    """
    property_token = request.args.get('property_token', type=int)
    search = request.args.get('search', type=str)
    limit = request.args.get('limit', type=int, default=100)

    # Get all properties
    properties = []
    for token, prop in predictor.property_map.items():
        if property_token is not None and token != property_token:
            continue
        if search and search.lower() not in prop.title.lower():
            continue

        properties.append({
            "property_token": token,
            "title": prop.title,
            "source": prop.source,
            "metadata": prop.metadata,
            "categories": [dataclasses.asdict(cat) for cat in prop.categories]
        })

        if len(properties) >= limit:
            break

    return jsonify(properties)


@app.route('/health', methods=['GET'])
def health_check():
    """Quick health check that returns immediately."""
    return jsonify({"status": "ok"}), 200


@app.route('/info', methods=['GET'])
def info():
    """Model information endpoint for tool registry"""
    return jsonify({
        "model": "ToxTransformer",
        "version": "2.1",
        "source": "https://github.com/biobricks-ai/toxtransformer",
        "description": "Decoder-only transformer for predicting 6,647 toxicology and bioactivity properties",
        "algorithm": "Multitask Encoder with MI-based contextualization",
        "properties_count": len(predictor.all_property_tokens),
        "capabilities": {
            "toxicity_prediction": True,
            "bioactivity_prediction": True,
            "adme_prediction": True,
            "environmental_toxicity": True,
            "multi_property_prediction": True,
            "context_aware_prediction": True
        },
        "input_formats": ["SMILES", "InChI"],
        "performance": {
            "prediction_time_gpu": "~3 seconds (T4)",
            "prediction_time_cached": "<0.1 seconds",
            "throughput_gpu": "~2,200 properties/second",
            "cuda_acceleration": True
        },
        "hardware": {
            "gpu_required": True,
            "gpu_type": "NVIDIA T4 or better",
            "gpu_memory_gb": 16
        },
        "endpoints": {
            "/predict (POST)": "Predict all properties - accepts SMILES or InChI",
            "/predict_all (GET)": "Predict all properties - accepts InChI",
            "/predict (GET)": "Predict single property - accepts InChI + property_token",
            "/properties": "List available properties",
            "/jobs (POST)": "Submit async prediction job",
            "/jobs/<job_id> (GET)": "Get job status and results",
            "/health": "Health check",
            "/info": "This endpoint"
        }
    })


if __name__ == '__main__':
    app.run(debug=True)
