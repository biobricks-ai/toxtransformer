from flask import Flask, request, jsonify
from werkzeug.serving import WSGIRequestHandler
import sqlite3
import threading
import logging
import dataclasses

import sys
sys.path.append('./flask_cvae')
from flask_cvae.predictor import Predictor, Prediction

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[logging.StreamHandler()])


logging.info("Starting Flask CVAE app...")
WSGIRequestHandler.protocol_version = "HTTP/1.1"

app = Flask(__name__)
app.config['TIMEOUT'] = 300  # 5 minutes in seconds

logging.info("Loading model...")
predict_lock = threading.Lock()
predictor = Predictor()
logging.info("Model loaded.")

@app.route('/predict_all', methods=['GET'])
def predict_all():
    logging.info(f"Predicting all properties for inchi: {request.args.get('inchi')}")
    inchi = request.args.get('inchi')
    with predict_lock:
        property_predictions : list[Prediction] = predictor.predict_all_properties(inchi)
    
    json_predictions = [dataclasses.asdict(p) for p in property_predictions]
    return jsonify(json_predictions)

@app.route('/predict', methods=['GET'])
def predict():
    logging.info(f"Predicting property for inchi: {request.args.get('inchi')} and property token: {request.args.get('property_token')}")
    inchi = request.args.get('inchi')
    property_token = request.args.get('property_token', None)
    if inchi is None or property_token is None:
        return jsonify({'error': 'inchi and property token parameters are required'})
    
    with predict_lock:
        prediction : Prediction = predictor.predict_property(inchi, int(property_token))

    return jsonify(dataclasses.asdict(prediction))

@app.route('/health', methods=['GET'])
def health_check():
    """Quick health check that returns immediately."""
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(debug=True)
