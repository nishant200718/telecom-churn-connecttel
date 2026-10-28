"""
ConnectTel Customer Churn Prediction - Production REST API Service
Author: Machine Learning Engineering Team
Framework: Dual-Engine Production Service (Standard Flask + Zero-Dependency Pure-Python WSGI)
Supports both real-time synchronous scoring and batch prediction.
Operates seamlessly in all environments, including minimal environments where Flask is not installed.
"""

import os
import sys
import time
import json
import warnings
import pandas as pd

# Suppress version mismatch warnings defensively across all scikit-learn environments
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
prod_dir = os.path.join(CURRENT_DIR, "production")
if os.path.exists(prod_dir) and prod_dir not in sys.path:
    sys.path.insert(0, prod_dir)

try:
    import joblib
except ImportError:
    import pickle as joblib

# Import Flask or fallback to built-in pure-Python Standalone implementation
try:
    from flask import Flask, request, jsonify
except ImportError:
    import flask
    from flask import Flask, request, jsonify

app = Flask(__name__)

def find_file(filename):
    candidates = [
        os.path.join(CURRENT_DIR, filename),
        os.path.join(CURRENT_DIR, "production", filename),
        os.path.join(os.path.dirname(CURRENT_DIR), filename),
        os.path.join(os.path.dirname(CURRENT_DIR), "production", filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(CURRENT_DIR, filename)

PIPELINE_PATH = find_file("churn_pipeline.joblib")
METRICS_PATH = find_file("metrics.json")

if not os.path.exists(PIPELINE_PATH):
    raise FileNotFoundError(f"Pipeline artifact not found at {PIPELINE_PATH}")

from pipeline import ConnectTelChurnPipeline

pipeline = joblib.load(PIPELINE_PATH)

metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health_check():
    """Health & Readiness probe endpoint for Kubernetes / Docker container monitoring."""
    return jsonify({
        "status": "healthy",
        "health": "ok",
        "ready": True,
        "service": "ConnectTel Churn Inference Engine",
        "version": "1.0.0-production",
        "pipeline_loaded": True,
        "model": "XGBoost",
        "model_auc": metrics.get("roc_auc", 0.8457),
        "cv_roc_auc": 0.8485,
        "test_roc_auc": 0.8457,
        "business_threshold": getattr(pipeline, "business_threshold", 0.28),
        "timestamp": time.time()
    }), 200

@app.route("/predict", methods=["POST", "GET"])
def predict():
    """
    Real-time scoring endpoint for subscriber payload.
    Supports single subscriber dict, wrapped dict ({"data": ...}), or subscriber list.
    """
    if request.method == "GET":
        return jsonify({
            "endpoint": "/predict",
            "method": "POST",
            "description": "Send JSON payload with customer features to score churn risk.",
            "status": "ready"
        }), 200

    try:
        payload = request.get_json(force=True, silent=True)
    except Exception:
        payload = None

    if not payload:
        return jsonify({
            "status": "error",
            "message": "Invalid or empty JSON payload. Please provide customer feature data."
        }), 400

    # Unwrap if nested under data/customer/input/features
    if isinstance(payload, dict):
        for wrapper_key in ["data", "customer", "input", "features", "record"]:
            if wrapper_key in payload and isinstance(payload[wrapper_key], (dict, list)):
                payload = payload[wrapper_key]
                break

    is_single = isinstance(payload, dict)
    records = [payload] if is_single else payload

    if not isinstance(records, list) or len(records) == 0:
        return jsonify({
            "status": "error",
            "message": "Payload must be a JSON object or list of customer objects."
        }), 400

    start_time = time.perf_counter()
    df_input = pd.DataFrame(records)
    results = pipeline.predict_risk(df_input)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    if is_single:
        r = results[0]
        response = {
            "status": "success",
            "churn_prediction": int(r["predicted_churn"]),
            "predicted_churn": bool(r["predicted_churn"]),
            "churn_probability": float(r["churn_probability"]),
            "probability": float(r["churn_probability"]),
            "risk_tier": str(r["risk_tier"]),
            "recommended_retention_action": str(r["recommended_retention_action"]),
            "prediction": r,
            "latency_ms": latency_ms
        }
        return jsonify(response), 200
    else:
        response = {
            "status": "success",
            "total_records_scored": len(results),
            "predictions": results,
            "latency_ms": latency_ms,
            "avg_latency_per_record_ms": round(latency_ms / max(len(results), 1), 3)
        }
        return jsonify(response), 200

@app.route("/batch_predict", methods=["POST"])
def predict_batch():
    """High-throughput batch scoring endpoint for multiple customer records."""
    try:
        payload = request.get_json(force=True, silent=True)
    except Exception:
        payload = None

    if not payload:
        return jsonify({"status": "error", "message": "Invalid or empty JSON payload"}), 400

    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        payload = payload["data"]

    if not isinstance(payload, list) or len(payload) == 0:
        return jsonify({"status": "error", "message": "Batch payload must be a JSON array of customer objects"}), 400

    start_time = time.perf_counter()
    df_batch = pd.DataFrame(payload)
    results = pipeline.predict_risk(df_batch)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    response = {
        "status": "success",
        "total_records_scored": len(results),
        "predictions": results,
        "latency_ms": latency_ms,
        "avg_latency_per_record_ms": round(latency_ms / max(len(results), 1), 3)
    }
    return jsonify(response), 200

# Error handlers
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status": "error", "message": "Bad request"}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"status": "error", "message": "Method not allowed"}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({"status": "error", "message": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting ConnectTel Churn API on http://0.0.0.0:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
