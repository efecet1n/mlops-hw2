# Flask API for Flight Delay Prediction
# MLOps HW2 - Efe Çetin

from flask import Flask, request, jsonify
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import (
    hash_airport_code,
    hash_airline_code,
    categorize_delay,
    extract_features
)

app = Flask(__name__)

# Delay category labels
DELAY_LABELS = {
    0: "On-time (0-10 min)",
    1: "Medium Delay (11-30 min)",
    2: "Large Delay (31+ min)"
}


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for smoke testing."""
    return jsonify({
        "status": "healthy",
        "service": "flight-delay-prediction"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict flight delay category.
    
    Request body:
        {
            "origin": "JFK",
            "dest": "LAX",
            "airline": "UA",
            "dep_time": 800,
            "arr_time": 1100,
            "elapsed_time": 180,
            "distance": 2475
        }
    
    Response:
        {
            "origin_hash": 42,
            "dest_hash": 17,
            "airline_hash": 5,
            "prediction": 0,
            "prediction_label": "On-time (0-10 min)"
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Required fields
        required = ["origin", "dest", "airline"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Extract hashed features
        origin_hash = hash_airport_code(data["origin"])
        dest_hash = hash_airport_code(data["dest"])
        airline_hash = hash_airline_code(data["airline"])
        
        # For now, use a simple heuristic-based prediction
        # In production, this would use the ML model
        # Random prediction based on hashes (placeholder)
        prediction = (origin_hash + dest_hash + airline_hash) % 3
        
        return jsonify({
            "origin_hash": origin_hash,
            "dest_hash": dest_hash,
            "airline_hash": airline_hash,
            "prediction": prediction,
            "prediction_label": DELAY_LABELS[prediction]
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/features", methods=["POST"])
def get_features():
    """
    Extract features from flight data without prediction.
    
    Request body:
        {
            "origin": "JFK",
            "dest": "LAX",
            "airline": "UA"
        }
    
    Response:
        {
            "origin_hash": 42,
            "dest_hash": 17,
            "airline_hash": 5
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        features = extract_features(
            data.get("origin", ""),
            data.get("dest", ""),
            data.get("airline", "")
        )
        
        return jsonify(features), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
