
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

from backend.utils import predict, random_forest, xgboost_model, label_encoder

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "CIC-IoT2023 Intrusion Detection API",
        "status": "running"
    })


@app.route("/models", methods=["GET"])
def get_models():
    return jsonify({
        "available_models": [
            "random_forest",
            "xgboost"
        ]
    })


@app.route("/predict", methods=["POST"])
def make_prediction():

    try:
        # Check CSV upload
        if "file" not in request.files:
            return jsonify({"error": "No CSV file uploaded"}), 400

        file = request.files["file"]

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Please upload a CSV file"}), 400

        # Get selected model
        model_name = request.form.get("model", "random_forest")

        # Read uploaded CSV
        df = pd.read_csv(file)

        if df.empty:
            return jsonify({"error": "CSV file is empty"}), 400

        # Generate predictions
        predictions = predict(df, model_name)

        # Calculate prediction counts
        prediction_counts = pd.Series(predictions).value_counts()

        # Prepare results
        results = []

        for index, prediction in enumerate(predictions):
            results.append({
                "row": index + 1,
                "prediction": prediction
            })

        return jsonify({
            "success": True,
            "model_used": model_name,
            "total_records": len(predictions),
            "prediction_counts": prediction_counts.to_dict(),
            "results": results
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
