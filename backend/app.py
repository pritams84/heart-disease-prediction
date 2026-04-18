from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained model and feature list
try:
    model = joblib.load("model.pkl")
    trained_features = joblib.load("features.pkl")
    print("Model and features loaded successfully!")
except Exception as e:
    print(f"Error loading model artifacts: {e}. Run train_model.py first.")
    exit(1)

@app.route("/")
def home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error loading index.html: {e}"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Create a DataFrame for the input
        input_df = pd.DataFrame([data])
        
        # Ensure numeric fields are numeric
        numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col])

        # One-hot encode the input using the same logic as training
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols)
        
        # Reindex to match the trained features (filling missing columns with 0)
        input_aligned = input_encoded.reindex(columns=trained_features, fill_value=0)
        
        # Make prediction
        prediction = model.predict(input_aligned)[0]
        
        result = "High Risk" if prediction == 1 else "Low Risk"
        return jsonify({
            "prediction": result,
            "risk_score": int(prediction),
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Invalid input data."
        }), 400

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

if __name__ == "__main__":
    app.run()
