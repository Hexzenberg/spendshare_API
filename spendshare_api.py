# source venv/bin/activate
# pip install -r requirements.txt
# deactivate 

import joblib
import os
import math
from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)  

# Load model as a local file 
model_path = "catboost_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Please ensure 'catboost_model.pkl' is uploaded to your repository.")

model = joblib.load(model_path)

@app.route("/")
def home():
    return "Welcome to the Expense Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert month to sin and cos values
        month = data.pop("month")  
        month_sin = math.sin(2 * math.pi * (month / 12))
        month_cos = math.cos(2 * math.pi * (month / 12))

        # Add transformed values to the data dictionary
        data["Month_sin"] = month_sin
        data["Month_cos"] = month_cos

        prediction = model.predict([list(data.values())])
        return jsonify({"prediction": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    # Render provides a dynamic PORT, so use os.environ.get("PORT")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
