# source venv/bin/activate
# pip install -r requirements.txt
# deactivate 

import joblib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)  

# Google Drive File ID to download the model
FILE_ID = "1JLIHJpaYfTkrTUKKTiiQPHV6pXcG7KOw"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

model_path = "catboost_model.pkl"
response = requests.get(DOWNLOAD_URL)
with open(model_path, "wb") as file:
    file.write(response.content)

# Loading the model
model = joblib.load(model_path)

@app.route("/")
def home():
    return "Welcome to the Expense Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Convert month to sin and cos values
        import math
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
    app.run(debug=True)
