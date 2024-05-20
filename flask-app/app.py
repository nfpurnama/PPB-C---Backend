from flask import Flask, request
import run_model

app = Flask(__name__)
model = run_model.ModelHandler("../model/Batik CNN-Batik CNN-86.95.h5")

@app.route("/predict-batik", methods=["POST", "GET"])
def predict():
    data = request.json
    base64_string = data['image']  # Get the base64 string from the request
    result = model.predict(base64_string)
    return {
        "prediction": result
    }