FROM python:3.12-slim

WORKDIR /app

# Copy model artifacts and dependencies
COPY MLProject/artifacts /app/artifacts
COPY MLProject/conda.yaml /app/conda.yaml

# Install dependencies
RUN pip install mlflow==2.19.0 scikit-learn==1.4.1 pandas==2.2.0 numpy==1.26.3 joblib==1.3.2 flask==2.0.1

# Copy model file
COPY MLProject/artifacts/model.pkl /app/model.pkl

# Create a Flask app for serving the model
RUN echo 'from flask import Flask, request, jsonify\n\
import numpy as np\n\
import joblib\n\
import json\n\
\n\
app = Flask(__name__)\n\
\n\
# Load the model\n\
model = joblib.load("/app/model.pkl")\n\
\n\
@app.route("/")\n\
def home():\n\
    return "Loan Approval Model API"\n\
\n\
@app.route("/predict", methods=["POST"])\n\
def predict():\n\
    try:\n\
        # Get input features from request\n\
        features = request.json\n\
        \n\
        # Convert to numpy array\n\
        input_features = np.array(list(features.values())).reshape(1, -1)\n\
        \n\
        # Make prediction\n\
        prediction = model.predict(input_features)\n\
        probability = model.predict_proba(input_features)[0][1]  # Probability of class 1\n\
        \n\
        # Return prediction\n\
        return jsonify({\n\
            "prediction": int(prediction[0]),\n\
            "probability": float(probability),\n\
            "status": "Approved" if prediction[0] == 1 else "Rejected"\n\
        })\n\
    except Exception as e:\n\
        return jsonify({"error": str(e)})\n\
\n\
# Load model info\n\
try:\n\
    with open("/app/artifacts/model_info.json", "r") as f:\n\
        model_info = json.load(f)\n\
except:\n\
    model_info = {}\n\
\n\
@app.route("/info", methods=["GET"])\n\
def info():\n\
    return jsonify(model_info)\n\
\n\
if __name__ == "__main__":\n\
    app.run(host="0.0.0.0", port=5000)' > /app/app.py

EXPOSE 5000

CMD ["python", "app.py"]
