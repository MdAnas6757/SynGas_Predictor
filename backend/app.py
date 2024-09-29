from flask import Flask, request, jsonify
from model import predict_syngas_yield_all
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    results = predict_syngas_yield_all(data)  # Get predictions from all models
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
