from flask import Flask, request, jsonify
from model import predict_syngas_yield_all
from flask_cors import CORS

app = Flask(__name__, static_folder="../frontend/build", static_url_path="")
CORS(app)  # Enable CORS for frontend-backend communication

# Serve the React frontend
@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    results = predict_syngas_yield_all(data)  # Get predictions from all models
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
