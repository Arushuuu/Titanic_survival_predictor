import os
import traceback
import joblib
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask and enable CORS (Cross-Origin Resource Sharing)
app = Flask(__name__)
CORS(app) 

# ==========================================
# 1. BULLETPROOF MODEL LOADING
# ==========================================
basedir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(basedir, "final_titanic_model.pkl")

model = None
try:
    print("--- ATTEMPTING TO LOAD MODEL VIA PICKLE ---")
    print(f"Target Path: {model_path}")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("--- MODEL LOADED SUCCESSFULLY ---")
except Exception as e:
    print("--- ERROR LOADING MODEL ---")
    print(f"Error Message: {e}")
    traceback.print_exc()
    model = None

# ==========================================
# 2. HEALTH CHECK ROUTE (For the browser)
# ==========================================
@app.route('/', methods=['GET'])
def home():
    if model is None:
        return "Titanic API is running, BUT THE MODEL FAILED TO LOAD. Check Render logs.", 500
    return "Titanic API is running perfectly and the Model is loaded!", 200

# ==========================================
# 3. PREDICTION ROUTE (For the frontend)
# ==========================================
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight requests from GitHub Pages
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    # Ensure model exists before trying to predict
    if model is None:
        return jsonify({"error": "Model not loaded on the server."}), 500

    try:
        # Get JSON data from frontend request
        data = request.get_json()
        
        # Extract features (Ensure these match your frontend JavaScript keys)
        pclass = float(data.get('Pclass', 3))
        sex = float(data.get('Sex', 0))
        age = float(data.get('Age', 29))
        sibsp = float(data.get('SibSp', 0))
        parch = float(data.get('Parch', 0))
        fare = float(data.get('Fare', 32.0))
        embarked = float(data.get('Embarked', 0))
        
        # Format for Scikit-Learn (2D array)
        features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        
        # Generate Prediction
        prediction = model.predict(features)[0]
        
        # Send response back to the website
        return jsonify({
            "prediction": int(prediction),
            "status": "success"
        })

    except Exception as e:
        print("--- PREDICTION ERROR ---")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

# Run the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

