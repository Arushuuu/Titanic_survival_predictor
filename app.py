import os
import pickle
import traceback
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.', static_url_path='')
basedir = os.path.abspath(os.path.dirname(__file__))

# Pointing explicitly to the new, clean model file
model_path = os.path.join(basedir, "final_titanic_model.pkl")

# Load the model at startup
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

@app.route('/')
def index():
    """Serves the main index.html frontend."""
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Handles the prediction requests from the frontend."""
    
    # Handle CORS preflight requests from the browser
    if request.method == 'OPTIONS':
        return jsonify({'message': 'CORS preflight successful'}), 200
        
    if not model:
        return jsonify({'error': 'Server Error: The machine learning model failed to load.'}), 500
        
    try:
        # Get data from the frontend JSON payload
        data = request.json
        
        # --- 1. DATA PREPROCESSING ---
        
        # Convert Gender string to float (Female = 1, Male = 0)
        gender_input = str(data.get('Sex', '')).lower()
        if gender_input == 'female':
            sex = 1.0
        else:
            sex = 0.0
            
        # Convert Embarked string to float (C = 0, Q = 1, S = 2)
        embarked_input = str(data.get('Embarked', '')).lower()
        if embarked_input == 'c':
            embarked = 0.0
        elif embarked_input == 'q':
            embarked = 1.0
        else:
            embarked = 2.0
            
        # --- 2. FEATURE ARRAY ASSEMBLY ---
        # Ensure these are in the EXACT order your Random Forest model was trained on!
        features = [
            float(data.get('Pclass', 3)),
            float(sex),
            float(data.get('Age', 30)),
            float(data.get('SibSp', 0)),
            float(data.get('Parch', 0)),
            float(data.get('Fare', 32.0)),
            float(embarked)
        ]
        
        # --- 3. MAKE PREDICTION ---
        prediction = model.predict([features])
        
        # --- 4. FORMAT RESPONSE ---
        survived = int(prediction[0])
        
        if survived == 1:
            result_text = "Survived"
        else:
            result_text = "Did Not Survive"
            
        return jsonify({
            'prediction': result_text, 
            'survived_flag': survived
        })
        
    except Exception as e:
        # If there is a math or conversion error, log it and send it to the frontend
        print("--- PREDICTION ERROR ---")
        traceback.print_exc()
        return jsonify({'error': f"Data processing error: {str(e)}"}), 400

if __name__ == '__main__':
    # Bind to 0.0.0.0 for Render deployment compatibility
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

