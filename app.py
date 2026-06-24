import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
# Enable CORS so your frontend can communicate with this backend
CORS(app)

# --- NEW: Use absolute path to reliably find the model file ---
basedir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(basedir, 'Titanic_rf.pkl')

# Load the trained model
try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return "Titanic ML API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load on the server.'}), 500

    try:
        data = request.json
        
        # Manual mapping based on alphabetical LabelEncoder sorting from your notebook
        sex_map = {'female': 0, 'male': 1}
        embarked_map = {'C': 0, 'Q': 1, 'S': 2}
        
        # Structure the data exactly as the model expects it: 
        # [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
        features = pd.DataFrame([{
            'Pclass': int(data['Pclass']),
            'Sex': sex_map[data['Sex']],
            'Age': float(data['Age']),
            'SibSp': int(data['SibSp']),
            'Parch': int(data['Parch']),
            'Fare': float(data['Fare']),
            'Embarked': embarked_map[data['Embarked']]
        }])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Convert numeric prediction back to readable text
        result = "Survived" if prediction == 1 else "Not Survived"
        
        return jsonify({'prediction': result})

    except KeyError as e:
        return jsonify({'error': f'Missing or incorrect data field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Use port Render assigns, or default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


    