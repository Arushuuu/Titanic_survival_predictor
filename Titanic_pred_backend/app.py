from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
# Enable CORS so your frontend can communicate with this backend
CORS(app)

# Load the trained model
model = joblib.load('titanic_rf_model.pkl')

@app.route('/')
def home():
    return "Titanic ML API is running!"

@app.route('/predict', methods=['POST'])
def predict():
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

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)