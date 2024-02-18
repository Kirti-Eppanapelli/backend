from flask import Flask, request, jsonify
import joblib
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load the updated model
try:
    model = joblib.load('form.pkl')
except FileNotFoundError:
    print("Error: Model file 'form.pkl' not found.")
except Exception as e:
    print("Error loading model:", str(e))
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model:
        # Get the form data
        data = request.json
        try:
            # Preprocess the data (convert to suitable format, handle missing values, etc.)
            # Make predictions
            prediction = model.predict(data)
            # Return predictions as JSON response
            return jsonify({'prediction': prediction.tolist()})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Model not loaded.'})

if __name__ == '__main__':
    app.run(debug=True)
