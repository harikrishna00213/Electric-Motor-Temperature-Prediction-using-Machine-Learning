import numpy as np
from flask import Flask, request, render_template
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.save")
scaler = joblib.load("transform.save")

# =========================
# Home Page
# =========================
@app.route('/')
def home():
    return render_template('Manual_predict.html')

# =========================
# Prediction Route
# =========================
@app.route('/y_predict', methods=['POST'])
def y_predict():
    try:
        # Get values from HTML form
        input_values = [float(x) for x in request.form.values()]
        print("Actual Input:", input_values)

        # Convert to numpy array
        x_test = np.array(input_values).reshape(1, -1)

        # Scale input
        x_test_scaled = scaler.transform(x_test)
        print("Scaled Input:", x_test_scaled)

        # Predict
        prediction = model.predict(x_test_scaled)

        return render_template(
            'Manual_predict.html',
            prediction_text=f"Permanent Magnet Surface Temperature: {prediction[0]:.2f} °C"
        )

    except Exception as e:
        return render_template(
            'Manual_predict.html',
            prediction_text=f"Error: {str(e)}"
        )

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
