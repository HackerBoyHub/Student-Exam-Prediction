from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load saved model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Home route
@app.route('/')
def home():
    return render_template('passAfailpredict.html')


# Predict route
@app.route('/predict', methods=['POST'])
def predict():

    study_hours = request.form.get('study_hours')
    previous_score = request.form.get('previous_score')

    if not study_hours or not previous_score:
        return render_template(
            'passAfailpredict.html',
            prediction_text="Please enter both values."
        )

    study_hours = float(study_hours)
    previous_score = float(previous_score)

    features = np.array([[study_hours, previous_score]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    result = "Pass" if prediction[0] == 1 else "Fail"

    return render_template(
        'passAfailpredict.html',
        prediction_text=f"Prediction: {result}"
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)