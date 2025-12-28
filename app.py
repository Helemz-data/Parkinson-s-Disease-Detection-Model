from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import librosa
import os

svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")

app = Flask(__name__)

scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_voice_features(file_path):
    y, sr = librosa.load(file_path, duration=3)

    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0])

    jitter = np.std(np.diff(y))
    shimmer = np.std(y)
    hnr = np.mean(librosa.effects.harmonic(y))

    return [pitch, jitter, shimmer, hnr]
VOICE_FEATURE_MAPPING = {
    "MDVP:Fo(Hz)": "pitch",
    "MDVP:Jitter(%)": "jitter",
    "MDVP:Shimmer": "shimmer",
    "HNR": "hnr"
}

@app.route("/")
def home():
    return render_template("index.html", feature_names=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    print(request.form)
    # Collect numeric inputs using feature names
    feature_dict = {}
    for feature in feature_names:
        value = request.form.get(feature)

        if value is None or value == "":
            return jsonify({
                "error": f"Missing feature: {feature}"
            }), 400

        feature_dict[feature] = float(value)

    # Handle optional voice upload
    voice = request.files.get("voice")
    if voice and voice.filename != "":
        file_path = os.path.join(UPLOAD_FOLDER, "voice.wav")
        voice.save(file_path)

        pitch, jitter, shimmer, hnr = extract_voice_features(file_path)

        extracted = {
            "pitch": pitch,
            "jitter": jitter,
            "shimmer": shimmer,
            "hnr": hnr
        }

        # Map extracted voice features to correct dataset columns
        for dataset_col, extracted_key in VOICE_FEATURE_MAPPING.items():
            if dataset_col in feature_dict:
                feature_dict[dataset_col] = extracted[extracted_key]

    # Rebuild feature vector in training order
    feature_vector = np.array(
        [feature_dict[f] for f in feature_names]
    ).reshape(1, -1)

    feature_vector = scaler.transform(feature_vector)

    model_choice = request.form.get("model")

    if model_choice == "rf":
        active_model = rf_model
    else:
        active_model = svm_model

    prediction = active_model.predict(feature_vector)[0]
    probability = active_model.predict_proba(feature_vector)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })

if __name__ == "__main__":
    app.run(debug=True)
