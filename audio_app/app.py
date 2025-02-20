import os
import numpy as np
import librosa
import joblib
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import requests
import soundfile as sf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Configuration Flask
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Modèle et encodeur
MODEL_URL = "https://huggingface.co/sonnayvan237/Audio_recognition/resolve/main/audio_recognition.h5"
ENCODER_URL = "https://huggingface.co/sonnayvan237/Audio_recognition/resolve/main/label_encoder_commands.pkl"
MODEL_PATH = "audio_recognition.h5"
ENCODER_PATH = "label_encoder_commands.pkl"

def download_file(url, local_path):
    """Télécharge un fichier si non disponible localement."""
    if not os.path.exists(local_path):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
        else:
            raise Exception(f"Erreur de téléchargement ({response.status_code})")

def load_resources():
    """Charge le modèle et l'encodeur."""
    download_file(MODEL_URL, MODEL_PATH)
    download_file(ENCODER_URL, ENCODER_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

# Charger le modèle et l'encodeur
MODEL, ENCODER = load_resources()

def preprocess_audio(file_path):
    """Convertit l'audio en MFCCs."""
    speech, sr = librosa.load(file_path, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=speech, sr=sr, n_mfcc=13, hop_length=int(sr / 32))
    mfcc = np.pad(mfcc, ((0, 0), (0, max(0, 32 - mfcc.shape[1]))), mode="constant")[:, :32]
    return np.expand_dims(mfcc.T, axis=0)

def predict_audio(file_path):
    """Effectue une prédiction sur un fichier audio."""
    try:
        inputs = preprocess_audio(file_path)
        logits = MODEL.predict(inputs)
        prediction = ENCODER.inverse_transform([np.argmax(logits)])[0]
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Fichier vide"}), 400
    
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(file_path)
    result = predict_audio(file_path)
    os.remove(file_path)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)