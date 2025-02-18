import os
import numpy as np
import soundfile as sf
import joblib
# Désactiver ONEDNN pour éviter les erreurs TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import requests
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from transformers import Wav2Vec2Processor
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 📌 Chemins des modèles
MODELS = {
    "command": {
        "url": "https://huggingface.co/sonnayvan237/Audio_recognition/tree/main/audio_recognition.h5",
        "local_path": "audio_recognition.h5",
        "encoder": "label_encoder_commands.pkl"
    },
}

# 📌 Téléchargement et chargement du modèle
def load_model(model_info):
    if not os.path.exists(model_info["local_path"]):
        print(f"Téléchargement du modèle depuis {model_info['url']}...")
        response = requests.get(model_info["url"], stream=True)
        if response.status_code == 200:
            with open(model_info["local_path"], "wb") as f:
                f.write(response.content)
            print(f"Modèle téléchargé : {model_info['local_path']}")
        else:
            print(f"⚠ Erreur lors du téléchargement : {response.status_code}")
            return None
    
    try:
        model_info["model"] = tf.keras.models.load_model(model_info["local_path"], compile=False)
        print(f"✅ Modèle chargé : {model_info['local_path']}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle {model_info['local_path']}: {e}")
        return None
    
    if os.path.exists(model_info["encoder"]):
        model_info["encoder"] = joblib.load(model_info["encoder"])
    else:
        print(f"❌ Encodeur manquant : {model_info['encoder']}")
        return None
    
    return model_info

# Charger les modèles
for model_type, info in MODELS.items():
    if not load_model(info):
        print(f"⚠ Problème avec le modèle {model_type}")

# 📌 Initialisation du processeur Wav2Vec2
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# 📌 Prétraitement audio
def preprocess_audio(file_path):
    try:
        # Charger l'audio avec la bonne fréquence d'échantillonnage
        speech, sr = librosa.load(file_path, sr=16000)

        # Extraction des MFCC (32 trames, 13 coefficients)
        mfcc = librosa.feature.mfcc(y=speech, sr=sr, n_mfcc=13, hop_length=int(sr / 32))
        
        # S'assurer que la taille est correcte (32, 13)
        if mfcc.shape[1] < 32:
            pad_width = 32 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :32]  # Tronquer à 32 trames

        # Transposer pour obtenir (32, 13) au lieu de (13, 32)
        mfcc = mfcc.T

        # Ajouter la dimension batch (1, 32, 13)
        return np.expand_dims(mfcc, axis=0)

    except Exception as e:
        return str(e)

# 📌 Prédiction
def predict_audio(file_path):
    model = MODELS["command"].get("model")
    encoder = MODELS["command"].get("encoder")

    if not model or not encoder:
        return {"error": "Modèle ou encodeur non disponible"}
    
    if isinstance(encoder, str):  # Vérifie si l'encodeur est une chaîne au lieu d'un objet
        return {"error": f"Problème de chargement de l'encodeur : {encoder}"}

    try:
        inputs = preprocess_audio(file_path)
        if isinstance(inputs, str):
            return {"error": inputs}

        logits = model(inputs)
        predicted_class = np.argmax(logits, axis=-1)[0]

        # Vérification avant d'utiliser inverse_transform
        if hasattr(encoder, "inverse_transform"):
            predicted_label = encoder.inverse_transform([predicted_class])[0]
        else:
            return {"error": "L'encodeur ne supporte pas inverse_transform"}

        return {"prediction": predicted_label}
    
    except Exception as e:
        return {"error": str(e)}

# 📌 Routes Flask
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier téléchargé"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Aucun fichier sélectionné"}), 400
    
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    
    result = predict_audio(file_path)
    os.remove(file_path)
    
    return jsonify(result)

# 📌 Lancer l'application
if __name__ == "__main__":
    app.run(debug=True)




# from flask import Flask, request, render_template, jsonify
# import tensorflow as tf
# import os
# import librosa
# import numpy as np
# import requests
# import joblib  # Pour charger les encoders

# app = Flask(_name_)
# UPLOAD_FOLDER = "static/uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Chemins des modèles hébergés
# MODELS = {
#     "command": {
#         "url": "https://huggingface.co/AloysRussel1/speech_command_model/resolve/main/speech_command_model.h5",
#         "local_path": "speech_command_model.h5",
#         "encoder": "label_encoder_commands.pkl"
#     },
#     "emotion": {
#         "url": "https://huggingface.co/AloysRussel1/speech_emotion_model/resolve/main/speech_emotion_model.h5",
#         "local_path": "speech_emotion_model.h5",
#         "encoder": "label_encoder_emotions.pkl"
#     }
# }

# # Fonction pour télécharger et charger un modèle
# def load_model(model_info):
#     # Vérifier si le modèle est déjà téléchargé
#     if not os.path.exists(model_info["local_path"]):
#         print(f"Téléchargement du modèle depuis {model_info['url']}...")
#         response = requests.get(model_info["url"], stream=True)
        
#         # Vérifier si la requête est réussie
#         if response.status_code == 200:
#             with open(model_info["local_path"], "wb") as f:
#                 f.write(response.content)
#             print(f"Modèle téléchargé : {model_info['local_path']}")
#         else:
#             print(f"⚠ Erreur lors du téléchargement : {response.status_code}")
#             return None

#     # Vérifier si le fichier est valide avant de charger
#     try:
#         model_info["model"] = tf.keras.models.load_model(model_info["local_path"])
#         print(f"Modèle chargé avec succès : {model_info['local_path']}")
#     except Exception as e:
#         print(f"❌ Erreur lors du chargement du modèle {model_info['local_path']}: {e}")
#         return None

#     # Charger l'encodeur associé
#     if os.path.exists(model_info["encoder"]):
#         model_info["encoder"] = joblib.load(model_info["encoder"])
#     else:
#         print(f"❌ Fichier d'encodeur manquant : {model_info['encoder']}")
#         return None
    
#     return model_info

# # Charger les modèles
# for model_type, info in MODELS.items():
#     if not load_model(info):
#         print(f"⚠ Problème avec le modèle {model_type}")

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "audio" not in request.files:
#         return jsonify({"error": "Aucun fichier envoyé"}), 400

#     file = request.files["audio"]
#     analysis_type = request.form.get("analysisType", "command")  # Par défaut : commande vocale

#     if analysis_type not in MODELS:
#         return jsonify({"error": "Type d'analyse invalide"}), 400

#     # Vérification du format audio
#     if not file.filename.endswith(".wav"):
#         return jsonify({"error": "Format non supporté. Veuillez uploader un fichier .wav"}), 400

#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)

#     # Prétraitement audio
#     audio, sr = librosa.load(file_path, sr=16000)
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

#     max_timesteps = 100
#     if mfccs.shape[1] < max_timesteps:
#         mfccs = np.pad(mfccs, ((0, 0), (0, max_timesteps - mfccs.shape[1])), mode='constant')
#     else:
#         mfccs = mfccs[:, :max_timesteps]

#     mfccs = np.expand_dims(mfccs.T, axis=0)

#     # Prédiction avec le bon modèle
#     model_info = MODELS[analysis_type]

#     if "model" not in model_info or model_info["model"] is None:
#         return jsonify({"error": "Le modèle n'est pas chargé correctement"}), 500

#     prediction = model_info["model"].predict(mfccs)
#     predicted_label = model_info["encoder"].inverse_transform([np.argmax(prediction)])[0]

#     return jsonify({"result": predicted_label})

# if _name_ == "_main_":
#     app.run(debug=True)
    
# if _name_ == "_main_":
#     port = int(os.environ.get("PORT", 10000))  # Render assigne un port automatiquement
#     app.run(host="0.0.0.0", port=port, debug=True)