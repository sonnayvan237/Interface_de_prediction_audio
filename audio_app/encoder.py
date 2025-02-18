from sklearn.preprocessing import LabelEncoder
import joblib

# ⚡ Liste des labels utilisés dans ton modèle
labels = ['no', 'two', 'backward', 'four', 'five', 'nine', 'right', 'follow', 'visual', 
'off', 'yes', 'six', 'dog', 'learn', 'left', 'bird', 'forward', 'wow', 'zero', 'eight', 'bed',
 'go', 'house', 'tree', 'seven','on', 'three', 'one', 'down', 'stop', 'up', 'happy', 'marvin', 
 'cat', 'sheila']  # Mets ici tes classes réelles

# 📌 Création et entraînement de l'encodeur
encoder = LabelEncoder()
encoder.fit(labels)

# 💾 Sauvegarde dans un fichier .pkl
joblib.dump(encoder, "label_encoder_commands.pkl")
print("✅ Encodeur sauvegardé avec succès !")