import os
import joblib
from flask import Flask, request, jsonify
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Chemin du modèle (modifiable via la var. d'env MODEL_PATH)
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

app = Flask(__name__)

# Chargement initial du modèle si existant
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# Mapping des vraies étiquettes métiers (cultivar 1,2,3)
REAL_LABELS = {
    0: 'barolo',
    1: 'grignolo',
    2: 'barbera'
}

@app.route('/train', methods=['GET'])
def train_and_save():
    seed = request.args.get('seed', type=int)
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    joblib.dump(clf, MODEL_PATH)
    # Stocker un échantillon par défaut pour GET /predict
    global default_sample
    default_sample = X_test[0].reshape(1, -1)
    global model
    model = clf
    return jsonify({
        "message": f"Modèle sauvegardé dans {MODEL_PATH}",
        "accuracy": acc,
        "seed_used": seed
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modèle non entraîné. Appelez d’abord /train"}), 400

    # GET sans param: utilise default_sample défini lors du dernier /train
    if request.method == 'GET':
        X = globals().get('default_sample')
        if X is None:
            return jsonify({"error": "Aucun échantillon par défaut. Appelez d'abord /train"}), 400
    else:
        data = request.get_json()
        if not data or 'instances' not in data:
            return jsonify({"error": "Clé 'instances' manquante dans le JSON"}), 400
        X = np.array(data['instances'])

    # Prédiction
    preds = model.predict(X)
    predicted_real = [REAL_LABELS[i] for i in preds]

    # GET renvoie un seul label, POST liste
    if request.method == 'GET':
        return jsonify({"predicted_real_label": predicted_real[0]})
    return jsonify({"predicted_real_labels": predicted_real}), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
