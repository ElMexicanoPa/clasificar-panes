# -*- coding: utf-8 -*-
import os
import cv2
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelos.pkl')

# Cargar modelos desde pickle
with open(MODEL_PATH, 'rb') as f:
    data = pickle.load(f)

modelos = data['modelos']
nombres = data['nombres']
print(f"Modelos cargados: {nombres}")

@app.route("/")
def index():
    return render_template("index.html", modelos=nombres)

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    img_data = data["imagen"].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "No se pudo procesar la imagen"}), 400

    img_resized = cv2.resize(img, (64, 64))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_flat = img_gray.flatten().reshape(1, -1) / 255.0

    clases = {0: "CONCHA 🐚", 1: "OJO 👁️"}
    resultado = {}

    for fondo in ["IM_F_BLANCO", "IM_F_COLOR"]:
        if fondo in modelos:
            pred = modelos[fondo].predict(img_flat)[0]
            proba = modelos[fondo].predict_proba(img_flat)[0]
            resultado[fondo] = {
                "clase": clases[pred],
                "modelo": nombres[fondo],
                "confianza": f"{max(proba):.2%}"
            }

    return jsonify(resultado)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Servidor corriendo en puerto {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
