# -*- coding: utf-8 -*-
import os
import pickle
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelos.pkl')

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

    img = Image.open(BytesIO(img_bytes)).convert("L").resize((64, 64))
    img_flat = np.array(img).flatten().reshape(1, -1) / 255.0

    clases = {0: "CONCHA 🐚", 1: "OJO 👁️"}
    resultado = {}
    UMBRAL_CONFIANZA = 0.70

    for fondo in ["IM_F_BLANCO", "IM_F_COLOR"]:
        if fondo in modelos:
            pred = modelos[fondo].predict(img_flat)[0]
            proba = modelos[fondo].predict_proba(img_flat)[0]
            confianza = max(proba)
            if confianza < UMBRAL_CONFIANZA:
                clase = "No es un pan 🚫"
            else:
                clase = clases[pred]
            resultado[fondo] = {
                "clase": clase,
                "modelo": nombres[fondo],
                "confianza": f"{confianza:.2%}"
            }

    return jsonify(resultado)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Servidor corriendo en puerto {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
