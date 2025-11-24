from flask import Flask, render_template, request, jsonify
import joblib, json
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
features = json.load(open("feature_list.json"))

@app.route("/")
def home():
    return render_template("index.html")

# -------------------------
# SIMULADOR ML + PRODUCTO
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    credit_type = data.get("credit_type", "personal")

    x = []
    for f in features:
        x.append(float(data.get(f, 0)))
    x = np.array(x).reshape(1, -1)

    pd_default = model.predict_proba(x)[0, 1]

    thresholds = {
        "personal": 0.35,
        "automotriz": 0.30,
        "hipotecario": 0.25,
        "nomina": 0.40,
        "pyme": 0.28
    }
    th = thresholds.get(credit_type, 0.35)

    decision = "APROBADO" if pd_default < th else "RECHAZADO"

    return jsonify({
        "pd_default": round(float(pd_default), 4),
        "decision": decision,
        "threshold_used": th
    })

# -------------------------
# NUEVO: SENSIBILIDAD PD vs TASA BANXICO
# -------------------------
@app.route("/sensitivity", methods=["POST"])
def sensitivity():
    data = request.json

    # vector base del cliente
    base = {}
    for f in features:
        base[f] = float(data.get(f, 0))

    # rango de tasas a simular
    rate_min = float(data.get("rate_min", 5))
    rate_max = float(data.get("rate_max", 15))
    steps = int(data.get("steps", 11))

    rates = np.linspace(rate_min, rate_max, steps)
    pds = []

    for r in rates:
        base["tasa_objetivo"] = r  # mover solo la tasa Banxico
        x = np.array([base[f] for f in features]).reshape(1, -1)
        pd_r = model.predict_proba(x)[0, 1]
        pds.append(float(pd_r))

    return jsonify({
        "rates": rates.tolist(),
        "pds": pds
    })

if __name__ == "__main__":
    app.run(debug=True)
