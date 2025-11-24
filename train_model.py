import os
import pandas as pd
import numpy as np
import requests, json, joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

# =============================
# 1) CONFIG BANXICO API
# =============================
BANXICO_TOKEN = "fd7467e959c6b931114ebabb3a555776bb93001a1ea1ed64c4cf7def38891b74"
BASE_URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/{series_id}/datos"

# IDs de ejemplo del catálogo SIE (ajusta si te sale vacío)
SERIES = {
    "tasa_objetivo": "SF61745",   # Tasa objetivo Banxico
    "inflacion_anual": "SP30578"  # Inflación anual INPC (puede variar)
}

def fetch_banxico_series(series_id):
    url = BASE_URL.format(series_id=series_id)
    params = {"token": BANXICO_TOKEN}
    headers = {"Bmx-Token": BANXICO_TOKEN, "Accept": "application/json"}

    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()["bmx"]["series"][0]["datos"]
    df = pd.DataFrame(data)
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True)
    df["dato"] = pd.to_numeric(df["dato"], errors="coerce")
    return df[["fecha", "dato"]].dropna()

def load_macro():
    dfs = []
    for name, sid in SERIES.items():
        tmp = fetch_banxico_series(sid).rename(columns={"dato": name})
        dfs.append(tmp.set_index("fecha"))
    macro = pd.concat(dfs, axis=1).sort_index().ffill()
    macro.reset_index(inplace=True)
    return macro

# =============================
# 2) CARGAR MICRODATOS
# =============================
def load_micro_data(path="application_train.csv"):
    df = pd.read_csv(path)
    return df

# =============================
# 3) FEATURE ENGINEERING
# =============================
def feature_engineering(df):
    df["income_monthly"] = df["AMT_INCOME_TOTAL"] / 12
    df["payment_burden"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["credit_income_ratio"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# =============================
# 4) MERGE MICRO + MACRO
# =============================
def merge_micro_macro(micro, macro):
    last = macro.iloc[-1]
    for col in ["tasa_objetivo", "inflacion_anual"]:
        micro[col] = last[col]
    return micro

# =============================
# 5) TRAIN + VALIDATE + PLOTS
# =============================
def train_model(df):
    target = "TARGET"

    features = [
        "income_monthly", "payment_burden", "credit_income_ratio",
        "DAYS_EMPLOYED", "DAYS_BIRTH",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "CNT_FAM_MEMBERS",
        "tasa_objetivo", "inflacion_anual"
    ]
    features = [f for f in features if f in df.columns]

    X = df[features].copy()
    y = df[target].copy()

    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    base_model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=3, max_iter=300, random_state=42
    )
    base_model.fit(X_train, y_train)

    model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    print("\n========== VALIDACIÓN ==========")
    print("AUC-ROC:", round(auc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, (proba > 0.5).astype(int)))
    print(classification_report(y_test, (proba > 0.5).astype(int)))

    os.makedirs("static", exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("static/roc_curve.png")
    plt.close()

    # Feature importance aproximada
    try:
        imp = X_train.var().sort_values(ascending=False)
        plt.figure()
        imp.head(10).plot(kind="bar")
        plt.title("Top 10 Feature Importance (approx)")
        plt.ylabel("Variance Importance")
        plt.tight_layout()
        plt.savefig("static/feature_importance.png")
        plt.close()
    except Exception as e:
        print("No se pudo guardar feature importance:", e)

    # KS curva simple
    kstable = pd.DataFrame({"y": y_test.values, "proba": proba})
    kstable["bucket"] = pd.qcut(kstable["proba"], 10, duplicates="drop")

    grouped = kstable.groupby("bucket").apply(
        lambda x: pd.Series({
            "bads": x["y"].sum(),
            "goods": (1 - x["y"]).sum()
        })
    )
    grouped["cum_bad_rate"] = (grouped["bads"] / grouped["bads"].sum()).cumsum()
    grouped["cum_good_rate"] = (grouped["goods"] / grouped["goods"].sum()).cumsum()

    plt.figure()
    plt.plot(grouped["cum_bad_rate"].values, label="Cumulative Bad Rate")
    plt.plot(grouped["cum_good_rate"].values, label="Cumulative Good Rate")
    plt.title("KS Curve")
    plt.legend()
    plt.savefig("static/ks_curve.png")
    plt.close()

    return model, features

def main():
    micro = load_micro_data("application_train.csv")
    macro = load_macro()

    micro = feature_engineering(micro)
    data = merge_micro_macro(micro, macro)

    model, features = train_model(data)

    joblib.dump(model, "model.pkl")
    with open("feature_list.json", "w") as f:
        json.dump(features, f)

    print("\nModelo guardado como model.pkl")
    print("Features guardadas como feature_list.json")
    print("Gráficas guardadas en static/")

if __name__ == "__main__":
    main()
