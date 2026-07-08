import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# 0. KONFIGURASI PATH 
# ═══════════════════════════════════════════════════════════════
MODEL_PATH  = Path("output/model/isolation_forest.pkl")
SCALER_PATH = Path("output/model/standard_scaler.pkl")
TEST_CSV    = Path("test_data.csv")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]   # 29 fitur


# ═══════════════════════════════════════════════════════════════
# 1. LOAD MODEL & SCALER
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  1. LOAD MODEL & SCALER")
print("═" * 60)

for path, label in [(MODEL_PATH, "Model"), (SCALER_PATH, "Scaler")]:
    if not path.exists():
        print(f"  ✗  {label} tidak ditemukan: {path}")
        print("     Pastikan kamu sudah menjalankan notebook sampai selesai.")
        sys.exit(1)

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print(f"  ✔  Model  loaded : {MODEL_PATH}  ({MODEL_PATH.stat().st_size/1024:.1f} KB)")
print(f"  ✔  Scaler loaded : {SCALER_PATH}  ({SCALER_PATH.stat().st_size/1024:.1f} KB)")
print(f"  ✔  n_estimators  : {model.n_estimators}")
print(f"  ✔  contamination : {model.contamination}")
print(f"  ✔  n_features    : {scaler.n_features_in_}")


# ═══════════════════════════════════════════════════════════════
# 2. LOAD TEST DATA
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  2. LOAD TEST DATA")
print("═" * 60)

if not TEST_CSV.exists():
    print(f"  ✗  File test tidak ditemukan: {TEST_CSV}")
    sys.exit(1)

df_test = pd.read_csv(TEST_CSV)
print(f"  ✔  {TEST_CSV}  →  {len(df_test)} baris, {df_test.shape[1]} kolom")

# Validasi kolom
missing_cols = [c for c in FEATURE_COLS if c not in df_test.columns]
if missing_cols:
    print(f"  ✗  Kolom kurang: {missing_cols}")
    sys.exit(1)

X_test = df_test[FEATURE_COLS].values

# Info per kategori (jika kolom TrueLabel ada)
if "TrueLabel" in df_test.columns:
    print(f"\n  Komposisi data test:")
    vc = df_test["TrueLabel"].value_counts()
    for cat, cnt in vc.items():
        print(f"    {cat:<25} : {cnt} baris")


# ═══════════════════════════════════════════════════════════════
# 3. PREPROCESSING — SCALING
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  3. PREPROCESSING — SCALING")
print("═" * 60)

X_scaled = scaler.transform(X_test)
print(f"  ✔  StandardScaler transform selesai.")
print(f"     shape  : {X_scaled.shape}")
print(f"     mean   : {X_scaled.mean():.4f}   (≈ 0 jika data mirip training)")
print(f"     std    : {X_scaled.std():.4f}    (≈ 1 jika data mirip training)")


# ═══════════════════════════════════════════════════════════════
# 4. PREDIKSI
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  4. PREDIKSI MODEL")
print("═" * 60)

predictions = model.predict(X_scaled)        # 1=Normal, -1=Anomaly
scores      = model.decision_function(X_scaled)  # Makin negatif → makin anomali

# Tambahkan ke DataFrame
df_result = df_test.copy()
df_result["Prediction"]     = predictions
df_result["AnomalyScore"]   = scores.round(6)
df_result["PredLabel"]      = df_result["Prediction"].map({1: "Normal", -1: "ANOMALY"})

total_normal  = (predictions == 1).sum()
total_anomaly = (predictions == -1).sum()
pct_anomaly   = total_anomaly / len(predictions) * 100

print(f"  Total data      : {len(predictions)}")
print(f"  Prediksi Normal : {total_normal}")
print(f"  Prediksi Anomaly: {total_anomaly}  ({pct_anomaly:.1f}%)")


# ═══════════════════════════════════════════════════════════════
# 5. DETAIL HASIL PER TRANSAKSI
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  5. DETAIL HASIL")
print("═" * 60)

id_col = "TransactionID" if "TransactionID" in df_result.columns else None

for idx, row in df_result.iterrows():
    tx_id   = row[id_col] if id_col else f"Row-{idx+1}"
    pred    = row["PredLabel"]
    score   = row["AnomalyScore"]
    amount  = row["Amount"]
    true_lbl = row.get("TrueLabel", "-")

    icon  = "🚨" if pred == "ANOMALY" else "✓ "
    score_tag = "← MENCURIGAKAN" if score < 0 else "← aman"

    print(f"  {icon}  [{tx_id}]  Amount={amount:>10.2f}  "
          f"Score={score:>8.4f}  {score_tag:<18}  "
          f"Pred={pred:<8}  (TrueLabel: {true_lbl})")


# ═══════════════════════════════════════════════════════════════
# 6. RINGKASAN AKURASI (jika TrueLabel tersedia)
# ═══════════════════════════════════════════════════════════════
if "TrueLabel" in df_result.columns:
    print("\n" + "═" * 60)
    print("  6. RINGKASAN AKURASI vs TRUE LABEL")
    print("═" * 60)

    # Mapping kategori ke binary: Fraud* / EdgeCase / Borderline → dianggap positif
    fraud_keywords = ["Fraud", "EdgeCase", "Borderline"]
    df_result["TrueBinary"] = df_result["TrueLabel"].apply(
        lambda x: "FraudGroup" if any(k in x for k in fraud_keywords) else "NormalGroup"
    )

    print(f"\n  {'TrueLabel':<25} {'Prediksi Normal':>15} {'Prediksi ANOMALY':>17}")
    print("  " + "-" * 60)

    grp = df_result.groupby(["TrueLabel", "PredLabel"]).size().unstack(fill_value=0)
    for lbl, row in grp.iterrows():
        n_normal  = row.get("Normal",  0)
        n_anomaly = row.get("ANOMALY", 0)
        print(f"  {lbl:<25} {n_normal:>15} {n_anomaly:>17}")

    print()
    # Hitung TP/TN/FP/FN sederhana
    tp = len(df_result[(df_result["TrueBinary"]=="FraudGroup")  & (df_result["PredLabel"]=="ANOMALY")])
    tn = len(df_result[(df_result["TrueBinary"]=="NormalGroup") & (df_result["PredLabel"]=="Normal")])
    fp = len(df_result[(df_result["TrueBinary"]=="NormalGroup") & (df_result["PredLabel"]=="ANOMALY")])
    fn = len(df_result[(df_result["TrueBinary"]=="FraudGroup")  & (df_result["PredLabel"]=="Normal")])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (tp + tn) / len(df_result)

    print(f"  * FraudGroup  = Fraud_Clear, Fraud_LargeAmount, EdgeCase, Borderline")
    print(f"  * NormalGroup = Normal, Normal_SmallAmount, Normal_ZeroAmount")
    print()
    print(f"  TP (Fraud terdeteksi)      : {tp}")
    print(f"  TN (Normal terdeteksi)     : {tn}")
    print(f"  FP (Normal salah anomaly)  : {fp}")
    print(f"  FN (Fraud lolos / missed)  : {fn}")
    print()
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")


# ═══════════════════════════════════════════════════════════════
# 7. SIMPAN HASIL
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  7. SIMPAN HASIL")
print("═" * 60)

OUT_PATH = Path("test_result.csv")
df_result.to_csv(OUT_PATH, index=False)
print(f"  ✔  Hasil disimpan ke: {OUT_PATH}")

print("\n" + "═" * 60)
print("  ✔  Testing selesai!")
print("═" * 60 + "\n")
