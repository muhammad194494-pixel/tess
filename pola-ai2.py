import json
import glob
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import os

# ─── Konfigurasi ─────────────────────────────────────
DATA_PATTERN = "aviator_data*.json"  # semua file JSON yang kamu upload
MODEL_FILE = "aviator_model.h5"
SEQ_LEN = 4
scaler = MinMaxScaler(feature_range=(0, 1))

# ─── Load semua file JSON ───────────────────────────
def load_all_data():
    files = sorted(glob.glob(DATA_PATTERN))
    all_data = []
    for f in files:
        try:
            with open(f, "r") as file:
                data = json.load(file)
                all_data.extend(data)
        except:
            print(f"⚠️ Gagal membaca {f}")
    return all_data

# ─── Build model LSTM ───────────────────────────────
def build_model():
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=(SEQ_LEN,1)),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model

# ─── Train model dari semua data ────────────────────
def train_model(data, save_path=MODEL_FILE):
    arr = np.array(data).reshape(-1,1)
    scaled = scaler.fit_transform(arr).flatten()

    X, y = [], []
    for i in range(len(scaled) - SEQ_LEN):
        X.append(scaled[i:i+SEQ_LEN])
        y.append(scaled[i+SEQ_LEN])

    X = np.array(X).reshape(-1, SEQ_LEN, 1)
    y = np.array(y)

    model = build_model()
    model.fit(X, y, epochs=500, batch_size=16, verbose=0)
    model.save(save_path)
    print(f"✅ Training selesai, model disimpan di {save_path}")
    return model

# ─── Load model yang sudah ada ──────────────────────
def load_trained_model():
    if os.path.exists(MODEL_FILE):
        print(f"🔄 Memuat model dari {MODEL_FILE}...")
        return load_model(MODEL_FILE)
    return None

# ─── Prediksi multiplier berikutnya ────────────────
def predict_next(model, last_n):
    arr = np.array(last_n).reshape(-1,1)
    scaled = scaler.transform(arr).flatten()
    X = scaled.reshape(1, SEQ_LEN, 1)
    pred_scaled = model.predict(X, verbose=0)
    pred_real = scaler.inverse_transform(pred_scaled)[0][0]
    return max(1.0, float(pred_real))

# ─── Main ──────────────────────────────────────────
data = load_all_data()
print(f"📂 Total data dari semua file: {len(data)} putaran")

if len(data) < SEQ_LEN + 1:
    print(f"⚠️ Data tidak cukup (butuh minimal {SEQ_LEN+1} putaran).")
    exit()

# Training atau reload model
model = train_model(data)

# Menu prediksi sederhana
while True:
    inp = input(f"\nMasukkan {SEQ_LEN} multiplier terakhir, pisahkan koma (q untuk keluar): ")
    if inp.lower() == 'q':
        break
    try:
        last_n = [float(x.strip()) for x in inp.split(",")]
        if len(last_n) != SEQ_LEN:
            print(f"⚠️ Harus tepat {SEQ_LEN} angka")
            continue
        pred = predict_next(model, last_n)
        print(f"🎯 Prediksi multiplier berikutnya: {pred:.2f}x")
    except:
        print("❌ Input salah")
