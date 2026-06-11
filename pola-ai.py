"""
Aviator AI Predictor | MaelCorp
================================
Progressive Sequential Training:
  Step 1 : Train  dari aviator_data.json
  Step 2 : Lanjut dari aviator_data1.json  (weights tidak di-reset)
  Step 3 : Lanjut dari aviator_data2.json
  ...
  Step N : Lanjut dari aviator_dataN.json
  Final  : Lanjut dari SEMUA data gabungan (paling akurat)

File yang dihasilkan:
  aviator_data.json     -> data utama (input baru disimpan di sini)
  aviator_model.keras   -> model tersimpan
  aviator_scaler.pkl    -> scaler tersimpan
"""

import json
import os
import glob
import numpy as np
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# === Konfigurasi ===
DATA_MAIN   = "aviator_data.json"
MODEL_FILE  = "aviator_model.keras"
SCALER_FILE = "aviator_scaler.pkl"
SEQ_LEN     = 4
MIN_DATA    = SEQ_LEN + 1

# === Load / Save Data ===
def load_data():
    try:
        with open(DATA_MAIN, "r") as f:
            return json.load(f)
    except:
        return []

def save_data(data):
    with open(DATA_MAIN, "w") as f:
        json.dump(data, f, indent=2)

# === Auto-detect semua file aviator_data*.json ===
def find_all_data_files():
    """
    Cari semua file aviator_data*.json di folder yang sama.
    Urutan: aviator_data.json, aviator_data1.json, aviator_data2.json, dst
    """
    files = []

    # File utama dulu
    if os.path.exists(DATA_MAIN):
        files.append(DATA_MAIN)

    # File bernomor (data1, data2, data3, ...)
    numbered = sorted(
        glob.glob("aviator_data[0-9]*.json"),
        key=lambda f: int(''.join(filter(str.isdigit, f)) or 0)
    )
    for f in numbered:
        if f not in files:
            files.append(f)

    return files

def load_file(filepath):
    try:
        with open(filepath, "r") as f:
            raw = json.load(f)
            # Support format list langsung atau {"data": [...]}
            if isinstance(raw, list):
                return [float(x) for x in raw if float(x) >= 1.0]
            elif isinstance(raw, dict):
                for key in ["data", "multipliers", "values", "results"]:
                    if key in raw and isinstance(raw[key], list):
                        return [float(x) for x in raw[key] if float(x) >= 1.0]
        return []
    except Exception as e:
        print(f"  [WARN] Gagal baca {filepath}: {e}")
        return []

# === Build Model (fresh) ===
def build_model():
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=(SEQ_LEN, 1)),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model

# === Siapkan X, y dari data ===
def prepare_xy(data, scaler, fit_scaler=False):
    arr = np.array(data).reshape(-1, 1)
    if fit_scaler:
        scaled = scaler.fit_transform(arr).flatten()
    else:
        scaled = scaler.transform(arr).flatten()

    X, y = [], []
    for i in range(len(scaled) - SEQ_LEN):
        X.append(scaled[i:i + SEQ_LEN])
        y.append(scaled[i + SEQ_LEN])

    if not X:
        return None, None

    return np.array(X).reshape(-1, SEQ_LEN, 1), np.array(y)

# === Save Model + Scaler ===
def save_model_and_scaler(model, scaler):
    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"  [SAVE] Model  -> {MODEL_FILE}")
    print(f"  [SAVE] Scaler -> {SCALER_FILE}")

# === Load Model + Scaler ===
def try_load_model_and_scaler():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            model  = load_model(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            print(f"  [OK] Model dimuat dari {MODEL_FILE}")
            return model, scaler
        except Exception as e:
            print(f"  [WARN] Gagal load model: {e}")
    return None, None

# === PROGRESSIVE SEQUENTIAL TRAINING ===
def progressive_train(silent=False):
    """
    Tahap 1: Temukan semua file data
    Tahap 2: Train satu per satu (weights tidak di-reset antar file)
    Tahap 3: Final train dari SEMUA data gabungan
    Model makin akurat di setiap tahap
    """
    files = find_all_data_files()
    if not files:
        print("[ERROR] Tidak ada file aviator_data*.json ditemukan!")
        return None, None

    print(f"\n[PROGRESSIVE TRAIN] Ditemukan {len(files)} file data:")
    for i, f in enumerate(files):
        d = load_file(f)
        print(f"  Step {i+1}: {f} ({len(d)} putaran)")

    # Kumpulkan semua data per file
    all_datasets = []
    for f in files:
        d = load_file(f)
        if d:
            all_datasets.append((f, d))

    if not all_datasets:
        print("[ERROR] Semua file kosong atau tidak valid!")
        return None, None

    # Scaler di-fit dari SEMUA data gabungan dulu (biar range konsisten)
    all_combined = []
    for _, d in all_datasets:
        all_combined.extend(d)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array(all_combined).reshape(-1, 1))

    # Build model fresh
    model = build_model()
    es = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)

    # === Step 1 s/d N: Train per file ===
    for i, (fname, dataset) in enumerate(all_datasets):
        if len(dataset) < MIN_DATA:
            print(f"\n  [SKIP] Step {i+1} ({fname}): data terlalu sedikit ({len(dataset)})")
            continue

        print(f"\n  === Step {i+1}/{len(all_datasets)+1}: Belajar dari {fname} ({len(dataset)} data) ===")
        X, y = prepare_xy(dataset, scaler, fit_scaler=False)
        if X is None:
            print(f"  [SKIP] Tidak cukup data untuk membuat sequence")
            continue

        hist = model.fit(X, y, epochs=300, batch_size=16, callbacks=[es], verbose=0)
        final_loss = hist.history['loss'][-1]
        epochs_done = len(hist.history['loss'])
        print(f"  [OK] Selesai ({epochs_done} epochs, loss: {final_loss:.6f})")

    # === Final Step: Train dari SEMUA data gabungan ===
    print(f"\n  === FINAL STEP: Belajar dari SEMUA {len(all_combined)} data gabungan ===")
    if len(all_combined) >= MIN_DATA:
        X_all, y_all = prepare_xy(all_combined, scaler, fit_scaler=False)
        if X_all is not None:
            hist = model.fit(X_all, y_all, epochs=500, batch_size=16, callbacks=[es], verbose=0)
            final_loss = hist.history['loss'][-1]
            epochs_done = len(hist.history['loss'])
            print(f"  [OK] Final training selesai ({epochs_done} epochs, loss: {final_loss:.6f})")
            print(f"  [OK] Model sudah belajar dari total {len(all_combined)} putaran!")

    save_model_and_scaler(model, scaler)
    return model, scaler

# === Cumulative Retrain (untuk input baru) ===
def cumulative_retrain(data):
    """
    Saat ada data baru di aviator_data.json:
    Jalankan progressive train ulang dari semua file.
    """
    print(f"\n[RETRAIN] Data baru ditambahkan. Memulai progressive retrain...")
    return progressive_train(silent=False)

# === Prediksi ===
def predict_next(model, scaler, last_n):
    arr    = np.array(last_n).reshape(-1, 1)
    scaled = scaler.transform(arr).flatten()
    X      = scaled.reshape(1, SEQ_LEN, 1)
    pred_s = model.predict(X, verbose=0)
    pred   = scaler.inverse_transform(pred_s)[0][0]
    return max(1.0, float(pred))

def prediction_label(pred):
    if pred >= 10:
        return ">>> HIGH multiplier (>= 10x) 🚀"
    elif pred >= 5:
        return ">>> MEDIUM multiplier (>= 5x)"
    else:
        return ">>> LOW multiplier (< 5x)"

# === Statistik ===
def show_stats(data):
    files = find_all_data_files()
    all_data = []
    for f in files:
        all_data.extend(load_file(f))

    arr = np.array(all_data) if all_data else np.array(data)
    print(f"\n=== Statistik dari {len(arr)} putaran total ===")
    for f in files:
        d = load_file(f)
        print(f"  {f}: {len(d)} putaran")
    print(f"  Rata-rata    : {arr.mean():.2f}x")
    print(f"  Median       : {np.median(arr):.2f}x")
    print(f"  Min / Max    : {arr.min():.2f}x / {arr.max():.2f}x")
    print(f"  Std Deviasi  : {arr.std():.2f}")
    print(f"  > 2x  : {(arr > 2).sum()} ({(arr > 2).mean()*100:.1f}%)")
    print(f"  > 5x  : {(arr > 5).sum()} ({(arr > 5).mean()*100:.1f}%)")
    print(f"  > 10x : {(arr > 10).sum()} ({(arr > 10).mean()*100:.1f}%)")
    print(f"  > 20x : {(arr > 20).sum()} ({(arr > 20).mean()*100:.1f}%)")
    print(f"  10 data terakhir: {[round(x,2) for x in data[-10:]]}")

# ===== MAIN =====
print("=" * 52)
print("    Aviator AI Predictor  |  MaelCorp")
print("    Progressive Sequential Training")
print("=" * 52)

# Tampilkan file yang ditemukan
files = find_all_data_files()
print(f"\n[SCAN] File data ditemukan: {len(files)}")
for f in files:
    d = load_file(f)
    print(f"  - {f}: {len(d)} putaran")

data = load_data()

# Coba load model yang sudah ada
model, scaler = try_load_model_and_scaler()

# Kalau belum ada model -> jalankan progressive train
if model is None:
    total = sum(len(load_file(f)) for f in files)
    if total >= MIN_DATA:
        print("\n[INFO] Belum ada model. Memulai Progressive Training...")
        model, scaler = progressive_train()
    else:
        print(f"\n[INFO] Belum cukup data (total {total}, butuh {MIN_DATA}). Input data dulu.")

# === Loop Menu ===
while True:
    print("\n--- Menu ------------------------------------------------")
    print(" 1. Input multiplier baru  (+ auto progressive retrain)")
    print(" 2. Prediksi multiplier berikutnya (manual)")
    print(" 3. Lihat statistik semua data")
    print(" 4. Progressive retrain manual dari semua file")
    print(" q. Keluar")
    print("---------------------------------------------------------")
    choice = input("Pilih: ").strip().lower()

    # 1. Input + Progressive Retrain
    if choice == '1':
        inp = input("Masukkan multiplier baru (pisahkan koma): ").strip()
        try:
            values = [float(x.strip()) for x in inp.split(",")]
            if any(v < 1.0 for v in values):
                print("[WARN] Multiplier tidak boleh < 1.0")
                continue

            data.extend(values)
            save_data(data)
            print(f"[OK] {len(values)} data baru disimpan ke {DATA_MAIN}")

            # Hitung total semua data
            files = find_all_data_files()
            total = sum(len(load_file(f)) for f in files)
            print(f"[INFO] Total semua data: {total} putaran dari {len(files)} file")

            if total >= MIN_DATA:
                model, scaler = cumulative_retrain(data)

                if model and len(data) >= SEQ_LEN:
                    last_n = data[-SEQ_LEN:]
                    pred   = predict_next(model, scaler, last_n)
                    print(f"\n  Prediksi putaran berikutnya : {pred:.2f}x")
                    print(f"  {prediction_label(pred)}")
            else:
                print(f"[WARN] Total data masih {total}, butuh {MIN_DATA} untuk training.")

        except Exception as e:
            print(f"[ERROR] {e}")

    # 2. Prediksi Manual
    elif choice == '2':
        if model is None:
            print("[WARN] Model belum tersedia.")
            continue
        print(f"Masukkan {SEQ_LEN} multiplier terakhir (pisahkan koma):")
        inp = input("> ").strip()
        try:
            last_n = [float(x.strip()) for x in inp.split(",")]
            if len(last_n) != SEQ_LEN:
                print(f"[WARN] Harus tepat {SEQ_LEN} angka")
                continue
            pred = predict_next(model, scaler, last_n)
            print(f"\n  Prediksi multiplier berikutnya : {pred:.2f}x")
            print(f"  {prediction_label(pred)}")
        except Exception as e:
            print(f"[ERROR] {e}")

    # 3. Statistik
    elif choice == '3':
        if not data and not find_all_data_files():
            print("[WARN] Belum ada data.")
        else:
            show_stats(data)

    # 4. Retrain Manual
    elif choice == '4':
        files = find_all_data_files()
        total = sum(len(load_file(f)) for f in files)
        if total < MIN_DATA:
            print(f"[WARN] Total data masih {total}, butuh minimal {MIN_DATA}")
        else:
            model, scaler = progressive_train()

    elif choice == 'q':
        print("Bye Maell!")
        break
    else:
        print("[WARN] Pilihan tidak dikenali")