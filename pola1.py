import json
import glob
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ─── Konfigurasi ────────────────────────────────────────────
DATA_PATTERN  = "aviator_data*.json"
MODEL_FILE    = "aviator_model.keras"
SCALER_FILE   = "aviator_scaler.pkl"
SEQ_LEN       = 4        # lebih panjang = lebih banyak konteks
TEST_SPLIT    = 0.15     # 15% data buat evaluasi
EPOCHS        = 300
BATCH_SIZE    = 16

# ─── Load semua file JSON ────────────────────────────────────
def load_all_data():
    files = sorted(glob.glob(DATA_PATTERN))
    if not files:
        print("⚠️  Tidak ada file aviator_data*.json ditemukan!")
        return []
    all_data = []
    for f in files:
        try:
            with open(f) as file:
                data = json.load(file)
                if isinstance(data, list) and len(data) > 0:
                    all_data.extend(data)
        except Exception as e:
            print(f"⚠️  Gagal baca {f}: {e}")
    return all_data

# ─── Preprocessing dengan log transform ─────────────────────
# Kenapa log? Data aviator sangat skewed (1x ~ 100x+),
# log transform bikin distribusinya lebih normal → LSTM lebih mudah belajar
def preprocess(data):
    arr = np.array(data, dtype=np.float32).reshape(-1, 1)
    arr = np.log(arr)  # log transform
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(arr).flatten()
    return scaled, scaler

def inverse_transform(scaler, scaled_val):
    # Balik: unscale → exp (kebalikan log)
    val = scaler.inverse_transform([[scaled_val]])[0][0]
    return float(np.exp(val))

# ─── Buat sequence X, y ──────────────────────────────────────
def make_sequences(scaled, seq_len):
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len])
    return np.array(X).reshape(-1, seq_len, 1), np.array(y)

# ─── Arsitektur model ────────────────────────────────────────
def build_model(seq_len):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, 1)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # output sudah dinormalisasi 0-1
    ])
    model.compile(optimizer=Adam(0.001), loss='huber')  # huber lebih robust dari mse
    return model

# ─── Training ────────────────────────────────────────────────
def train_model(data):
    print(f"🔧 Preprocessing {len(data)} data poin...")
    scaled, scaler = preprocess(data)

    # Split train/test
    split = int(len(scaled) * (1 - TEST_SPLIT))
    train_scaled = scaled[:split]
    test_scaled  = scaled[split - SEQ_LEN:]  # overlap SEQ_LEN buat bisa buat sequence

    X_train, y_train = make_sequences(train_scaled, SEQ_LEN)
    X_test,  y_test  = make_sequences(test_scaled,  SEQ_LEN)

    print(f"📊 Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    model = build_model(SEQ_LEN)

    callbacks = [
        EarlyStopping(patience=30, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=15, min_lr=1e-5)
    ]

    print("🚀 Training... (sabar brok)\n")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluasi di test set
    preds_scaled = model.predict(X_test, verbose=0).flatten()
    preds_real = np.array([inverse_transform(scaler, v) for v in preds_scaled])
    truth_real = np.array([inverse_transform(scaler, v) for v in y_test])

    mae = mean_absolute_error(truth_real, preds_real)
    direction_acc = np.mean((preds_real > 2.0) == (truth_real > 2.0)) * 100

    print(f"\n📈 Evaluasi Model:")
    print(f"   MAE (rata-rata selisih): {mae:.3f}x")
    print(f"   Akurasi arah (>2x):      {direction_acc:.1f}%")
    print(f"   ⚠️  Ingat: Aviator pakai RNG, akurasi 100% tidak mungkin!")

    # Simpan model + scaler
    model.save(MODEL_FILE)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n✅ Model disimpan: {MODEL_FILE}")
    print(f"✅ Scaler disimpan: {SCALER_FILE}")
    return model, scaler

# ─── Load model + scaler ─────────────────────────────────────
def load_saved():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        print("🔄 Memuat model yang sudah ada...")
        model = load_model(MODEL_FILE)
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None

# ─── Prediksi + analisis probabilitas ────────────────────────
def predict_next(model, scaler, last_n):
    arr = np.array(last_n, dtype=np.float32).reshape(-1, 1)
    log_arr = np.log(arr)
    scaled = scaler.transform(log_arr).flatten()
    X = scaled.reshape(1, SEQ_LEN, 1)

    # Prediksi utama
    pred_scaled = model.predict(X, verbose=0)[0][0]
    pred_real   = inverse_transform(scaler, pred_scaled)

    # Monte Carlo dropout untuk estimasi uncertainty
    # (jalankan prediksi 50x dengan dropout aktif → ambil distribusinya)
    mc_preds = []
    for _ in range(50):
        p = model(X, training=True).numpy()[0][0]
        mc_preds.append(inverse_transform(scaler, p))
    mc_preds = np.array(mc_preds)

    low  = np.percentile(mc_preds, 10)
    high = np.percentile(mc_preds, 90)
    conf = max(0, 100 - (high - low) / pred_real * 100)

    # Probabilitas berdasarkan data historis (global)
    # → Ini lebih jujur daripada berpura-pura model bisa prediksi exact
    thresholds = [1.5, 2.0, 3.0, 5.0, 10.0]
    all_data   = load_all_data()
    probs = {}
    for t in thresholds:
        probs[t] = np.mean(np.array(all_data) >= t) * 100

    return {
        "pred"    : pred_real,
        "low"     : low,
        "high"    : high,
        "conf"    : conf,
        "probs"   : probs,
        "mc_std"  : np.std(mc_preds)
    }

# ─── Display hasil prediksi ───────────────────────────────────
def display_result(result):
    pred = result["pred"]
    low  = result["low"]
    high = result["high"]
    conf = result["conf"]

    print("\n" + "═"*45)
    print(f"  🎯 PREDIKSI  :  {pred:.2f}x")
    print(f"  📉 Range     :  {low:.2f}x  ~  {high:.2f}x")
    print(f"  🔮 Konfiden  :  {conf:.0f}%")
    print("─"*45)
    print("  📊 PROBABILITAS HISTORIS:")
    for t, p in result["probs"].items():
        bar = "█" * int(p / 5)
        print(f"   ≥{t:5.1f}x  {p:5.1f}%  {bar}")
    print("─"*45)
    if pred < 1.5:
        level = "🔴 RENDAH"
    elif pred < 3.0:
        level = "🟡 SEDANG"
    elif pred < 10.0:
        level = "🟢 TINGGI"
    else:
        level = "🚀 JACKPOT"
    print(f"  Sinyal: {level}")
    print("═"*45)

# ─── Main ─────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════╗")
    print("║     AVIATOR AI PREDICTOR  v3.0           ║")
    print("║     Powered by LSTM + Monte Carlo        ║")
    print("╚══════════════════════════════════════════╝\n")

    data = load_all_data()
    print(f"📂 Total data: {len(data)} putaran\n")

    if len(data) < SEQ_LEN + 10:
        print(f"⚠️  Data minimal {SEQ_LEN + 10} putaran.")
        return

    # Cek ada model tersimpan atau tidak
    model, scaler = load_saved()

    if model is None:
        print("📭 Belum ada model. Mulai training...\n")
        model, scaler = train_model(data)
    else:
        choice = input("Model ditemukan. Retrain? (y/n) [default: n]: ").strip().lower()
        if choice == 'y':
            model, scaler = train_model(data)

    print("\n✅ Model siap. Mulai prediksi!\n")

    while True:
        print(f"Masukkan {SEQ_LEN} multiplier terakhir (pisah koma)")
        print("Contoh: 1.23, 4.56, 2.10, 1.01, 3.40, 1.88, 7.22, 2.55")
        inp = input(">> ").strip()

        if inp.lower() in ('q', 'quit', 'exit'):
            print("👋 Sampai jumpa!")
            break

        try:
            last_n = [float(x.strip()) for x in inp.split(",")]
            if len(last_n) != SEQ_LEN:
                print(f"⚠️  Harus tepat {SEQ_LEN} angka\n")
                continue
            if any(v < 1.0 for v in last_n):
                print("⚠️  Multiplier minimal 1.0\n")
                continue

            result = predict_next(model, scaler, last_n)
            display_result(result)

        except ValueError:
            print("❌ Input tidak valid. Gunakan angka desimal.\n")

if __name__ == "__main__":
    main()
