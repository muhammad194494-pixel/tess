"""
Aviator AI Predictor - POWER EDITION | MaelCorp
==================================================
Fitur Power:
- Tidak ada winsorizing, outlier dipertahankan dengan log transform
- Sample weighting: outlier besar diberi bobot lebih
- Data augmentation: memperkaya sample multiplier ekstrem
- Quantile regression (prediksi median 0.5 + CI dari quantile 0.1 & 0.9)
- Hybrid LSTM + Dense dengan dropout kalibrasi
- Evaluasi dengan MAPE (error relatif) lebih adil untuk outlier
- Incremental update dengan prioritasi data terbaru
"""

import json
import os
import glob
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, LayerNormalization,
                                     Attention, GlobalAveragePooling1D, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# ========== KONFIGURASI ==========
DATA_MAIN   = "aviator_data.json"
MODEL_FILE  = "aviator_power_model.keras"
SCALER_FILE = "aviator_power_scaler.pkl"
SEQ_LEN     = 4
N_FEATURES  = 8   # Fitur: log, return, vol5, mean5, momentum, skew7, rsi14, slope5
MIN_DATA    = SEQ_LEN + 20
LOOKAHEAD   = 1
AUGMENT_MULTIPLIER = 2  # Perbanyak data outlier hingga 2x lipat

# ========== PREPROCESSING ==========
def log_transform(x):
    return np.log1p(x)

def inverse_log_transform(x):
    return np.expm1(x)

def compute_rsi(series, window=14):
    delta = np.diff(series, prepend=series[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(window)/window, mode='same')
    avg_loss = np.convolve(loss, np.ones(window)/window, mode='same')
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_power_features(series, seq_len=SEQ_LEN):
    """
    Fitur: 
    0. Log multiplier
    1. Return (perubahan log)
    2. Volatilitas window 5
    3. Rolling mean window 5
    4. Momentum (log - mean5)
    5. Skewness window 7 (kecondongan)
    6. RSI window 14
    7. Slope linear window 5
    """
    series = np.array(series)
    n = len(series)
    log_s = log_transform(series)
    
    # Return
    returns = np.diff(log_s, prepend=log_s[0])
    
    # Volatilitas 5
    vol5 = np.array([np.std(log_s[max(0,i-4):i+1]) for i in range(n)])
    
    # Rolling mean 5
    mean5 = np.convolve(log_s, np.ones(5)/5, mode='same')
    
    # Momentum
    momentum = log_s - mean5
    
    # Skewness window 7 (handle NaN)
    skew7 = np.zeros(n)
    for i in range(7, n+1):
        window = log_s[i-7:i]
        skew7[i-1] = np.mean((window - np.mean(window))**3) / (np.std(window)**3 + 1e-8)
    
    # RSI
    rsi = compute_rsi(series, 14)
    
    # Slope linear window 5
    def local_slope(y, win=5):
        slopes = np.zeros(len(y))
        for i in range(win-1, len(y)):
            y_win = y[i-win+1:i+1]
            x = np.arange(win)
            slope = np.polyfit(x, y_win, 1)[0]
            slopes[i] = slope
        return slopes
    slope5 = local_slope(log_s, 5)
    
    # Stack
    features = np.column_stack([log_s, returns, vol5, mean5, momentum, skew7, rsi, slope5])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features

def create_sequences(features, target_log, seq_len=SEQ_LEN, lookahead=1):
    X, y = [], []
    for i in range(len(target_log) - seq_len - lookahead + 1):
        X.append(features[i:i+seq_len])
        y.append(target_log[i+seq_len+lookahead-1])
    return np.array(X), np.array(y)

def augment_outlier_sequences(X, y, multipliers, threshold=20.0, aug_factor=2):
    """
    Augmentasi data: duplikat sequence yang mengandung multiplier > threshold,
    lalu tambahkan noise kecil pada fitur untuk variasi.
    """
    # Tentukan indeks mana yang targetnya > threshold (dalam multiplier asli)
    target_mult = inverse_log_transform(y)
    outlier_idx = np.where(target_mult > threshold)[0]
    if len(outlier_idx) == 0:
        return X, y
    
    n_aug = min(len(outlier_idx) * (aug_factor - 1), len(X) // 2)
    if n_aug == 0:
        return X, y
    
    chosen = np.random.choice(outlier_idx, n_aug, replace=True)
    X_aug = X[chosen].copy()
    y_aug = y[chosen].copy()
    
    # Tambah noise kecil pada fitur (0.5% - 1% dari std)
    noise = np.random.normal(0, 0.01, X_aug.shape)
    X_aug += noise
    
    # Gabungkan
    X = np.concatenate([X, X_aug], axis=0)
    y = np.concatenate([y, y_aug], axis=0)
    
    # Shuffle agar tidak bias
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]

# ========== MODEL POWER: LSTM + QUANTILE HEADS ==========
def build_power_model(seq_len, n_features):
    """
    Model output median (quantile 0.5) dan juga memberikan interval
    via MC Dropout (tidak perlu quantile heads terpisah, cukup dengan
    training standard loss dan MC Dropout saat inference).
    Namun kita gunakan Huber loss untuk ketahanan outlier.
    """
    inputs = Input(shape=(seq_len, n_features))
    
    # LSTM dengan dropout untuk MC Dropout (training=True nanti)
    x = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
    x = LayerNormalization()(x)
    x = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=output)
    # Huber loss lebih robust terhadap outlier
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
    return model

# ========== DATA LOADING ==========
def load_data():
    try:
        with open(DATA_MAIN, "r") as f:
            return json.load(f)
    except:
        return []

def save_data(data):
    with open(DATA_MAIN, "w") as f:
        json.dump(data, f, indent=2)

def find_all_data_files():
    files = []
    if os.path.exists(DATA_MAIN):
        files.append(DATA_MAIN)
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
            if isinstance(raw, list):
                return [float(x) for x in raw if float(x) >= 1.0]
            elif isinstance(raw, dict):
                for key in ["data", "multipliers", "values", "results"]:
                    if key in raw and isinstance(raw[key], list):
                        return [float(x) for x in raw[key] if float(x) >= 1.0]
        return []
    except:
        return []

# ========== TRAINING POWER ==========
def train_power_model():
    files = find_all_data_files()
    if not files:
        print("[ERROR] Tidak ada file data!")
        return None, None, None, None
    
    print(f"\n[POWER TRAIN] Ditemukan {len(files)} file:")
    all_multipliers = []
    for f in files:
        d = load_file(f)
        if d:
            all_multipliers.extend(d)
            print(f"  - {f}: {len(d)} putaran")
    
    total = len(all_multipliers)
    if total < MIN_DATA:
        print(f"[ERROR] Total data {total} < {MIN_DATA}")
        return None, None, None, None
    
    # Tidak ada winsorizing, langsung log transform
    print(f"[INFO] Total data: {total}, max multiplier: {max(all_multipliers):.2f}")
    
    # Buat fitur
    features = create_power_features(all_multipliers, seq_len=SEQ_LEN)
    target_log = log_transform(all_multipliers)
    X, y = create_sequences(features, target_log, seq_len=SEQ_LEN, lookahead=LOOKAHEAD)
    
    # Augmentasi data outlier (multiplier > 20x)
    X, y = augment_outlier_sequences(X, y, inverse_log_transform(y), threshold=20.0, aug_factor=AUGMENT_MULTIPLIER)
    print(f"[AUGMENT] Total sequences setelah augmentasi: {len(X)}")
    
    # Split time-series (70% train, 15% val, 15% test)
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # Sample weighting: beri bobot lebih pada sample yang target multiplier > 10x
    y_train_mult = inverse_log_transform(y_train)
    weights = np.ones_like(y_train_mult)
    weights[y_train_mult > 10] = 2.0   # bobot dua kali lipat
    weights[y_train_mult > 30] = 4.0
    
    print(f"[SPLIT] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"[WEIGHT] Rata-rata bobot: {np.mean(weights):.2f}")
    
    # Build model
    model = build_power_model(SEQ_LEN, N_FEATURES)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=1)
    ]
    
    print("\n[POWER TRAINING] Memulai training dengan sample weights...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        sample_weight=weights,
        epochs=400,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluasi di test set
    y_pred_log = model.predict(X_test, verbose=0).flatten()
    y_true = inverse_log_transform(y_test)
    y_pred = inverse_log_transform(y_pred_log)
    
    errors_abs = np.abs(y_true - y_pred)
    mae = np.mean(errors_abs)
    medae = np.median(errors_abs)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # Hitung error terpisah untuk outlier >20x
    outlier_mask = y_true > 20
    if np.any(outlier_mask):
        mae_outlier = np.mean(errors_abs[outlier_mask])
        mape_outlier = np.mean(np.abs((y_true[outlier_mask] - y_pred[outlier_mask]) / (y_true[outlier_mask] + 1e-8))) * 100
        print(f"\n[TEST] MAE (all): {mae:.2f}x, MAE (outlier>20x): {mae_outlier:.2f}x")
        print(f"[TEST] MAPE (all): {mape:.1f}%, MAPE (outlier>20x): {mape_outlier:.1f}%")
    else:
        print(f"\n[TEST] MAE: {mae:.2f}x, MAPE: {mape:.1f}%")
    
    # Simpan scaler (hanya parameter untuk prediksi)
    joblib.dump({
        'seq_len': SEQ_LEN,
        'n_features': N_FEATURES,
        'lookahead': LOOKAHEAD,
        'log_used': True
    }, SCALER_FILE)
    model.save(MODEL_FILE)
    print(f"[SAVE] Model disimpan ke {MODEL_FILE}")
    
    return model, all_multipliers, (X_test, y_test), (mae, mape)

# ========== PREDIKSI DENGAN MC DROPOUT BATCH ==========
def predict_power(model, last_multipliers, n_iter=50):
    if len(last_multipliers) < SEQ_LEN:
        return None, None, None
    
    # Buat fitur dari sequence terakhir
    # Kita perlu mempertahankan skala yang sama (tanpa clipping)
    features = create_power_features(last_multipliers, seq_len=SEQ_LEN)
    X_last = features[-SEQ_LEN:].reshape(1, SEQ_LEN, N_FEATURES)
    
    # Batch MC Dropout
    X_batch = np.repeat(X_last, n_iter, axis=0)
    preds_log = model(X_batch, training=True).numpy().flatten()
    pred_log_mean = np.mean(preds_log)
    pred_log_std = np.std(preds_log)
    
    pred_mult = inverse_log_transform(pred_log_mean)
    lower = inverse_log_transform(pred_log_mean - 1.96 * pred_log_std)
    upper = inverse_log_transform(pred_log_mean + 1.96 * pred_log_std)
    
    return max(1.0, pred_mult), max(1.0, lower), max(1.0, upper)

# ========== INCREMENTAL UPDATE (BOBOT DATA BARU) ==========
def incremental_update_power(model, new_multipliers, old_multipliers, epochs=30):
    combined = old_multipliers + new_multipliers
    features = create_power_features(combined, seq_len=SEQ_LEN)
    target_log = log_transform(combined)
    X, y = create_sequences(features, target_log, seq_len=SEQ_LEN, lookahead=LOOKAHEAD)
    
    # Ambil 30% data terakhir untuk fine-tune
    n = len(X)
    last_n = max(50, int(0.3 * n))
    X_inc, y_inc = X[-last_n:], y[-last_n:]
    
    # Bobot untuk data baru: beri prioritas
    weights = np.ones(len(y_inc))
    # Data yang lebih baru (20% paling baru) diberi bobot 2x
    new_last = int(0.2 * len(y_inc))
    if new_last > 0:
        weights[-new_last:] = 2.0
    
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='huber', metrics=['mae'])
    model.fit(X_inc, y_inc, sample_weight=weights, epochs=epochs, batch_size=32, verbose=0)
    print(f"[INC UPDATE] Fine-tune dengan {len(X_inc)} sample (bobot prioritas data terbaru)")
    return model

# ========== STATISTIK ==========
def show_power_stats(all_data):
    import pandas as pd
    arr = np.array(all_data)
    print(f"\n=== STATISTIK POWER ({len(arr)} putaran) ===")
    print(f"  Rata-rata      : {arr.mean():.2f}x")
    print(f"  Median         : {np.median(arr):.2f}x")
    print(f"  Min / Max      : {arr.min():.2f}x / {arr.max():.2f}x")
    print(f"  Std Deviasi    : {arr.std():.2f}")
    print(f"  Skewness       : {pd.Series(arr).skew():.2f}")
    print(f"  > 2x  : {(arr > 2).sum()} ({(arr > 2).mean()*100:.1f}%)")
    print(f"  > 5x  : {(arr > 5).sum()} ({(arr > 5).mean()*100:.1f}%)")
    print(f"  > 10x : {(arr > 10).sum()} ({(arr > 10).mean()*100:.1f}%)")
    print(f"  > 20x : {(arr > 20).sum()} ({(arr > 20).mean()*100:.1f}%)")
    print(f"  10 data terakhir: {[round(x,2) for x in arr[-10:]]}")

# ========== EVALUASI MENGGUNAKAN DATA TERBARU ==========
def evaluate_power(model, all_data):
    if len(all_data) < SEQ_LEN + 20:
        print("[WARN] Data terlalu sedikit untuk evaluasi.")
        return
    features = create_power_features(all_data, seq_len=SEQ_LEN)
    target_log = log_transform(all_data)
    X, y = create_sequences(features, target_log, seq_len=SEQ_LEN, lookahead=LOOKAHEAD)
    # Gunakan 15% terakhir sebagai test
    n = len(X)
    test_size = max(20, int(0.15 * n))
    X_test, y_test = X[-test_size:], y[-test_size:]
    y_pred_log = model.predict(X_test, verbose=0).flatten()
    y_true = inverse_log_transform(y_test)
    y_pred = inverse_log_transform(y_pred_log)
    errors_abs = np.abs(y_true - y_pred)
    mae = np.mean(errors_abs)
    medae = np.median(errors_abs)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    print(f"\n=== EVALUASI POWER ===")
    print(f"MAE   : {mae:.2f}x")
    print(f"MedAE : {medae:.2f}x")
    print(f"MAPE  : {mape:.1f}%")
    # Error untuk outlier >20x
    outlier_mask = y_true > 20
    if np.any(outlier_mask):
        mae_out = np.mean(errors_abs[outlier_mask])
        mape_out = np.mean(np.abs((y_true[outlier_mask] - y_pred[outlier_mask]) / (y_true[outlier_mask] + 1e-8))) * 100
        print(f"MAE (>20x) : {mae_out:.2f}x, MAPE (>20x): {mape_out:.1f}%")

# ========== MAIN ==========
if __name__ == "__main__":
    import pandas as pd
    print("="*60)
    print("   Aviator AI Predictor - POWER EDITION")
    print("   (No Winsorizing + Weighted + Augmentation + MC Dropout)")
    print("="*60)
    
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = load_model(MODEL_FILE)
        params = joblib.load(SCALER_FILE)
        all_data = []
        for f in find_all_data_files():
            all_data.extend(load_file(f))
        print(f"[LOAD] Model POWER ditemukan. Total data: {len(all_data)} putaran")
    else:
        print("[TRAIN] Belum ada model POWER, memulai training...")
        model, all_data, _, _ = train_power_model()
        if model is None:
            exit()
    
    while True:
        print("\n--- MENU POWER -----------------------------------------")
        print(" 1. Input multiplier baru (auto incremental update)")
        print(" 2. Prediksi next multiplier (+ CI dengan MC Dropout)")
        print(" 3. Lihat statistik semua data")
        print(" 4. Retrain full model POWER (dari awal)")
        print(" 5. Evaluasi model (dengan data terbaru)")
        print(" q. Keluar")
        print("---------------------------------------------------------")
        choice = input("Pilih: ").strip().lower()
        
        if choice == '1':
            inp = input("Masukkan multiplier baru (pisahkan koma): ").strip()
            try:
                values = [float(x.strip()) for x in inp.split(",")]
                if any(v < 1.0 for v in values):
                    print("[WARN] Multiplier tidak boleh < 1.0")
                    continue
                current_data = load_data()
                current_data.extend(values)
                save_data(current_data)
                print(f"[OK] {len(values)} data baru ditambahkan.")
                
                old_data = []
                for f in find_all_data_files():
                    old_data.extend(load_file(f))
                model = incremental_update_power(model, values, old_data, epochs=20)
                model.save(MODEL_FILE)
                print("[OK] Model diupdate secara incremental.")
            except Exception as e:
                print(f"[ERROR] {e}")
        
        elif choice == '2':
            print(f"Masukkan {SEQ_LEN} multiplier terakhir (pisahkan koma):")
            inp = input("> ").strip()
            try:
                last_n = [float(x.strip()) for x in inp.split(",")]
                if len(last_n) != SEQ_LEN:
                    print(f"[WARN] Harus tepat {SEQ_LEN} angka")
                    continue
                pred, lower, upper = predict_power(model, last_n, n_iter=50)
                if pred:
                    print(f"\n  Prediksi multiplier berikutnya: {pred:.2f}x")
                    print(f"  Interval kepercayaan 95%   : [{lower:.2f}, {upper:.2f}]")
                    if pred >= 10:
                        print("  >>> HIGH RISK (>=10x) 🚀")
                    elif pred >= 5:
                        print("  >>> MEDIUM RISK (>=5x)")
                    else:
                        print("  >>> LOW RISK (<5x)")
            except Exception as e:
                print(f"[ERROR] {e}")
        
        elif choice == '3':
            all_data = []
            for f in find_all_data_files():
                all_data.extend(load_file(f))
            if all_data:
                show_power_stats(all_data)
            else:
                print("[WARN] Tidak ada data.")
        
        elif choice == '4':
            print("[RETRAIN] Melakukan training POWER dari awal...")
            model, all_data, _, _ = train_power_model()
            if model:
                print("[OK] Retrain selesai.")
        
        elif choice == '5':
            all_data = []
            for f in find_all_data_files():
                all_data.extend(load_file(f))
            if len(all_data) >= MIN_DATA:
                evaluate_power(model, all_data)
            else:
                print(f"[WARN] Data terlalu sedikit (butuh {MIN_DATA})")
        
        elif choice == 'q':
            print("Terima kasih! MaelCorp POWER siap membantu prediksi ekstrem. 🚀")
            break
        else:
            print("[WARN] Pilihan tidak valid.")
