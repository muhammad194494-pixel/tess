"""
Aviator AI Predictor - Advanced Edition | MaelCorp
====================================================
Fitur:
- Log transform & winsorizing untuk outlier ekstrem
- Feature engineering: rolling mean/std, return, momentum
- Arsitektur Bidirectional LSTM + Attention
- Validation split untuk menghindari overfitting
- MC Dropout untuk confidence interval
- Incremental learning (update dengan data baru)
- Auto save best model berdasarkan val loss
"""

import json
import os
import glob
import numpy as np
import joblib
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, Dropout, 
                                     Input, Attention, Concatenate, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# === Konfigurasi ===
DATA_MAIN   = "aviator_data.json"
MODEL_FILE  = "aviator_model_advanced.keras"
SCALER_FILE = "aviator_scaler_advanced.pkl"
SEQ_LEN     = 4                    # Lebih panjang dari sebelumnya
N_FEATURES  = 5                     # Jumlah fitur engineered
MIN_DATA    = SEQ_LEN + 20
LOOKBACK    = SEQ_LEN

# === Transformasi Outlier ===
def winsorize_series(data, limits=(0.01, 0.99)):
    """Winsorize untuk batasi outlier ekstrem"""
    arr = np.array(data)
    low = np.percentile(arr, limits[0]*100)
    high = np.percentile(arr, limits[1]*100)
    return np.clip(arr, low, high)

def log_transform(x):
    return np.log1p(x)  # log(1+x)

def inverse_log_transform(x):
    return np.expm1(x)

# === Feature Engineering ===
def create_features(series, seq_len=SEQ_LEN):
    """
    Membuat fitur dari raw multiplier:
    1. Raw multiplier (log transformed)
    2. Return % dari step sebelumnya
    3. Rolling mean (window 3)
    4. Rolling std (window 3)
    5. Momentum (diff dari mean)
    """
    series = np.array(series)
    n = len(series)
    
    # Log transform untuk stabilisasi varians
    log_series = log_transform(series)
    
    # Return (persentase perubahan log)
    returns = np.diff(log_series, prepend=log_series[0])
    
    # Rolling features
    roll_mean = np.convolve(log_series, np.ones(3)/3, mode='same')
    roll_std = np.array([np.std(log_series[max(0,i-2):i+1]) for i in range(n)])
    # Momentum: (log_series - roll_mean)
    momentum = log_series - roll_mean
    
    # Stack features: [log_price, return, roll_mean, roll_std, momentum]
    features = np.column_stack([log_series, returns, roll_mean, roll_std, momentum])
    return features

def create_sequences(features, target_log, seq_len=SEQ_LEN):
    """Buat X (sequences) dan y (target log)"""
    X, y = [], []
    for i in range(len(target_log) - seq_len):
        X.append(features[i:i+seq_len])
        y.append(target_log[i+seq_len])
    return np.array(X), np.array(y)

# === Model dengan Attention ===
def build_attention_model(seq_len, n_features):
    """
    Model: Bidirectional LSTM + Self-Attention + Dense layers
    """
    inputs = Input(shape=(seq_len, n_features))
    
    # Bidirectional LSTM layer 1
    lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputs)
    lstm1 = Dropout(0.3)(lstm1)
    
    # Bidirectional LSTM layer 2
    lstm2 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(lstm1)
    lstm2 = Dropout(0.3)(lstm2)
    
    # Self-Attention sederhana
    attention = Dense(1, activation='tanh')(lstm2)
    attention = Flatten()(attention)
    attention_weights = Dense(seq_len, activation='softmax')(attention)
    attention_weights = tf.keras.layers.Reshape((seq_len, 1))(attention_weights)
    context = tf.keras.layers.Multiply()([lstm2, attention_weights])
    context = tf.keras.layers.GlobalAveragePooling1D()(context)
    
    # Dense layers
    dense1 = Dense(32, activation='relu')(context)
    dense1 = Dropout(0.2)(dense1)
    dense2 = Dense(16, activation='relu')(dense1)
    output = Dense(1)(dense2)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Untuk menghindari import tf di atas, kita gunakan:
import tensorflow as tf

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

# === Training dengan Validation Split ===
def train_model(model, X_train, y_train, X_val, y_val, epochs=300, patience=30):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=0)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    best_loss = min(history.history['val_loss'])
    return model, best_loss

# === Progressive Training dengan Validation ===
def progressive_train_advanced():
    files = find_all_data_files()
    if not files:
        print("[ERROR] Tidak ada file data!")
        return None, None
    
    print(f"\n[PROGRESSIVE TRAIN] Ditemukan {len(files)} file:")
    all_multipliers = []
    for f in files:
        d = load_file(f)
        if d:
            all_multipliers.extend(d)
            print(f"  - {f}: {len(d)} putaran")
    
    if len(all_multipliers) < MIN_DATA:
        print(f"[ERROR] Total data {len(all_multipliers)} < {MIN_DATA} (minimal)")
        return None, None
    
    # Winsorize extreme outliers (batasi di percentile 99)
    multipliers_wins = winsorize_series(all_multipliers, limits=(0.01, 0.99))
    
    # Buat fitur & sequence
    features = create_features(multipliers_wins, seq_len=SEQ_LEN)
    target_log = log_transform(multipliers_wins)
    
    X, y = create_sequences(features, target_log, seq_len=SEQ_LEN)
    
    # Split time-series (train: 80% awal, val: 20% akhir)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\n[SPLIT] Train: {len(X_train)} sequences | Val: {len(X_val)} sequences")
    
    # Build model baru
    model = build_attention_model(SEQ_LEN, N_FEATURES)
    model, best_val_loss = train_model(model, X_train, y_train, X_val, y_val, epochs=300, patience=40)
    
    print(f"[TRAINING] Best val loss: {best_val_loss:.6f}")
    
    # Simpan model dan scaler (sebenarnya kita tidak perlu scaler karena kita pakai log & winsorize)
    # Tapi kita simpan parameter winsorize dan feature transform untuk konsistensi prediksi
    joblib.dump({
        'limits': (0.01, 0.99),
        'log_used': True,
        'seq_len': SEQ_LEN
    }, SCALER_FILE)
    model.save(MODEL_FILE)
    print(f"[SAVE] Model disimpan ke {MODEL_FILE}")
    return model, all_multipliers

# === Incremental Update (tanpa retrain full) ===
def incremental_update(model, new_multipliers, old_multipliers, epochs=50):
    """
    Update model dengan data baru tanpa melupakan data lama.
    Menggunakan weighted training: data baru lebih berbobot.
    """
    if len(new_multipliers) < 5:
        print("[INFO] Data baru terlalu sedikit, skip incremental update")
        return model
    
    all_data = old_multipliers + new_multipliers
    # Winsorize ulang dengan data gabungan
    all_wins = winsorize_series(all_data, limits=(0.01, 0.99))
    features = create_features(all_wins, seq_len=SEQ_LEN)
    target_log = log_transform(all_wins)
    X, y = create_sequences(features, target_log, seq_len=SEQ_LEN)
    
    # Beri bobot lebih pada sequence yang mengandung data baru
    # Cari index di mana sequence mengandung setidaknya 1 data dari new_multipliers
    # (sederhana: semua sequence terakhir saja yang diutamakan)
    n_new = len(new_multipliers)
    weights = np.ones(len(X))
    # Sequence yang mengandung data baru (misal 20% terakhir)
    weights[-min(50, len(X)//5):] = 3.0
    
    # Split train/val (val dari data lama)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    w_train = weights[:split_idx]
    
    # Fine-tune model dengan learning rate lebih kecil
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), 
              sample_weight=w_train, epochs=epochs, batch_size=32, 
              callbacks=callbacks, verbose=0)
    print(f"[INCREMENTAL] Model updated dengan {len(new_multipliers)} data baru")
    return model

# === Prediksi dengan Confidence Interval (MC Dropout) ===
def predict_with_ci(model, last_multipliers, n_iter=50):
    """
    Menggunakan Monte Carlo Dropout dengan BATCH prediction.
    Jauh lebih cepat karena 50 forward pass dilakukan sekaligus (paralel).
    """
    if len(last_multipliers) < SEQ_LEN:
        print(f"[ERROR] Butuh {SEQ_LEN} data terakhir")
        return None, None, None
    
    # Preprocess last_multipliers (clip ke 1-50 untuk stabilitas)
    last_clipped = np.clip(last_multipliers, 1, 50)
    
    # Buat fitur untuk sequence terakhir
    temp_series = np.array(last_clipped)
    features_last = create_features(temp_series, seq_len=SEQ_LEN)
    X_last = features_last[-SEQ_LEN:].reshape(1, SEQ_LEN, N_FEATURES)
    
    # Buat batch dengan n_iter salinan
    X_batch = np.repeat(X_last, n_iter, axis=0)
    
    # Jalankan prediksi dalam satu batch dengan training=True (aktifkan dropout)
    preds_log = model(X_batch, training=True).numpy().flatten()
    
    # Hitung statistik
    pred_log_mean = np.mean(preds_log)
    pred_log_std = np.std(preds_log)
    
    # Inverse log transform
    pred_mult = inverse_log_transform(pred_log_mean)
    lower = inverse_log_transform(pred_log_mean - 1.96 * pred_log_std)
    upper = inverse_log_transform(pred_log_mean + 1.96 * pred_log_std)
    
    return max(1.0, pred_mult), max(1.0, lower), max(1.0, upper)
# === Statistik ===
def show_advanced_stats(all_data):
    arr = np.array(all_data)
    print(f"\n=== STATISTIK ADVANCED ({len(arr)} putaran) ===")
    print(f"  Rata-rata      : {arr.mean():.2f}x")
    print(f"  Median         : {np.median(arr):.2f}x")
    print(f"  Min / Max      : {arr.min():.2f}x / {arr.max():.2f}x")
    print(f"  Std Deviasi    : {arr.std():.2f}")
    print(f"  Skewness       : {float(pd.Series(arr).skew()):.2f}")  # butuh pandas, optional
    print(f"  > 2x  : {(arr > 2).sum()} ({(arr > 2).mean()*100:.1f}%)")
    print(f"  > 5x  : {(arr > 5).sum()} ({(arr > 5).mean()*100:.1f}%)")
    print(f"  > 10x : {(arr > 10).sum()} ({(arr > 10).mean()*100:.1f}%)")
    print(f"  > 20x : {(arr > 20).sum()} ({(arr > 20).mean()*100:.1f}%)")
    print(f"  10 data terakhir: {[round(x,2) for x in arr[-10:]]}")

# === Main ===
if __name__ == "__main__":
    import pandas as pd  # untuk skewness, optional
    
    print("="*60)
    print("   Aviator AI Predictor - ADVANCED EDITION")
    print("   (Bidirectional LSTM + Attention + MC Dropout)")
    print("="*60)
    
    # Cek model yang sudah ada
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        print("[LOAD] Model existing ditemukan.")
        model = load_model(MODEL_FILE)
        params = joblib.load(SCALER_FILE)
        all_data = []
        for f in find_all_data_files():
            all_data.extend(load_file(f))
        print(f"[LOAD] Total data: {len(all_data)} putaran")
    else:
        print("[TRAIN] Belum ada model, memulai progressive training...")
        model, all_data = progressive_train_advanced()
        if model is None:
            exit()
    
    # Menu interaktif
    while True:
        print("\n--- MENU ADVANCED --------------------------------------")
        print(" 1. Input multiplier baru (auto incremental update)")
        print(" 2. Prediksi next multiplier (+ Confidence Interval)")
        print(" 3. Lihat statistik semua data")
        print(" 4. Retrain full progressive (dari awal)")
        print(" 5. Evaluasi model (backtest)")
        print(" q. Keluar")
        print("--------------------------------------------------------")
        choice = input("Pilih: ").strip().lower()
        
        if choice == '1':
            inp = input("Masukkan multiplier baru (pisahkan koma): ").strip()
            try:
                values = [float(x.strip()) for x in inp.split(",")]
                if any(v < 1.0 for v in values):
                    print("[WARN] Multiplier tidak boleh < 1.0")
                    continue
                # Load data existing
                current_data = load_data()
                current_data.extend(values)
                save_data(current_data)
                print(f"[OK] {len(values)} data baru ditambahkan.")
                
                # Lakukan incremental update
                old_data = []
                for f in find_all_data_files():
                    old_data.extend(load_file(f))
                model = incremental_update(model, values, old_data, epochs=30)
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
                pred, lower, upper = predict_with_ci(model, last_n)
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
                show_advanced_stats(all_data)
            else:
                print("[WARN] Tidak ada data.")
        
        elif choice == '4':
            print("[RETRAIN] Melakukan progressive training dari awal...")
            model, all_data = progressive_train_advanced()
            if model:
                print("[OK] Retrain selesai.")
        
        elif choice == '5':
            # Backtest sederhana: prediksi setiap step pada data terakhir 20%
            all_data = []
            for f in find_all_data_files():
                all_data.extend(load_file(f))
            if len(all_data) < SEQ_LEN + 20:
                print("[WARN] Data terlalu sedikit untuk evaluasi.")
                continue
            test_size = min(50, len(all_data)//5)
            test_data = all_data[-test_size:]
            errors = []
            for i in range(len(test_data) - SEQ_LEN):
                window = test_data[i:i+SEQ_LEN]
                actual = test_data[i+SEQ_LEN]
                pred, _, _ = predict_with_ci(model, window)
                if pred:
                    errors.append(abs(pred - actual))
            if errors:
                mae = np.mean(errors)
                print(f"\n[EVALUASI] MAE pada {len(errors)} prediksi: {mae:.2f}x")
                print("  (Semakin kecil semakin akurat)")
            else:
                print("[WARN] Tidak cukup sequence untuk evaluasi.")
        
        elif choice == 'q':
            print("Terima kasih! MaelCorp Advanced AI siap membantu.")
            break
        else:
            print("[WARN] Pilihan tidak valid.")
