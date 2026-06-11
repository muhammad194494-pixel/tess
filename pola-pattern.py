import json
import glob
import numpy as np
import os

# ─── Konfigurasi ────────────────────────────────────
DATA_PATTERN = "aviator_data*.json"
SEQ_LEN      = 4    # panjang pola yang dicari
TOP_K        = 10   # ambil 10 pola paling mirip

# ─── Load semua data ─────────────────────────────────
def load_all_data():
    files = sorted(glob.glob(DATA_PATTERN))
    all_data = []
    for f in files:
        try:
            with open(f) as file:
                data = json.load(file)
                if isinstance(data, list) and len(data) > 0:
                    all_data.extend(data)
        except:
            print(f"⚠️  Gagal baca {f}")
    return all_data

# ─── Build database pola dari semua data ─────────────
# Setiap "pola" = sequence 4 angka + angka berikutnya
def build_pattern_db(data, seq_len):
    patterns = []
    for i in range(len(data) - seq_len):
        seq  = data[i : i + seq_len]
        next_val = data[i + seq_len]
        patterns.append((seq, next_val))
    return patterns

# ─── Hitung kemiripan pakai Euclidean distance ────────
# Normalize dulu biar 1.93 vs 19.3 tidak terlalu jauh
def similarity_score(seq_a, seq_b):
    a = np.array(seq_a, dtype=float)
    b = np.array(seq_b, dtype=float)

    # Normalize ke rasio (bagi dengan nilai pertama)
    # Ini fokus ke SHAPE/TREN, bukan nilai absolut
    a_norm = a / a[0] if a[0] != 0 else a
    b_norm = b / b[0] if b[0] != 0 else b

    # Euclidean distance → makin kecil makin mirip
    dist = np.sqrt(np.sum((a_norm - b_norm) ** 2))
    return dist

# ─── Cari TOP_K pola paling mirip ────────────────────
def find_similar_patterns(input_seq, pattern_db, top_k):
    scored = []
    for seq, next_val in pattern_db:
        dist = similarity_score(input_seq, seq)
        scored.append((dist, seq, next_val))

    # Sort by distance (ascending = paling mirip duluan)
    scored.sort(key=lambda x: x[0])
    return scored[:top_k]

# ─── Analisis hasil ───────────────────────────────────
def analyze_results(matches):
    next_vals = [m[2] for m in matches]
    dists     = [m[0] for m in matches]
    seqs      = [m[1] for m in matches]

    avg     = np.mean(next_vals)
    median  = np.median(next_vals)
    minimum = np.min(next_vals)
    maximum = np.max(next_vals)

    # Weighted average (pola lebih mirip = bobot lebih besar)
    weights = 1 / (np.array(dists) + 1e-6)
    weighted_avg = np.average(next_vals, weights=weights)

    # Probabilitas per range
    thresholds = [1.5, 2.0, 3.0, 5.0, 10.0]
    probs = {}
    for t in thresholds:
        probs[t] = np.mean(np.array(next_vals) >= t) * 100

    # Kategori dominan
    cats = {'< 2x': 0, '2x-5x': 0, '5x-10x': 0, '> 10x': 0}
    for v in next_vals:
        if v < 2:     cats['< 2x']    += 1
        elif v < 5:   cats['2x-5x']   += 1
        elif v < 10:  cats['5x-10x']  += 1
        else:         cats['> 10x']   += 1

    dominant = max(cats, key=cats.get)

    return {
        "next_vals"    : next_vals,
        "seqs"         : seqs,
        "dists"        : dists,
        "avg"          : avg,
        "median"       : median,
        "min"          : minimum,
        "max"          : maximum,
        "weighted_avg" : weighted_avg,
        "probs"        : probs,
        "cats"         : cats,
        "dominant"     : dominant,
    }

# ─── Display ─────────────────────────────────────────
def display(input_seq, result, top_k):
    print("\n" + "═"*50)
    print(f"  🔍 INPUT   : {input_seq}")
    print("─"*50)
    print(f"  ✅ Ditemukan {top_k} pola paling mirip:\n")

    for i, (dist, seq, nxt) in enumerate(
        zip(result["dists"], result["seqs"], result["next_vals"])
    ):
        bar = "█" * int(min(nxt, 20) / 1)
        similarity = max(0, 100 - dist * 30)
        print(f"  [{i+1:2d}] {seq}  →  {nxt:6.2f}x   sim={similarity:.0f}%")

    print("─"*50)
    print(f"  🎯 Prediksi (weighted) : {result['weighted_avg']:.2f}x")
    print(f"  📊 Median              : {result['median']:.2f}x")
    print(f"  📉 Range               : {result['min']:.2f}x  ~  {result['max']:.2f}x")
    print("─"*50)
    print("  📈 PROBABILITAS dari pola serupa:")
    for t, p in result["probs"].items():
        filled  = int(p / 5)
        empty   = 20 - filled
        bar     = "█" * filled + "░" * empty
        print(f"   ≥{t:5.1f}x  [{bar}]  {p:.0f}%")
    print("─"*50)
    print("  📦 Distribusi kategori:")
    total = sum(result["cats"].values())
    for cat, count in result["cats"].items():
        pct = count / total * 100
        bar = "█" * count
        print(f"   {cat:8s}  {bar:<12}  {count}/{total}  ({pct:.0f}%)")
    print("─"*50)

    dom = result["dominant"]
    if dom == "< 2x":
        signal = "🔴 CRASH CEPET"
    elif dom == "2x-5x":
        signal = "🟡 SEDANG"
    elif dom == "5x-10x":
        signal = "🟢 LUMAYAN TINGGI"
    else:
        signal = "🚀 POTENSI JACKPOT"

    print(f"  🏁 Sinyal dominan : {signal}")
    print("═"*50)

# ─── Main ─────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════╗")
    print("║   AVIATOR PATTERN MATCHER  v1.0              ║")
    print("║   Cari pola mirip dari data historis         ║")
    print("╚══════════════════════════════════════════════╝\n")

    data = load_all_data()
    if not data:
        print("❌ Tidak ada data. Taruh aviator_data*.json di folder ini.")
        return

    print(f"📂 Data loaded: {len(data)} putaran")
    pattern_db = build_pattern_db(data, SEQ_LEN)
    print(f"🗂️  Database pola: {len(pattern_db)} sequence\n")

    while True:
        print(f"Masukkan {SEQ_LEN} multiplier terakhir (pisah koma)")
        print(f"Contoh: 1.93, 2.93, 2.28, 1.94")
        inp = input(">> ").strip()

        if inp.lower() in ('q', 'quit', 'exit'):
            print("👋 Sampai jumpa!")
            break

        try:
            input_seq = [float(x.strip()) for x in inp.split(",")]
            if len(input_seq) != SEQ_LEN:
                print(f"⚠️  Harus tepat {SEQ_LEN} angka\n")
                continue
            if any(v < 1.0 for v in input_seq):
                print("⚠️  Nilai minimal 1.0\n")
                continue

            matches = find_similar_patterns(input_seq, pattern_db, TOP_K)
            result  = analyze_results(matches)
            display(input_seq, result, TOP_K)

        except ValueError:
            print("❌ Input tidak valid\n")

if __name__ == "__main__":
    main()
