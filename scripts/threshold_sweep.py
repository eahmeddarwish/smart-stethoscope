# NOTE: standalone hardware bring-up / R&D script, kept for reference.
# It predates the src/smart_stethoscope/ package and still uses the
# original hardcoded paths (~/Desktop/Smart_Stethoscope) -- adjust the
# SETTINGS/paths at the top before running it on your own machine.
#!/usr/bin/env python3
# =========================================
# Smart Stethoscope — Threshold Sweep v9
#
# بيجمع كل الـ probabilities الأول
# ثم يجرب كل threshold ويطبع النتايج
# =========================================

import os
import re
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from scipy.signal import butter, filtfilt, iirnotch

warnings.filterwarnings('ignore')

# ── TFLite ───────────────────────────────────────────────
try:
    from ai_edge_litert.interpreter import Interpreter
    print("✅ ai_edge_litert loaded")
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
        print("✅ tflite_runtime loaded")
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("✅ tensorflow TFLite loaded")

# =========================================
# ⚙️  PATHS
# =========================================
BASE_DIR     = '/home/pi/Desktop/Smart_Stethoscope'
MODEL_PATH   = os.path.join(BASE_DIR, 'models/heart_model_v9_int8.tflite')
ENCODER_PATH = os.path.join(BASE_DIR, 'models/heart_label_encoder_v9.pkl')
CONFIG_PATH  = os.path.join(BASE_DIR, 'models/heart_config_v9.pkl')
DATASET_DIR  = os.path.join(BASE_DIR, 'datasets/Heartbeats')

# =========================================
# LOAD CONFIG + MODEL + ENCODER
# =========================================
print("\n📂 Loading model files...")

for p in [MODEL_PATH, ENCODER_PATH]:
    if not os.path.exists(p):
        print(f"❌ Not found: {p}")
        sys.exit(1)

with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'rb') as f:
        config = pickle.load(f)
    SAMPLE_RATE      = config['sample_rate']
    N_MELS           = config['n_mels']
    N_FFT            = config['n_fft']
    HOP_LENGTH       = config['hop_length']
    DURATION         = config['duration']
    OLD_THRESHOLD    = config['murmur_threshold']
    MURMUR_IDX       = config['murmur_idx']
    print(f"✅ Config loaded — old threshold was: {OLD_THRESHOLD}")
else:
    SAMPLE_RATE   = 2000
    N_MELS        = 64
    N_FFT         = 512
    HOP_LENGTH    = 128
    DURATION      = 3
    OLD_THRESHOLD = 0.25
    MURMUR_IDX    = list(le.classes_).index('murmur')
    print("⚠️  No config file — using v9 hardcoded defaults")

SAMPLES = SAMPLE_RATE * DURATION
CLASSES = list(le.classes_)

print(f"   Classes     : {CLASSES}")
print(f"   MURMUR_IDX  : {MURMUR_IDX} → '{CLASSES[MURMUR_IDX]}'")
print(f"   Sample rate : {SAMPLE_RATE} Hz")

interp = Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
in_det  = interp.get_input_details()
out_det = interp.get_output_details()
print(f"✅ TFLite loaded — input: {in_det[0]['shape']}")

# =========================================
# FEATURE EXTRACTION
# (نفس الـ notebook v10 بالضبط)
# =========================================
def bandpass_filter(audio):
    nyq  = SAMPLE_RATE / 2
    b, a = butter(4, [20/nyq, 1900/nyq], btype='band')
    return filtfilt(b, a, audio)

def notch_filter(audio):
    nyq  = SAMPLE_RATE / 2
    b, a = iirnotch(50/nyq, 30)
    return filtfilt(b, a, audio)

def extract_features(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    except:
        return None
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]
    try:
        audio = bandpass_filter(audio)
        audio = notch_filter(audio)
    except:
        pass
    mx = np.max(np.abs(audio))
    if mx > 0:
        audio = audio / mx
    mel    = librosa.feature.melspectrogram(
                y=audio, sr=SAMPLE_RATE,
                n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db[..., np.newaxis].astype(np.float32)

# =========================================
# LOAD DATASET من الـ CSV
# (نفس منطق notebook v10 — timestamp matching)
# =========================================
def load_dataset():
    LABEL_MAP = {
        'artifact'  : 'artifact',
        'murmur'    : 'murmur',
        'normal'    : 'normal',
        'extrahls'  : 'extrahls',
        'extrastole': 'extrahls',
    }
    disk_index = {}
    for folder in ['set_a', 'set_b']:
        fpath = os.path.join(DATASET_DIR, folder)
        if not os.path.exists(fpath):
            continue
        for fname in os.listdir(fpath):
            if not fname.endswith('.wav'):
                continue
            ts = re.findall(r'\d{10,}', fname)
            if ts:
                rest = fname.split(ts[0])[-1].replace('.wav', '')
                disk_index[f'{ts[0]}{rest}'] = os.path.join(fpath, fname)

    records = []
    for csv_name, source in [('set_a.csv', 'set_a'), ('set_b.csv', 'set_b')]:
        csv_path = os.path.join(DATASET_DIR, csv_name)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            raw_label   = str(row['label']).strip().lower()
            clean_label = LABEL_MAP.get(raw_label)
            if not clean_label:
                continue
            fname = str(row['fname'])
            ts    = re.findall(r'\d{10,}', fname)
            if ts:
                rest     = fname.split(ts[0])[-1].replace('.wav', '')
                wav_path = disk_index.get(f'{ts[0]}{rest}')
            else:
                basename = fname.split('/')[-1]
                folder   = fname.split('/')[0] if '/' in fname else source
                wav_path = os.path.join(DATASET_DIR, folder, basename)
                if not os.path.exists(wav_path):
                    wav_path = None

            if wav_path and os.path.exists(wav_path):
                records.append({'path': wav_path, 'label': clean_label})

    df_out = pd.DataFrame(records).drop_duplicates(subset='path')
    return df_out

print("\n" + "="*60)
print("📂 LOADING DATASET...")
print("="*60)
df = load_dataset()
print(f"✅ {len(df)} samples found")
print(df['label'].value_counts().to_string())

# =========================================
# COLLECT ALL PROBABILITIES (مرة واحدة بس)
# =========================================
print("\n" + "="*60)
print("🔬 RUNNING INFERENCE (collecting probabilities)...")
print("="*60)

y_true_labels = []   # ['murmur', 'normal', ...]
all_probs     = []   # (N, 4) — كل الـ softmax outputs

errors = 0
for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
    feat = extract_features(row['path'])
    if feat is None:
        errors += 1
        continue
    inp = feat[np.newaxis, ...]
    interp.set_tensor(in_det[0]['index'], inp)
    interp.invoke()
    probs = interp.get_tensor(out_det[0]['index'])[0].copy()  # (4,)
    all_probs.append(probs)
    y_true_labels.append(row['label'])

all_probs = np.array(all_probs)        # (N, 4)
y_true_labels = np.array(y_true_labels)

print(f"\n✅ Collected: {len(all_probs)} samples")
if errors:
    print(f"⚠️  Skipped : {errors} files")

# =========================================
# THRESHOLD SWEEP
# =========================================
def evaluate_threshold(threshold):
    """بيرجع dict بكل الـ metrics لـ threshold معين"""
    y_pred = []
    for probs in all_probs:
        if probs[MURMUR_IDX] >= threshold:
            y_pred.append('murmur')
        else:
            y_pred.append(CLASSES[np.argmax(probs)])
    y_pred = np.array(y_pred)

    correct  = np.sum(y_true_labels == y_pred)
    accuracy = correct / len(y_true_labels)

    # Murmur-specific
    murmur_mask = (y_true_labels == 'murmur')
    tp = int(np.sum((y_true_labels == 'murmur') & (y_pred == 'murmur')))
    fn = int(np.sum((y_true_labels == 'murmur') & (y_pred != 'murmur')))
    fp = int(np.sum((y_true_labels != 'murmur') & (y_pred == 'murmur')))
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1        = 2*recall*precision / (recall+precision) if (recall+precision) > 0 else 0

    # Normal recall (مهم ما يتأثرش كتير)
    tn_n = int(np.sum((y_true_labels == 'normal') & (y_pred == 'normal')))
    tot_n = int(np.sum(y_true_labels == 'normal'))
    normal_recall = tn_n / tot_n if tot_n > 0 else 0

    return {
        'accuracy'      : accuracy,
        'murmur_recall' : recall,
        'murmur_prec'   : precision,
        'murmur_f1'     : f1,
        'normal_recall' : normal_recall,
        'tp': tp, 'fn': fn, 'fp': fp
    }

print("\n" + "="*60)
print("📊 THRESHOLD SWEEP RESULTS")
print("="*60)
print(f"\n{'Thresh':>7} | {'Mur.Rec':>8} | {'Mur.Prec':>9} | {'Acc':>7} | {'Norm.Rec':>9} | {'FP':>5} | {'FN':>5}")
print("-" * 70)

thresholds  = np.arange(0.20, 0.71, 0.05)
results     = {}
best_thresh = OLD_THRESHOLD
best_score  = -1

for th in thresholds:
    m = evaluate_threshold(th)
    results[th] = m

    # ✅ Clinical rule: murmur recall ≥ 85% أولاً، ثم أعلى accuracy
    is_safe   = m['murmur_recall'] >= 0.85
    score     = m['accuracy'] if is_safe else -1
    marker    = " ◄ BEST" if (is_safe and score > best_score) else ""

    if is_safe and score > best_score:
        best_score  = score
        best_thresh = th

    safety_icon = "✅" if is_safe else "❌"
    print(f"  {th:.2f}   | {m['murmur_recall']*100:>7.1f}% | "
          f"{m['murmur_prec']*100:>8.1f}% | {m['accuracy']*100:>6.1f}% | "
          f"{m['normal_recall']*100:>8.1f}% | {m['fp']:>5} | {m['fn']:>5}  "
          f"{safety_icon}{marker}")

# =========================================
# RECOMMENDATION
# =========================================
print("\n" + "="*60)
print("💡 RECOMMENDATION")
print("="*60)

bm = results[best_thresh]
print(f"\n   Old threshold : {OLD_THRESHOLD:.2f}")
print(f"   New threshold : {best_thresh:.2f}  ← USE THIS")
print(f"\n   With threshold = {best_thresh:.2f}:")
print(f"   ✅ Murmur Recall    : {bm['murmur_recall']*100:.1f}%  (target ≥ 85%)")
print(f"   ✅ Murmur Precision : {bm['murmur_prec']*100:.1f}%")
print(f"   ✅ Overall Accuracy : {bm['accuracy']*100:.1f}%")
print(f"   ✅ Normal Recall    : {bm['normal_recall']*100:.1f}%")
print(f"      False Positives  : {bm['fp']}  (normal→murmur)")
print(f"      False Negatives  : {bm['fn']}  (missed murmurs)")

# =========================================
# UPDATE CONFIG FILE بالـ threshold الجديد
# =========================================
if os.path.exists(CONFIG_PATH):
    print(f"\n🔄 Updating heart_config_v9.pkl with new threshold...")
    with open(CONFIG_PATH, 'rb') as f:
        config = pickle.load(f)
    config['murmur_threshold'] = float(best_thresh)
    with open(CONFIG_PATH, 'wb') as f:
        pickle.dump(config, f)
    print(f"✅ Config updated: murmur_threshold = {best_thresh:.2f}")
else:
    print(f"\n⚠️  Config file not found — update threshold manually in GUI:")
    print(f"   MURMUR_THRESHOLD = {best_thresh:.2f}")

print("\n" + "="*60)
print("✅ Sweep complete!")
print("="*60)