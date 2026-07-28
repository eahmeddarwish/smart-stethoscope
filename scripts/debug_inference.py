# NOTE: standalone hardware bring-up / R&D script, kept for reference.
# It predates the src/smart_stethoscope/ package and still uses the
# original hardcoded paths (~/Desktop/Smart_Stethoscope) -- adjust the
# SETTINGS/paths at the top before running it on your own machine.
#!/usr/bin/env python3
# =========================================
# Smart Stethoscope — Full Inference
# WAV → VGGish → Classifier → Result
# =========================================

import os
import pickle
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

# =========================================
# SETTINGS
# =========================================
VGGISH_SR      = 16000
DURATION       = 3
SAMPLES        = VGGISH_SR * DURATION
CONF_THRESHOLD = 0.60

# =========================================
# PATHS
# =========================================
BASE_DIR        = '/home/pi/Desktop/Smart_Stethoscope'
VGGISH_PATH     = os.path.join(BASE_DIR, 'vggish_model')
CLASSIFIER_PATH = os.path.join(BASE_DIR, 'models/vggish_classifier.tflite')
ENCODER_PATH    = os.path.join(BASE_DIR, 'models/heart_label_encoder_vggish.pkl')

# =========================================
# LOAD MODELS
# =========================================
print('\n' + '='*50)
print('  Smart Stethoscope — Full Inference')
print('='*50)

print('\n📦 Loading VGGish...')
vggish = tf.saved_model.load(VGGISH_PATH)
print('   ✅ VGGish ready')

print('\n📦 Loading classifier...')
interpreter = Interpreter(model_path=CLASSIFIER_PATH)
interpreter.allocate_tensors()
id_ = interpreter.get_input_details()
od_ = interpreter.get_output_details()
print('   ✅ Classifier ready')

print('\n📦 Loading label encoder...')
with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)
CLASSES = [str(c) for c in le.classes_]
print(f'   ✅ Classes: {CLASSES}')

# =========================================
# FEATURE EXTRACTION
# =========================================
def extract_embedding(file_path):
    """WAV → VGGish embedding (128-dim)"""
    try:
        audio, _ = librosa.load(file_path, sr=VGGISH_SR)
    except Exception as e:
        print(f'   ❌ Error loading: {e}')
        return None

    # Pad or trim to 3 seconds
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]

    # Normalize
    mx = np.max(np.abs(audio))
    if mx > 0:
        audio = audio / mx

    audio = audio.astype(np.float32)

    # VGGish → mean of windows
    embeddings = vggish(audio)
    return np.mean(embeddings.numpy(), axis=0)  # (128,)

# =========================================
# PREDICT FUNCTION
# =========================================
def predict(file_path):
    """WAV file → diagnosis"""
    print(f'\n🎵 Processing: {os.path.basename(file_path)}')

    # Extract embedding
    embedding = extract_embedding(file_path)
    if embedding is None:
        return None, 0.0, None

    # Classify
    inp = embedding[np.newaxis, ...].astype(np.float32)
    interpreter.set_tensor(id_[0]['index'], inp)
    interpreter.invoke()
    probs    = interpreter.get_tensor(od_[0]['index'])[0]
    idx      = np.argmax(probs)
    conf     = float(probs[idx])
    pred_raw = CLASSES[idx]
    pred     = pred_raw if conf >= CONF_THRESHOLD else 'uncertain'

    # Print result
    print(f'\n   {"="*40}')
    print(f'   DIAGNOSIS: {pred.upper()}')
    print(f'   Confidence: {conf*100:.1f}%')
    print(f'   {"="*40}')
    print(f'\n   All probabilities:')
    for i, cls in enumerate(CLASSES):
        bar  = '█' * int(probs[i] * 20)
        mark = '◄' if i == idx else ''
        print(f'   {cls:12s}: {probs[i]*100:5.1f}%  {bar} {mark}')

    return pred, conf, pred_raw

# =========================================
# DIAGNOSIS INFO
# =========================================
DIAGNOSIS_INFO = {
    'normal'  : '✅ Normal Heart Sound — No abnormalities detected',
    'murmur'  : '🔴 Heart Murmur — Abnormal turbulent flow detected',
    'extrahls': '🟡 Extra Heart Sounds (S3/S4) — Additional sounds detected',
    'artifact': '⚠️  Recording Artifact — Poor quality recording',
    'uncertain': '❓ Uncertain — Confidence below threshold, please re-record'
}

# =========================================
# MAIN — TEST ON REFERENCE AUDIO
# =========================================
if __name__ == '__main__':
    import sys

    # لو في argument، استخدمه كملف
    if len(sys.argv) > 1:
        wav_file = sys.argv[1]
        if os.path.exists(wav_file):
            pred, conf, raw = predict(wav_file)
            if pred:
                print(f'\n   {DIAGNOSIS_INFO.get(pred, pred)}')
        else:
            print(f'❌ File not found: {wav_file}')

    # لو مفيش argument، نشغّل على الـ reference audio
    else:
        REF_DIR = os.path.join(BASE_DIR, 'SmartStethoscope/reference_audio')
        ref_files = [f for f in os.listdir(REF_DIR) if f.endswith('.wav')]

        if not ref_files:
            print(f'\n⚠️  No WAV files in {REF_DIR}')
            print('Usage: python test_full_inference.py <path_to_wav>')
        else:
            print(f'\n🔍 Testing on {len(ref_files)} reference files...')
            for fname in sorted(ref_files):
                path = os.path.join(REF_DIR, fname)
                pred, conf, raw = predict(path)
                if pred:
                    print(f'   {DIAGNOSIS_INFO.get(pred, pred)}')

    print('\n✅ Done!')