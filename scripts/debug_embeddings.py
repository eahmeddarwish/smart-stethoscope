# NOTE: standalone hardware bring-up / R&D script, kept for reference.
# It predates the src/smart_stethoscope/ package and still uses the
# original hardcoded paths (~/Desktop/Smart_Stethoscope) -- adjust the
# SETTINGS/paths at the top before running it on your own machine.
#!/usr/bin/env python3
# =========================================
# Smart Stethoscope — Inference Test
# بدون tensorflow — embeddings جاهزة
# Kuwait College of Science & Technology
# =========================================

import os
import pickle
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# =========================================
# SETTINGS
# =========================================
CONF_THRESHOLD = 0.60

# =========================================
# PATHS
# =========================================
BASE_DIR      = '/home/pi/Desktop/Smart_Stethoscope'
MODEL_PATH    = os.path.join(BASE_DIR, 'models/vggish_classifier.tflite')
ENCODER_PATH  = os.path.join(BASE_DIR, 'models/heart_label_encoder_vggish.pkl')
EMBEDDINGS    = os.path.join(BASE_DIR, 'models/embeddings_124.pkl')

# =========================================
# LOAD
# =========================================
print('\n' + '='*50)
print('  Smart Stethoscope — Inference Test')
print('='*50)

print('\n📦 Loading classifier...')
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
id_ = interpreter.get_input_details()
od_ = interpreter.get_output_details()
print('   ✅ Done')

print('\n📦 Loading label encoder...')
with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)
CLASSES = [str(c) for c in le.classes_]
print(f'   ✅ Classes: {CLASSES}')

print('\n📦 Loading embeddings...')
with open(EMBEDDINGS, 'rb') as f:
    data = pickle.load(f)
print(f'   ✅ {len(data)} samples loaded')

# =========================================
# PREDICT
# =========================================
def predict(embedding):
    inp = embedding[np.newaxis, ...].astype(np.float32)
    interpreter.set_tensor(id_[0]['index'], inp)
    interpreter.invoke()
    probs    = interpreter.get_tensor(od_[0]['index'])[0]
    idx      = np.argmax(probs)
    conf     = float(probs[idx])
    pred_raw = CLASSES[idx]
    pred     = pred_raw if conf >= CONF_THRESHOLD else 'uncertain'
    return pred, conf, pred_raw

# =========================================
# RUN INFERENCE
# =========================================
print(f'\n🔍 Running inference on {len(data)} samples...')
print(f'   Threshold: {CONF_THRESHOLD*100:.0f}%')
print('─' * 50)

results = []
for item in data:
    pred, conf, pred_raw = predict(item['embedding'])
    results.append({
        'true': item['label'],
        'pred': pred,
        'raw' : pred_raw,
        'conf': conf
    })

# =========================================
# RESULTS
# =========================================
import numpy as np

total         = len(results)
overall_acc   = sum(r['true'] == r['raw']  for r in results) / total * 100
certain       = [r for r in results if r['pred'] != 'uncertain']
certain_acc   = sum(r['true'] == r['pred'] for r in certain) / len(certain) * 100 if certain else 0
uncertain_pct = (total - len(certain)) / total * 100

print('\n' + '='*50)
print('  RESULTS')
print('='*50)
print(f'  Total samples    : {total}')
print(f'  Overall accuracy : {overall_acc:.1f}%')
print(f'  Certain (≥{CONF_THRESHOLD*100:.0f}%)   : {len(certain)} ({100-uncertain_pct:.1f}%)')
print(f'  Certain accuracy : {certain_acc:.1f}%')
print(f'  Uncertain        : {uncertain_pct:.1f}%')
print('='*50)

print('\n  Per-class breakdown:')
print(f'  {"Class":12s} {"Total":>6} {"Correct":>8} {"Acc":>8}')
print('  ' + '-'*38)
for cls in CLASSES:
    sub     = [r for r in results if r['true'] == cls]
    correct = sum(r['raw'] == cls for r in sub)
    total_c = len(sub)
    acc     = correct/total_c*100 if total_c > 0 else 0
    mark    = '✅' if acc >= 70 else '⚠️ ' if acc >= 50 else '❌'
    print(f'  {mark} {cls:10s} {total_c:>6} {correct:>8} {acc:>7.1f}%')

covered = sorted(set(r['pred'] for r in certain))
print(f'\n  Classes covered  : {covered}')
print(f'\n✅ Done!')
