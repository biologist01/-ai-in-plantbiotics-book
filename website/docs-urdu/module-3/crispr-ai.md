---
sidebar_position: 3
---

# AI کے ساتھ CRISPR ٹارگٹ کی پیش گوئی

## تعارف

مصنوعی ذہانت CRISPR جین ایڈیٹنگ کو بہتر بنا رہی ہے، گائیڈ RNA کی کارکردگی کی پیش گوئی، آف ٹارگٹ اثرات کا پتہ لگانے، اور فصلوں کی بہتری کے لیے بہترین ایڈیٹنگ حکمت عملی ڈیزائن کرنے میں 🧬✂️۔

## CRISPR-Cas9 کی بنیادیں

CRISPR-Cas9 ایک جین ایڈیٹنگ ٹول ہے:
- **گائیڈ RNA (sgRNA)**: ٹارگٹ سائٹ تلاش کرتا ہے
- **Cas9**: DNA کاٹتا ہے
- **PAM**: ضروری موٹف (NGG)

## sgRNA کارکردگی کی پیش گوئی

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def encode_sgrna(sequence):
    """30bp sgRNA + PAM کو انکوڈ کریں"""
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 
               'G': [0,0,1,0], 'T': [0,0,0,1]}
    
    encoded = []
    for nuc in sequence:
        encoded.append(mapping.get(nuc, [0,0,0,0]))
    
    return np.array(encoded)

def build_sgrna_predictor():
    model = models.Sequential([
        layers.Conv1D(64, 5, activation='relu', input_shape=(30, 4)),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # کارکردگی سکور
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ماڈل ٹرین کریں
model = build_sgrna_predictor()
```

## آف ٹارگٹ پیش گوئی

```python
def calculate_mismatch_score(sgrna, off_target):
    """مسمیچ سکور حساب کریں"""
    score = 0
    weights = [1.0] * 20  # پوزیشن ویٹس
    
    # بیج ریجن (PAM کے قریب) زیادہ اہم
    weights[-12:] = [1.5] * 12
    
    for i, (s, o) in enumerate(zip(sgrna, off_target)):
        if s != o:
            score += weights[i]
    
    return score

def predict_off_targets(sgrna, genome_sequences, threshold=3):
    """ممکنہ آف ٹارگٹس تلاش کریں"""
    off_targets = []
    
    for seq_name, sequence in genome_sequences.items():
        for i in range(len(sequence) - 23):
            candidate = sequence[i:i+20]
            pam = sequence[i+20:i+23]
            
            if pam in ['NGG', 'NAG']:  # PAM چیک
                score = calculate_mismatch_score(sgrna[:20], candidate)
                
                if score <= threshold:
                    off_targets.append({
                        'location': f"{seq_name}:{i}",
                        'sequence': candidate,
                        'mismatch_score': score
                    })
    
    return off_targets
```

## ڈیپ لرننگ سے آف ٹارگٹ

```python
def build_off_target_predictor():
    # دو سیکوینسز کا موازنہ
    sgrna_input = layers.Input(shape=(23, 4), name='sgrna')
    target_input = layers.Input(shape=(23, 4), name='target')
    
    # دونوں کے لیے ایک جیسا انکوڈر
    encoder = models.Sequential([
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.GlobalMaxPooling1D()
    ])
    
    sgrna_encoded = encoder(sgrna_input)
    target_encoded = encoder(target_input)
    
    # فرق حساب کریں
    diff = layers.Subtract()([sgrna_encoded, target_encoded])
    concat = layers.Concatenate()([sgrna_encoded, target_encoded, diff])
    
    # پیش گوئی
    x = layers.Dense(128, activation='relu')(concat)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model([sgrna_input, target_input], output)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    return model
```

## فصلوں کی بہتری کی مثال

```python
# خشک سالی مزاحمت کے لیے CRISPR ڈیزائن

target_genes = {
    'DREB1': 'ATGGTCGATCGATCGATCGAGG',  # خشک سالی ردعمل
    'NAC': 'GCTAGCTAGCTAGCTAGCTAGG',    # تناؤ رواداری
    'LEA': 'TGCATGCATGCATGCATGCAGG'     # پانی کی کمی
}

def design_crispr_edit(gene_name, target_sequence):
    """بہترین sgRNA ڈیزائن کریں"""
    
    # PAM سائٹس تلاش کریں
    pam_sites = []
    for i in range(len(target_sequence) - 23):
        if target_sequence[i+21:i+23] == 'GG':
            sgrna = target_sequence[i:i+20]
            pam_sites.append({
                'position': i,
                'sgrna': sgrna,
                'pam': target_sequence[i+20:i+23]
            })
    
    # ہر سائٹ کو سکور کریں
    for site in pam_sites:
        site['efficiency'] = predict_efficiency(site['sgrna'])
        site['off_targets'] = count_off_targets(site['sgrna'])
    
    # بہترین سائٹ منتخب کریں
    best = max(pam_sites, 
               key=lambda x: x['efficiency'] - x['off_targets'] * 0.1)
    
    return best

# استعمال
for gene, seq in target_genes.items():
    result = design_crispr_edit(gene, seq)
    print(f"جین: {gene}")
    print(f"بہترین sgRNA: {result['sgrna']}")
    print(f"کارکردگی: {result['efficiency']:.2f}")
```

## خلاصہ

| ٹاسک | ML طریقہ |
|------|----------|
| کارکردگی پیش گوئی | CNN/LSTM |
| آف ٹارگٹ پیش گوئی | سیامیز نیٹ ورک |
| ایڈیٹ ڈیزائن | رینفورسمنٹ لرننگ |

## اگلے اقدامات

- [جینومک سلیکشن](/docs/module-3/genomic-selection) - بریڈنگ
