---
sidebar_position: 4
---

# جینومک سلیکشن اور بریڈنگ

## تعارف

جینومک سلیکشن فصلوں کی بریڈنگ پروگراموں کو تیز کر رہی ہے۔ ML سے بریڈنگ ویلیوز کی پیش گوئی اور بہترین کراسز ڈیزائن کریں 🌾۔

## روایتی بمقابلہ جینومک سلیکشن

| طریقہ | وقت | درستگی |
|-------|-----|--------|
| روایتی | 10-15 سال | کم |
| جینومک | 3-5 سال | زیادہ |

## جینومک پیش گوئی ماڈلز

### GBLUP (Genomic BLUP)

```python
import numpy as np
from scipy.linalg import inv

def gblup(X, y):
    """
    X: جینوٹائپ میٹرکس (n x p)
    y: فینوٹائپ ویکٹر
    """
    n, p = X.shape
    
    # جینومک ریلیشن شپ میٹرکس
    W = X - np.mean(X, axis=0)
    G = W @ W.T / p
    
    # BLUP حل
    h2 = 0.5  # ہیریٹیبیلٹی
    lamb = (1 - h2) / h2
    
    V = G + lamb * np.eye(n)
    V_inv = inv(V)
    
    # بریڈنگ ویلیوز
    gebv = G @ V_inv @ y
    
    return gebv
```

### Ridge Regression (rrBLUP)

```python
from sklearn.linear_model import Ridge

def rrblup(X, y, alpha=1.0):
    """
    rrBLUP ماڈل
    """
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    
    # مارکر اثرات
    marker_effects = model.coef_
    
    # GEBV حساب کریں
    gebv = X @ marker_effects
    
    return gebv, marker_effects
```

## ڈیپ لرننگ سے خصوصیت پیش گوئی

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_genomic_predictor(n_markers):
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(n_markers,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# استعمال
X_train, X_test, y_train, y_test = train_test_split(genotypes, phenotypes)

model = build_genomic_predictor(n_markers=X_train.shape[1])
model.fit(X_train, y_train, epochs=100, batch_size=32,
          validation_data=(X_test, y_test))
```

## ملٹی ٹریٹ ماڈل

```python
def build_multi_trait_model(n_markers, n_traits):
    input_layer = layers.Input(shape=(n_markers,))
    
    # مشترکہ پرتیں
    x = layers.Dense(512, activation='relu')(input_layer)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    
    # ہر خصوصیت کے لیے الگ برانچ
    outputs = []
    for i in range(n_traits):
        branch = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(1, name=f'trait_{i}')(branch)
        outputs.append(output)
    
    model = models.Model(input_layer, outputs)
    model.compile(optimizer='adam', 
                  loss='mse',
                  loss_weights=[1.0] * n_traits)
    
    return model

# استعمال
model = build_multi_trait_model(n_markers=10000, n_traits=3)
# خصوصیات: پیداوار، اونچائی، پختگی
```

## بہترین کراس سلیکشن

```python
import numpy as np
from itertools import combinations

def select_optimal_crosses(parents_gebv, n_crosses=10):
    """
    بہترین والدین کے جوڑے منتخب کریں
    """
    n_parents = len(parents_gebv)
    crosses = []
    
    for i, j in combinations(range(n_parents), 2):
        # اوسط GEBV
        mean_gebv = (parents_gebv[i] + parents_gebv[j]) / 2
        
        # جینیاتی تنوع (فرض کریں)
        diversity = abs(parents_gebv[i] - parents_gebv[j])
        
        # مشترکہ سکور
        score = mean_gebv + 0.1 * diversity
        
        crosses.append({
            'parent1': i,
            'parent2': j,
            'expected_gebv': mean_gebv,
            'score': score
        })
    
    # سب سے زیادہ سکور والے کراسز
    crosses.sort(key=lambda x: x['score'], reverse=True)
    
    return crosses[:n_crosses]

# استعمال
parents = np.array([2.5, 3.1, 1.8, 2.9, 3.5])
best_crosses = select_optimal_crosses(parents)
for cross in best_crosses[:3]:
    print(f"والدین {cross['parent1']} x {cross['parent2']}: "
          f"متوقع GEBV = {cross['expected_gebv']:.2f}")
```

## کراس ویلیڈیشن

```python
from sklearn.model_selection import KFold

def cross_validate_genomic_model(X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True)
    
    correlations = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # ماڈل ٹرین کریں
        gebv, _ = rrblup(X_train, y_train)
        
        # ٹیسٹ پر پیش گوئی
        _, marker_effects = rrblup(X_train, y_train)
        y_pred = X_test @ marker_effects
        
        # ارتباط
        corr = np.corrcoef(y_test, y_pred)[0, 1]
        correlations.append(corr)
    
    return np.mean(correlations), np.std(correlations)

# استعمال
acc, std = cross_validate_genomic_model(genotypes, phenotypes)
print(f"پیش گوئی درستگی: {acc:.3f} ± {std:.3f}")
```

## خلاصہ

- جینومک سلیکشن بریڈنگ کو تیز کرتی ہے
- GBLUP اور rrBLUP بنیادی ماڈلز ہیں
- ڈیپ لرننگ پیچیدہ خصوصیات کے لیے

## اگلے اقدامات

- [جینومکس پروجیکٹ](/docs/module-3/genomics-project)
