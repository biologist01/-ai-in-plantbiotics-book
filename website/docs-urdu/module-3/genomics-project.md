---
sidebar_position: 6
---

# پروجیکٹ: خصوصیت پیش گوئی سسٹم

## منصوبے کا جائزہ

اس پروجیکٹ میں ہم ایک مکمل جینومک پیش گوئی پائپ لائن بنائیں گے جو SNP ڈیٹا سے پودوں کی خصوصیات پیش گوئی کرے 🧬۔

## مقاصد

- SNP ڈیٹا پروسیس کرنا
- جینومک فیچرز نکالنا
- پیش گوئی ماڈل بنانا
- ماڈل کی تشخیص

## ڈیٹا سیٹ تیاری

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# مصنوعی جینومک ڈیٹا
np.random.seed(42)

n_samples = 500
n_markers = 5000

# جینوٹائپ میٹرکس (0, 1, 2)
genotypes = np.random.randint(0, 3, size=(n_samples, n_markers))

# QTL اثرات
n_qtl = 50
qtl_positions = np.random.choice(n_markers, n_qtl, replace=False)
qtl_effects = np.random.normal(0, 1, n_qtl)

# فینوٹائپ
genetic_values = genotypes[:, qtl_positions] @ qtl_effects
noise = np.random.normal(0, 2, n_samples)
phenotypes = genetic_values + noise

print(f"جینوٹائپ شیپ: {genotypes.shape}")
print(f"فینوٹائپ رینج: {phenotypes.min():.2f} - {phenotypes.max():.2f}")
```

## ڈیٹا پری پروسیسنگ

```python
def preprocess_genotypes(X):
    """
    جینوٹائپ پری پروسیسنگ
    """
    # مسنگ ویلیوز امپیوٹ کریں
    X = np.where(np.isnan(X), 1, X)
    
    # MAF فلٹر
    maf = np.mean(X, axis=0) / 2
    maf = np.minimum(maf, 1 - maf)
    keep = maf > 0.05
    X = X[:, keep]
    
    # سینٹر اور سکیل
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X = np.nan_to_num(X, 0)
    
    return X

# پروسیس کریں
X_processed = preprocess_genotypes(genotypes.astype(float))
print(f"پروسیسڈ شیپ: {X_processed.shape}")

# ٹرین/ٹیسٹ تقسیم
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, phenotypes, test_size=0.2, random_state=42
)
```

## ماڈل 1: Ridge Regression

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def train_ridge_model(X_train, y_train, X_test, y_test):
    # بہترین الفا تلاش کریں
    alphas = [0.1, 1, 10, 100, 1000]
    best_alpha = None
    best_score = -np.inf
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        if score > best_score:
            best_score = score
            best_alpha = alpha
    
    # فائنل ماڈل
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_train, y_train)
    
    return final_model, best_alpha

model_ridge, alpha = train_ridge_model(X_train, y_train, X_test, y_test)
y_pred_ridge = model_ridge.predict(X_test)

print(f"بہترین الفا: {alpha}")
print(f"R²: {r2_score(y_test, y_pred_ridge):.3f}")
print(f"ارتباط: {np.corrcoef(y_test, y_pred_ridge)[0,1]:.3f}")
```

## ماڈل 2: نیورل نیٹورک

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_genomic_nn(input_dim):
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ماڈل بنائیں
model_nn = create_genomic_nn(X_train.shape[1])
model_nn.summary()

# کال بیکس
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
]

# ٹرین کریں
history = model_nn.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
```

## ماڈل 3: Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingRegressor

# PCA سے فیچرز کم کریں
from sklearn.decomposition import PCA

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Gradient Boosting
model_gb = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model_gb.fit(X_train_pca, y_train)
y_pred_gb = model_gb.predict(X_test_pca)

print(f"Gradient Boosting R²: {r2_score(y_test, y_pred_gb):.3f}")
```

## ماڈلز کا موازنہ

```python
import matplotlib.pyplot as plt

def compare_models(y_test, predictions_dict):
    fig, axes = plt.subplots(1, len(predictions_dict), figsize=(15, 5))
    
    for idx, (name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        
        corr = np.corrcoef(y_test, y_pred)[0, 1]
        r2 = r2_score(y_test, y_pred)
        
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('حقیقی قدر')
        ax.set_ylabel('پیش گوئی')
        ax.set_title(f'{name}\nr={corr:.3f}, R²={r2:.3f}')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()

# Neural Network پیش گوئی
y_pred_nn = model_nn.predict(X_test).flatten()

# موازنہ
predictions = {
    'Ridge': y_pred_ridge,
    'Neural Network': y_pred_nn,
    'Gradient Boosting': y_pred_gb
}

compare_models(y_test, predictions)
```

## فیچر اہمیت

```python
def get_top_markers(model, n_top=20):
    """
    سب سے اہم مارکرز نکالیں
    """
    coefficients = np.abs(model.coef_)
    top_indices = np.argsort(coefficients)[-n_top:]
    
    return top_indices, coefficients[top_indices]

# Ridge ماڈل سے
top_markers, importances = get_top_markers(model_ridge)

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_markers)), importances)
plt.xlabel('اہمیت')
plt.ylabel('مارکر انڈیکس')
plt.title('سب سے اہم جینیٹک مارکرز')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
```

## کراس ویلیڈیشن

```python
from sklearn.model_selection import KFold

def k_fold_validation(X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train_fold = X[train_idx]
        X_test_fold = X[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]
        
        # Ridge ماڈل
        model = Ridge(alpha=100)
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        
        corr = np.corrcoef(y_test_fold, y_pred)[0, 1]
        r2 = r2_score(y_test_fold, y_pred)
        
        results.append({'fold': fold + 1, 'correlation': corr, 'r2': r2})
        print(f"Fold {fold + 1}: r = {corr:.3f}, R² = {r2:.3f}")
    
    results_df = pd.DataFrame(results)
    print(f"\nاوسط: r = {results_df['correlation'].mean():.3f} ± "
          f"{results_df['correlation'].std():.3f}")
    
    return results_df

results = k_fold_validation(X_processed, phenotypes)
```

## مکمل پائپ لائن

```python
class GenomicPredictionPipeline:
    def __init__(self):
        self.pca = None
        self.model = None
        
    def fit(self, X, y):
        # PCA
        self.pca = PCA(n_components=min(100, X.shape[1]))
        X_pca = self.pca.fit_transform(X)
        
        # ماڈل
        self.model = Ridge(alpha=100)
        self.model.fit(X_pca, y)
        
        return self
    
    def predict(self, X):
        X_pca = self.pca.transform(X)
        return self.model.predict(X_pca)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            'correlation': np.corrcoef(y, y_pred)[0, 1],
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred)
        }

# استعمال
pipeline = GenomicPredictionPipeline()
pipeline.fit(X_train, y_train)

train_metrics = pipeline.evaluate(X_train, y_train)
test_metrics = pipeline.evaluate(X_test, y_test)

print("ٹریننگ نتائج:", train_metrics)
print("ٹیسٹ نتائج:", test_metrics)
```

## خلاصہ

اس پروجیکٹ میں ہم نے سیکھا:
- جینومک ڈیٹا پروسیسنگ
- مختلف ML ماڈلز کا موازنہ
- کراس ویلیڈیشن تکنیک
- مکمل پائپ لائن ڈیزائن

## اگلا ماڈیول

[ماڈیول 4: IoT تعارف](/docs/module-4/iot-intro)
