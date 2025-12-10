---
sidebar_position: 3
---

# پیداوار پیش گوئی ماڈلز

## تعارف

فصل کی پیداوار پیش گوئی کسانوں کو بہتر فیصلے کرنے میں مدد دیتی ہے۔ ML ماڈلز سینسر ڈیٹا، موسم، اور تاریخی ڈیٹا سے پیداوار کا اندازہ لگاتے ہیں 🌾📈۔

## ڈیٹا کی اقسام

| ڈیٹا | ذریعہ | اہمیت |
|------|-------|-------|
| موسمی | ویدر سٹیشن | بہت زیادہ |
| مٹی | سینسرز | زیادہ |
| سیٹلائٹ | NDVI | زیادہ |
| تاریخی | ریکارڈز | درمیانی |

## ڈیٹا سیٹ تیاری

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# مصنوعی فصل ڈیٹا
np.random.seed(42)
n_samples = 500

data = {
    'temperature_avg': np.random.uniform(15, 35, n_samples),
    'rainfall': np.random.uniform(0, 300, n_samples),
    'humidity_avg': np.random.uniform(40, 90, n_samples),
    'soil_ph': np.random.uniform(5.5, 8.0, n_samples),
    'nitrogen': np.random.uniform(0, 150, n_samples),
    'phosphorus': np.random.uniform(0, 100, n_samples),
    'potassium': np.random.uniform(0, 200, n_samples),
    'ndvi_avg': np.random.uniform(0.2, 0.9, n_samples),
}

# پیداوار کیلکولیٹ کریں (کوئنٹل/ایکڑ)
data['yield'] = (
    20 + 
    0.5 * data['temperature_avg'] +
    0.02 * data['rainfall'] +
    0.1 * data['humidity_avg'] +
    -2 * np.abs(data['soil_ph'] - 6.5) +
    0.05 * data['nitrogen'] +
    10 * data['ndvi_avg'] +
    np.random.normal(0, 3, n_samples)
)

df = pd.DataFrame(data)
print(df.describe())
```

## فیچر انجینئرنگ

```python
def create_features(df):
    """
    نئے فیچرز بنائیں
    """
    df = df.copy()
    
    # انٹرایکشنز
    df['temp_rainfall'] = df['temperature_avg'] * df['rainfall']
    df['npk_total'] = df['nitrogen'] + df['phosphorus'] + df['potassium']
    
    # تناسب
    df['n_p_ratio'] = df['nitrogen'] / (df['phosphorus'] + 1)
    
    # کیٹیگوری
    df['temp_category'] = pd.cut(df['temperature_avg'], 
                                  bins=[0, 20, 25, 30, 50],
                                  labels=['سرد', 'معتدل', 'گرم', 'بہت گرم'])
    
    # درجہ حرارت اونچا/نیچا
    df['is_optimal_temp'] = ((df['temperature_avg'] >= 20) & 
                              (df['temperature_avg'] <= 30)).astype(int)
    
    return df

df_features = create_features(df)
print(f"فیچرز کی تعداد: {df_features.shape[1]}")
```

## ماڈل 1: Random Forest

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# فیچرز اور ٹارگٹ
feature_cols = ['temperature_avg', 'rainfall', 'humidity_avg', 
                'soil_ph', 'nitrogen', 'phosphorus', 'potassium', 'ndvi_avg']

X = df[feature_cols]
y = df['yield']

# تقسیم
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ماڈل
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

# پیش گوئی
y_pred_rf = rf_model.predict(X_test)

# نتائج
print(f"R² سکور: {r2_score(y_test, y_pred_rf):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")

# فیچر اہمیت
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nفیچر اہمیت:")
print(importance)
```

## ماڈل 2: Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

print(f"Gradient Boosting R²: {r2_score(y_test, y_pred_gb):.3f}")
print(f"Gradient Boosting RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_gb)):.2f}")
```

## ماڈل 3: نیورل نیٹورک

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

# نارملائز کریں
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ماڈل
def build_yield_predictor(input_dim):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(16, activation='relu'),
        
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

nn_model = build_yield_predictor(X_train.shape[1])

# ٹریننگ
history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ],
    verbose=0
)

y_pred_nn = nn_model.predict(X_test_scaled).flatten()
print(f"Neural Network R²: {r2_score(y_test, y_pred_nn):.3f}")
```

## ٹائم سیریز پیش گوئی

```python
from sklearn.linear_model import LinearRegression

def create_time_features(dates):
    """
    تاریخ سے فیچرز
    """
    df = pd.DataFrame({'date': dates})
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['season'] = df['month'].apply(lambda x: 
        'بہار' if x in [3,4,5] else
        'گرمی' if x in [6,7,8] else
        'خزاں' if x in [9,10,11] else 'سردی')
    
    return df

# موسمی رجحان
def seasonal_yield_model(historical_data):
    """
    موسمی پیداوار ماڈل
    """
    # مہینے کی اوسط
    monthly_avg = historical_data.groupby('month')['yield'].mean()
    
    # ٹرینڈ
    yearly_avg = historical_data.groupby('year')['yield'].mean()
    
    return {
        'monthly_pattern': monthly_avg,
        'yearly_trend': yearly_avg
    }
```

## انسمبل ماڈل

```python
class YieldEnsemble:
    def __init__(self):
        self.models = {}
        self.weights = {}
    
    def add_model(self, name, model, weight=1.0):
        self.models[name] = model
        self.weights[name] = weight
    
    def fit(self, X, y):
        for name, model in self.models.items():
            model.fit(X, y)
    
    def predict(self, X):
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # وزنی اوسط
        total_weight = sum(self.weights.values())
        ensemble_pred = np.zeros(len(X))
        
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred / total_weight
        
        return ensemble_pred

# استعمال
ensemble = YieldEnsemble()
ensemble.add_model('rf', RandomForestRegressor(n_estimators=50), weight=2)
ensemble.add_model('gb', GradientBoostingRegressor(n_estimators=50), weight=1)

ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)

print(f"Ensemble R²: {r2_score(y_test, y_pred_ensemble):.3f}")
```

## ماڈل موازنہ

```python
def compare_models(y_test, predictions_dict):
    """
    ماڈلز کا موازنہ
    """
    results = []
    
    for name, y_pred in predictions_dict.items():
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({
            'ماڈل': name,
            'R²': f"{r2:.3f}",
            'RMSE': f"{rmse:.2f}"
        })
    
    return pd.DataFrame(results)

# موازنہ
predictions = {
    'Random Forest': y_pred_rf,
    'Gradient Boosting': y_pred_gb,
    'Neural Network': y_pred_nn,
    'Ensemble': y_pred_ensemble
}

comparison = compare_models(y_test, predictions)
print("\nماڈلز کا موازنہ:")
print(comparison.to_string(index=False))
```

## پیداوار پیش گوئی API

```python
class YieldPredictor:
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler
    
    def predict(self, field_data):
        """
        فیلڈ ڈیٹا سے پیداوار پیش گوئی
        """
        features = np.array([[
            field_data['temperature'],
            field_data['rainfall'],
            field_data['humidity'],
            field_data['soil_ph'],
            field_data['nitrogen'],
            field_data['phosphorus'],
            field_data['potassium'],
            field_data['ndvi']
        ]])
        
        if self.scaler:
            features = self.scaler.transform(features)
        
        prediction = self.model.predict(features)[0]
        
        return {
            'predicted_yield': round(prediction, 2),
            'unit': 'کوئنٹل/ایکڑ',
            'confidence': 'درمیانی'
        }

# استعمال
predictor = YieldPredictor(rf_model)

field_info = {
    'temperature': 28,
    'rainfall': 150,
    'humidity': 65,
    'soil_ph': 6.5,
    'nitrogen': 80,
    'phosphorus': 40,
    'potassium': 60,
    'ndvi': 0.7
}

result = predictor.predict(field_info)
print(f"\n📊 پیش گوئی: {result['predicted_yield']} {result['unit']}")
```

## خلاصہ

- پیداوار پیش گوئی کسانوں کی مدد کرتی ہے
- متعدد ماڈلز دستیاب ہیں
- انسمبل ماڈل بہتر نتائج دیتا ہے
- فیچر انجینئرنگ اہم ہے

## اگلے اقدامات

- [سمارٹ آبپاشی](/docs/module-4/smart-irrigation)
