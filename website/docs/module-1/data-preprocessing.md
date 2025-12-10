---
sidebar_position: 2
---

# Data Preprocessing for Plant Science

Data preprocessing is the foundation of successful machine learning in plant science. Raw agricultural data is often messy, incomplete, and inconsistent. This lesson teaches you how to transform raw data into clean, analysis-ready datasets.

## Why Preprocessing Matters

**Real-world scenario**: You have sensor data from 100 tomato plants over 90 days. The dataset has:
- 15% missing values (sensor failures)
- Outliers (temperature spikes from direct sunlight)
- Different scales (temperature in °C, humidity in %)
- Noise from environmental interference

**Without proper preprocessing**, your ML model will learn from errors instead of patterns.

## Common Data Issues in Agriculture

### 1. Missing Data

**Causes:**
- Sensor failures or disconnections
- Weather station downtime
- Manual data entry errors
- Network connectivity issues

**Example:**
```python
import pandas as pd
import numpy as np

# Sample data with missing values
data = pd.DataFrame({
    'plant_id': [1, 2, 3, 4, 5],
    'height_cm': [45.2, np.nan, 52.1, 48.3, np.nan],
    'leaf_count': [12, 15, np.nan, 14, 16],
    'soil_moisture': [0.35, 0.42, 0.38, np.nan, 0.40]
})

print("Missing values per column:")
print(data.isnull().sum())
```

**Output:**
```
plant_id        0
height_cm       2
leaf_count      1
soil_moisture   1
```

### 2. Outliers

**Causes:**
- Sensor calibration drift
- Physical damage to sensors
- Extreme weather events
- Data entry mistakes

**Detection:**
```python
import matplotlib.pyplot as plt

# Temperature data with outlier
temperatures = [22, 23, 24, 22, 23, 95, 24, 23, 22]  # 95°C is outlier

plt.boxplot(temperatures)
plt.ylabel('Temperature (°C)')
plt.title('Temperature Data with Outlier')
plt.show()
```

### 3. Different Scales

**Problem**: ML algorithms sensitive to feature magnitudes.

**Example:**
```python
data = pd.DataFrame({
    'temperature_celsius': [22, 24, 26],      # Range: 20-30
    'humidity_percent': [65, 70, 75],         # Range: 40-100
    'soil_nitrogen_ppm': [45, 50, 48]         # Range: 0-200
})

# Different ranges affect model training!
```

## Handling Missing Data

### Strategy 1: Remove Missing Data

**When to use**: Small percentage of missing data (less than 5%)

```python
# Remove rows with any missing values
data_clean = data.dropna()

# Remove columns with many missing values
data_clean = data.dropna(axis=1, thresh=len(data)*0.7)  # Keep if <30% missing
```

### Strategy 2: Simple Imputation

**When to use**: Random missing patterns

```python
from sklearn.impute import SimpleImputer

# Mean imputation for numeric data
imputer = SimpleImputer(strategy='mean')
data[['height_cm', 'soil_moisture']] = imputer.fit_transform(
    data[['height_cm', 'soil_moisture']]
)

# Mode imputation for categorical data
imputer_cat = SimpleImputer(strategy='most_frequent')
data[['plant_variety']] = imputer_cat.fit_transform(data[['plant_variety']])
```

**Strategies:**
- `mean`: Average value (for normal distributions)
- `median`: Middle value (robust to outliers)
- `most_frequent`: Mode (for categorical data)
- `constant`: Specific value (e.g., 0)

### Strategy 3: Forward/Backward Fill

**When to use**: Time-series data

```python
# Carry last observation forward
data['soil_moisture'] = data['soil_moisture'].fillna(method='ffill')

# Use next observation
data['soil_moisture'] = data['soil_moisture'].fillna(method='bfill')
```

### Strategy 4: Interpolation

**When to use**: Smooth time-series data

```python
# Linear interpolation
data['height_cm'] = data['height_cm'].interpolate(method='linear')

# Time-based interpolation
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.set_index('timestamp')
data['temperature'] = data['temperature'].interpolate(method='time')
```

### Strategy 5: Predictive Imputation

**When to use**: Complex patterns, enough data

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Use other features to predict missing values
imputer = IterativeImputer(max_iter=10, random_state=42)
data_imputed = imputer.fit_transform(data)
```

## Handling Outliers

### Detection Methods

#### 1. Statistical Methods (Z-Score)

```python
from scipy import stats
import numpy as np

def remove_outliers_zscore(data, column, threshold=3):
    """Remove outliers using z-score method"""
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

# Apply
data_clean = remove_outliers_zscore(data, 'temperature', threshold=3)
```

#### 2. Interquartile Range (IQR)

```python
def remove_outliers_iqr(data, column):
    """Remove outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Apply
data_clean = remove_outliers_iqr(data, 'soil_nitrogen')
```

#### 3. Domain Knowledge

**Best approach**: Use agricultural expertise!

```python
# Define realistic bounds based on plant biology
def remove_outliers_domain(data):
    """Remove outliers using domain knowledge"""
    return data[
        (data['temperature'] >= 5) & (data['temperature'] <= 45) &  # °C
        (data['humidity'] >= 20) & (data['humidity'] <= 100) &      # %
        (data['soil_moisture'] >= 0) & (data['soil_moisture'] <= 1) # Ratio
    ]
```

### Handling Outliers (Not Removing)

```python
# Cap at percentiles
data['temperature'] = data['temperature'].clip(
    lower=data['temperature'].quantile(0.05),
    upper=data['temperature'].quantile(0.95)
)

# Log transformation for skewed data
data['yield_log'] = np.log1p(data['yield'])
```

## Feature Scaling

### Why Scale?

Many ML algorithms (SVM, Neural Networks, K-Means) are sensitive to feature magnitudes.

### Method 1: Standardization (Z-Score Normalization)

**Formula**: z = (x - μ) / σ

**Result**: Mean = 0, Std = 1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['temperature', 'humidity', 'nitrogen']])

print("Mean:", scaled_data.mean(axis=0))  # ~[0, 0, 0]
print("Std:", scaled_data.std(axis=0))    # ~[1, 1, 1]
```

**When to use**: Features follow normal distribution

### Method 2: Min-Max Normalization

**Formula**: `x_norm = (x - x_min) / (x_max - x_min)`

**Result**: Range = [0, 1]

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['temperature', 'humidity']])

print("Min:", scaled_data.min(axis=0))  # [0, 0]
print("Max:", scaled_data.max(axis=0))  # [1, 1]
```

**When to use**: Need specific range, neural networks

### Method 3: Robust Scaling

**Uses**: Median and IQR (robust to outliers)

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaled_data = scaler.fit_transform(data[['yield', 'plant_height']])
```

**When to use**: Data has outliers

## Feature Engineering

### Creating Meaningful Features

#### 1. Time-Based Features

```python
# Convert to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Extract features
data['day_of_year'] = data['timestamp'].dt.dayofyear
data['week_of_year'] = data['timestamp'].dt.isocalendar().week
data['month'] = data['timestamp'].dt.month
data['is_growing_season'] = data['month'].isin([4, 5, 6, 7, 8, 9])

# Growing degree days (GDD)
data['gdd'] = np.maximum(data['avg_temp'] - 10, 0)  # Base temp 10°C
```

#### 2. Aggregated Features

```python
# Rolling averages (smooth noisy data)
data['temp_7day_avg'] = data['temperature'].rolling(window=7).mean()
data['rainfall_30day_sum'] = data['rainfall'].rolling(window=30).sum()

# Growth rates
data['height_growth_rate'] = data['height'].diff() / data['days'].diff()
```

#### 3. Interaction Features

```python
# Combine features that interact
data['heat_moisture_index'] = data['temperature'] * data['humidity'] / 100
data['light_temp_ratio'] = data['light_hours'] / (data['temperature'] + 1)

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data[['nitrogen', 'phosphorus']])
```

#### 4. Domain-Specific Indices

```python
# Vegetation indices (from satellite/drone imagery)
data['ndvi'] = (data['nir'] - data['red']) / (data['nir'] + data['red'])
data['evi'] = 2.5 * (data['nir'] - data['red']) / (data['nir'] + 6*data['red'] - 7.5*data['blue'] + 1)

# Stress indices
data['water_stress'] = 1 - (data['soil_moisture'] / data['field_capacity'])
data['nutrient_stress'] = data['optimal_n'] - data['current_n']
```

## Encoding Categorical Variables

### One-Hot Encoding

**When to use**: Nominal categories (no order)

```python
# Manual
data_encoded = pd.get_dummies(data, columns=['variety', 'soil_type'])

# Using sklearn
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded = encoder.fit_transform(data[['variety']])
```

**Example:**
```
variety_tomato    variety_pepper    variety_lettuce
      1                 0                 0
      0                 1                 0
      0                 0                 1
```

### Label Encoding

**When to use**: Ordinal categories (have order)

```python
from sklearn.preprocessing import LabelEncoder

# Disease severity: low < medium < high
encoder = LabelEncoder()
data['severity_encoded'] = encoder.fit_transform(data['disease_severity'])

# Output: low=0, medium=1, high=2
```

### Target Encoding

**When to use**: High cardinality categorical features

```python
# Mean encoding (use with caution - can leak info!)
variety_means = data.groupby('variety')['yield'].mean()
data['variety_encoded'] = data['variety'].map(variety_means)
```

## Handling Imbalanced Data

### Problem

```python
# Disease detection dataset
print(data['disease'].value_counts())
# healthy:    9500 (95%)
# diseased:    500 (5%)

# Model will predict "healthy" for everything!
```

### Solution 1: Resampling

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Oversample minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Undersample majority class
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
```

### Solution 2: Class Weights

```python
from sklearn.ensemble import RandomForestClassifier

# Automatically balance class weights
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)
```

## Complete Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Define preprocessing for numeric columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['temperature', 'humidity', 'nitrogen']),
        ('cat', categorical_transformer, ['variety', 'soil_type'])
    ])

# Create full pipeline with model
from sklearn.ensemble import RandomForestRegressor

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Train
full_pipeline.fit(X_train, y_train)

# Predict (preprocessing happens automatically!)
predictions = full_pipeline.predict(X_test)
```

## Best Practices

### 1. **Understand Your Data First**
```python
# Explore before preprocessing
print(data.describe())
print(data.info())
data.hist(bins=50, figsize=(20,15))
plt.show()
```

### 2. **Split Before Preprocessing**
```python
# WRONG: Fit on all data (data leakage!)
scaler.fit(data)

# CORRECT: Fit only on training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3. **Document All Steps**
```python
# Keep track of preprocessing decisions
preprocessing_log = {
    'missing_values': 'median imputation',
    'outliers': 'IQR method, removed 2%',
    'scaling': 'StandardScaler',
    'features_added': ['gdd', 'ndvi', 'temp_7day_avg'],
    'features_removed': ['sensor_id', 'field_notes']
}
```

### 4. **Validate Results**
```python
# Check preprocessing output
assert not data_processed.isnull().any().any(), "Still have missing values!"
assert data_processed['temperature'].min() >= -5, "Unrealistic temperature"
print(f"Dataset shape: {data_processed.shape}")
print(f"Memory usage: {data_processed.memory_usage().sum() / 1024**2:.2f} MB")
```

## Real-World Example: Wheat Yield Prediction

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('wheat_yield_data.csv')

# 1. Handle missing values
imputer = SimpleImputer(strategy='median')
numeric_cols = ['temperature', 'rainfall', 'soil_n', 'soil_p', 'soil_k']
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# 2. Remove outliers
data = data[data['yield'] < data['yield'].quantile(0.99)]

# 3. Feature engineering
data['gdd'] = np.maximum(data['temperature'] - 10, 0)
data['rainfall_30d'] = data['rainfall'].rolling(30, min_periods=1).sum()
data['npk_ratio'] = data['soil_n'] / (data['soil_p'] + data['soil_k'] + 1)

# 4. Split data
features = ['temperature', 'rainfall', 'soil_n', 'soil_p', 'soil_k', 
            'gdd', 'rainfall_30d', 'npk_ratio']
X = data[features]
y = data['yield']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f} tons/ha")
print(f"R²: {r2:.3f}")
```

## Summary

| Task | Method | Use Case |
|------|--------|----------|
| **Missing Data** | Mean/Median Imputation | Random missing values |
| | Forward/Backward Fill | Time-series |
| | Predictive Imputation | Complex patterns |
| **Outliers** | Z-score | Normal distribution |
| | IQR | Skewed distribution |
| | Domain Knowledge | Always preferred! |
| **Scaling** | StandardScaler | Normal distribution |
| | MinMaxScaler | Need [0,1] range |
| | RobustScaler | Data has outliers |
| **Categorical** | One-Hot | Nominal categories |
| | Label Encoding | Ordinal categories |

## Next Steps

Now that you can preprocess plant data, you're ready to build classification models! 

**Continue to:** [Plant Classification Models →](./classification-models)

---

**🌱 Good preprocessing = Good models!**
