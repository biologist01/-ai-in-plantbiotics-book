---
sidebar_position: 2
---

# پودوں کی سائنس کے لیے ڈیٹا پری پروسیسنگ

ڈیٹا پری پروسیسنگ پودوں کی سائنس میں کامیاب مشین لرننگ کی بنیاد ہے۔ خام زرعی ڈیٹا اکثر گندا، نامکمل، اور غیر مستقل ہوتا ہے۔ یہ لیکچر آپ کو خام ڈیٹا کو صاف، تجزیہ کے لیے تیار ڈیٹاسیٹ میں تبدیل کرنے کا طریقہ سکھاتی ہے۔

## پری پروسیسنگ کیوں ضروری ہے؟

**حقیقی دنیا کی مثال**: آپ کے پاس 100 ٹماٹر پودوں سے 90 دنوں کا سینسر ڈیٹا ہے۔ ڈیٹاسیٹ میں:
- 15% مسنگ ویلیوز (سینسر کی ناکامی)
- آؤٹ لائرز (براہ راست دھوپ سے درجہ حرارت میں اضافہ)
- مختلف اسکیلز (درجہ حرارت °C میں، نمی % میں)
- ماحولیاتی مداخلت سے شور

**مناسب پری پروسیسنگ کے بغیر**، آپ کا ML ماڈل غلطیوں سے سیکھے گا نمونوں کے بجائے۔

## زراعت میں عام ڈیٹا مسائل

### 1. مسنگ ڈیٹا

**وجوہات:**
- سینسر کی ناکامی یا ڈس کنکشن
- ویذر اسٹیشن ڈاؤن ٹائم
- دستی ڈیٹا انٹری کی غلطیاں
- نیٹورک کنیکٹیویٹی کے مسائل

**مثال:**
```python
import pandas as pd
import numpy as np

# مسنگ ویلیوز کے ساتھ نمونہ ڈیٹا
data = pd.DataFrame({
    'plant_id': [1, 2, 3, 4, 5],
    'height_cm': [45.2, np.nan, 52.1, 48.3, np.nan],
    'leaf_count': [12, 15, np.nan, 14, 16],
    'soil_moisture': [0.35, 0.42, 0.38, np.nan, 0.40]
})

print("ہر کالم میں مسنگ ویلیوز:")
print(data.isnull().sum())
```

**آؤٹ پٹ:**
```
plant_id        0
height_cm       2
leaf_count      1
soil_moisture   1
dtype: int64
```

### مسنگ ڈیٹا کو ہینڈل کرنے کے طریقے

#### 1. ڈیلیٹ کرنا (Deletion)

**جب استعمال کریں:** جب ڈیٹا کافی مقدار میں ہو اور مسنگ ویلیوز کم ہوں۔

```python
# مکمل کیسز ڈیلیٹ کریں
data_drop_na = data.dropna()
print(f"اصل ریکارڈز: {len(data)}")
print(f"کلین ریکارڈز: {len(data_drop_na)}")

# مخصوص کالم کے لیے
data_drop_col = data.dropna(subset=['height_cm'])
print(f"اونچائی کالم کلین کرنے کے بعد: {len(data_drop_col)}")
```

#### 2. ایمپیوٹیشن (Imputation)

**جب استعمال کریں:** جب ڈیٹا محدود ہو اور مسنگ ویلیوز کو اندازہ لگایا جا سکے۔

```python
# مین سے ایمپیوٹ کریں
data_mean = data.copy()
data_mean['height_cm'] = data_mean['height_cm'].fillna(data_mean['height_cm'].mean())
data_mean['soil_moisture'] = data_mean['soil_moisture'].fillna(data_mean['soil_moisture'].mean())

print("مین ایمپیوٹیشن کے بعد:")
print(data_mean)

# میڈین سے ایمپیوٹ کریں (آؤٹ لائرز کے لیے بہتر)
data_median = data.copy()
data_median['height_cm'] = data_median['height_cm'].fillna(data_median['height_cm'].median())

# فارورڈ فل (ٹائم سیریز کے لیے)
data_ffill = data.copy()
data_ffill = data_ffill.fillna(method='ffill')

print("فارورڈ فل کے بعد:")
print(data_ffill)
```

#### 3. ایڈوانس ایمپیوٹیشن

```python
# KNN ایمپیوٹر
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=2)
data_knn = data.copy()
numeric_cols = ['height_cm', 'leaf_count', 'soil_moisture']
data_knn[numeric_cols] = imputer.fit_transform(data_knn[numeric_cols])

print("KNN ایمپیوٹیشن کے بعد:")
print(data_knn)
```

### 2. آؤٹ لائرز (Outliers)

**زرعی آؤٹ لائرز کی مثالیں:**
- درجہ حرارت سینسر پر دھوپ کی براہ راست تابش
- بارش گیج میں پانی کا جمع ہونا
- مٹی سینسر کی خرابی سے غلط ریڈنگز

```python
import matplotlib.pyplot as plt
import seaborn as sns

# نمونہ ڈیٹا کے ساتھ آؤٹ لائرز
np.random.seed(42)
normal_data = np.random.normal(25, 5, 100)
outliers = np.array([80, -10, 90])  # آؤٹ لائرز
all_temps = np.concatenate([normal_data, outliers])

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(all_temps, bins=20, alpha=0.7)
plt.title('درجہ حرارت کی تقسیم (آؤٹ لائرز کے ساتھ)')
plt.xlabel('درجہ حرارت (°C)')

# باکس پلٹ
plt.subplot(1, 2, 2)
plt.boxplot(all_temps)
plt.title('درجہ حرارت باکس پلٹ')
plt.ylabel('درجہ حرارت (°C)')
plt.tight_layout()
plt.show()
```

#### آؤٹ لائرز کا پتہ لگانا

```python
# IQR طریقہ
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Z-اسکور طریقہ
from scipy import stats

def detect_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    outliers = data[z_scores > threshold]
    return outliers

# مثال
temp_data = pd.DataFrame({'temperature': all_temps})

outliers_iqr, lb, ub = detect_outliers_iqr(temp_data, 'temperature')
outliers_z = detect_outliers_zscore(temp_data, 'temperature')

print(f"IQR طریقہ سے آؤٹ لائرز: {len(outliers_iqr)}")
print(f"Z-اسکور طریقہ سے آؤٹ لائرز: {len(outliers_z)}")
print(f"IQR حدود: {lb:.1f} - {ub:.1f}")
```

#### آؤٹ لائرز کو ہینڈل کرنا

```python
# آؤٹ لائرز کو کیپ کریں
def cap_outliers(data, column, lower_percentile=5, upper_percentile=95):
    lower_cap = data[column].quantile(lower_percentile/100)
    upper_cap = data[column].quantile(upper_percentile/100)
    
    data[column] = np.clip(data[column], lower_cap, upper_cap)
    return data

# آؤٹ لائرز کو ہٹائیں
def remove_outliers(data, column, method='iqr'):
    if method == 'iqr':
        outliers, _, _ = detect_outliers_iqr(data, column)
    else:
        outliers = detect_outliers_zscore(data, column)
    
    clean_data = data[~data.index.isin(outliers.index)]
    return clean_data

# مثال
temp_data_capped = cap_outliers(temp_data.copy(), 'temperature')
temp_data_clean = remove_outliers(temp_data.copy(), 'temperature')

print(f"اصل ڈیٹا: {len(temp_data)} پوائنٹس")
print(f"کیپڈ ڈیٹا: {len(temp_data_capped)} پوائنٹس")
print(f"کلین ڈیٹا: {len(temp_data_clean)} پوائنٹس")
```

### 3. ڈیٹا نارملائزیشن اور اسکیلنگ

**کیوں ضروری ہے:**
- مختلف یونٹس (درجہ حرارت °C، نمی %، نمی 0-1)
- الگورتھم کی کارکردگی بہتر بنانا
- گرادیئنٹ ڈیسینٹ کو تیز کرنا

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# نمونہ ڈیٹا
plant_data = pd.DataFrame({
    'temperature': [20, 25, 30, 35, 40],
    'humidity': [40, 50, 60, 70, 80],
    'soil_moisture': [0.1, 0.3, 0.5, 0.7, 0.9],
    'leaf_area': [50, 100, 150, 200, 250]  # مختلف اسکیل
})

print("اصل ڈیٹا:")
print(plant_data.describe())

# StandardScaler (Z-اسکور نارملائزیشن)
scaler_standard = StandardScaler()
data_standard = pd.DataFrame(
    scaler_standard.fit_transform(plant_data),
    columns=plant_data.columns
)

print("\nStandardScaler کے بعد:")
print(data_standard.describe())

# MinMaxScaler (0-1 رینج)
scaler_minmax = MinMaxScaler()
data_minmax = pd.DataFrame(
    scaler_minmax.fit_transform(plant_data),
    columns=plant_data.columns
)

print("\nMinMaxScaler کے بعد:")
print(data_minmax.describe())

# RobustScaler (آؤٹ لائرز کے لیے)
scaler_robust = RobustScaler()
data_robust = pd.DataFrame(
    scaler_robust.fit_transform(plant_data),
    columns=plant_data.columns
)

print("\nRobustScaler کے بعد:")
print(data_robust.describe())
```

### 4. کیٹیگوریکل ڈیٹا کو ہینڈل کرنا

```python
# لیبل انکوڈنگ
from sklearn.preprocessing import LabelEncoder

plant_types = pd.DataFrame({
    'plant_type': ['tomato', 'wheat', 'rice', 'corn', 'tomato'],
    'growth_stage': ['seedling', 'vegetative', 'flowering', 'fruiting', 'vegetative']
})

# لیبل انکوڈر
le_type = LabelEncoder()
le_stage = LabelEncoder()

plant_types_encoded = plant_types.copy()
plant_types_encoded['plant_type_encoded'] = le_type.fit_transform(plant_types['plant_type'])
plant_types_encoded['growth_stage_encoded'] = le_stage.fit_transform(plant_types['growth_stage'])

print("اصل اور انکوڈڈ ڈیٹا:")
print(plant_types_encoded)

# ون ہاٹ انکوڈنگ
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(plant_types[['plant_type', 'growth_stage']])
feature_names = encoder.get_feature_names_out(['plant_type', 'growth_stage'])

plant_types_onehot = pd.DataFrame(encoded_features, columns=feature_names)
print("\nOne-Hot Encoded:")
print(plant_types_onehot)
```

### 5. فیچر انجینئرنگ

```python
# زرعی ڈومین کے لیے مخصوص فیچرز
def create_agricultural_features(df):
    df = df.copy()
    
    # ماحولیاتی اسٹریس انڈیکس
    df['environmental_stress'] = (
        (df['temperature'] - 25)**2 / 100 +  # بہترین درجہ حرارت 25°C
        (df['humidity'] - 60)**2 / 100 +     # بہترین نمی 60%
        (1 - df['soil_moisture'])**2         # بہترین نمی 1.0
    ) / 3
    
    # پانی کی ضرورت انڈیکس
    df['water_stress'] = np.where(
        df['soil_moisture'] < 0.3,
        (0.3 - df['soil_moisture']) / 0.3,
        0
    )
    
    # نشوونما کی شرح (اگر ٹائم سیریز ڈیٹا ہو)
    if 'height_cm' in df.columns:
        df['growth_rate'] = df['height_cm'].diff().fillna(0)
    
    # موسمی کیٹیگریز
    df['temp_category'] = pd.cut(
        df['temperature'],
        bins=[-10, 10, 20, 30, 40, 50],
        labels=['very_cold', 'cold', 'optimal', 'warm', 'hot']
    )
    
    # نمی کیٹیگریز
    df['humidity_category'] = pd.cut(
        df['humidity'],
        bins=[0, 30, 50, 70, 100],
        labels=['very_dry', 'dry', 'optimal', 'humid']
    )
    
    return df

# مثال ڈیٹا
sample_data = pd.DataFrame({
    'temperature': [22, 28, 35, 18, 25],
    'humidity': [45, 65, 30, 80, 55],
    'soil_moisture': [0.2, 0.6, 0.1, 0.8, 0.4],
    'height_cm': [10, 15, 12, 20, 18]
})

enhanced_data = create_agricultural_features(sample_data)
print("اضافی فیچرز کے ساتھ ڈیٹا:")
print(enhanced_data[['environmental_stress', 'water_stress', 'temp_category', 'humidity_category']])
```

### 6. ڈیٹا کی توازن (Balancing)

زرعی ڈیٹاسیٹ اکثر غیر متوازن ہوتے ہیں (زیادہ صحت مند پودے، کم بیمار پودے)۔

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# غیر متوازن ڈیٹا کی مثال
X_imbalanced = np.random.rand(1000, 3)
y_imbalanced = np.random.choice([0, 1], size=1000, p=[0.9, 0.1])  # 90% صحت مند، 10% بیمار

print("اصل کلاس ڈسٹری بیوشن:")
print(Counter(y_imbalanced))

# SMOTE سے اوور سیمیپلنگ
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_imbalanced, y_imbalanced)

print("SMOTE کے بعد:")
print(Counter(y_smote))

# انڈر سیمیپلنگ
rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_imbalanced, y_imbalanced)

print("انڈر سیمیپلنگ کے بعد:")
print(Counter(y_under))
```

## پری پروسیسنگ پائپ لائن

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

def agricultural_preprocessing_pipeline():
    """
    مکمل زرعی ڈیٹا پری پروسیسنگ پائپ لائن
    """
    
    # کسٹم ٹرانس فارمرز
    def handle_outliers(X):
        df = pd.DataFrame(X, columns=['temp', 'humidity', 'moisture'])
        # آؤٹ لائرز کو کیپ کریں
        for col in df.columns:
            lower = df[col].quantile(0.05)
            upper = df[col].quantile(0.95)
            df[col] = np.clip(df[col], lower, upper)
        return df.values
    
    def create_features(X):
        df = pd.DataFrame(X, columns=['temp', 'humidity', 'moisture'])
        # فیچرز بنائیں
        df['stress_index'] = (
            (df['temp'] - 25)**2 + 
            (df['humidity'] - 60)**2 + 
            (1 - df['moisture'])**2
        ) / 3
        return df.values
    
    # پائپ لائن بنائیں
    pipeline = Pipeline([
        ('outlier_handling', FunctionTransformer(handle_outliers)),
        ('imputation', KNNImputer(n_neighbors=3)),
        ('feature_creation', FunctionTransformer(create_features)),
        ('scaling', StandardScaler())
    ])
    
    return pipeline

# استعمال
raw_data = np.array([
    [25, 65, 0.3],
    [np.nan, 45, 0.1],
    [30, np.nan, 0.5],
    [22, 70, np.nan]
])

pipeline = agricultural_preprocessing_pipeline()
processed_data = pipeline.fit_transform(raw_data)

print("پری پروسیسڈ ڈیٹا شیپ:", processed_data.shape)
print("پری پروسیسڈ ڈیٹا:")
print(processed_data)
```

## پری پروسیسنگ کی توثیق

```python
def validate_preprocessing(original_data, processed_data):
    """
    پری پروسیسنگ کی کوالٹی چیک کریں
    """
    results = {}
    
    # مسنگ ویلیوز چیک کریں
    missing_original = np.isnan(original_data).sum()
    missing_processed = np.isnan(processed_data).sum()
    results['missing_values_handled'] = missing_processed == 0
    
    # ڈیٹا کی حدود چیک کریں
    results['reasonable_ranges'] = (
        processed_data.min() >= -5 and processed_data.max() <= 5
    )
    
    # آؤٹ لائرز چیک کریں (Z-اسکور > 3)
    z_scores = np.abs(stats.zscore(processed_data, axis=0))
    outlier_count = (z_scores > 3).sum()
    results['outliers_controlled'] = outlier_count < len(processed_data) * 0.05
    
    return results

# توثیق
validation_results = validate_preprocessing(raw_data, processed_data)
print("پری پروسیسنگ توثیق:")
for check, passed in validation_results.items():
    print(f"  {check}: {'✅ پاس' if passed else '❌ فیل'}")
```

## اگلے اقدامات

- [ریگریشن ماڈلز](/docs/module-1/regression-models)
- [کلاسفیکیشن ماڈلز](/docs/module-1/classification-models)

## خلاصہ

ڈیٹا پری پروسیسنگ ML کی کامیابی کی کلید ہے:
- مسنگ ویلیوز کو ایمپیوٹ یا ڈیلیٹ کریں
- آؤٹ لائرز کا پتہ لگا کر ہینڈل کریں
- ڈیٹا کو اسکیل اور نارملائز کریں
- نئے فیچرز بنائیں
- پائپ لائنز سے آٹومیٹ کریں

صاف ڈیٹا = بہتر ماڈلز!