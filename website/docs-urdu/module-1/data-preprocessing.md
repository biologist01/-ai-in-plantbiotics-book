---
sidebar_position: 2
---

# پودوں کی سائنس کے لیے ڈیٹا کی پیشگی پروسیسنگ

ڈیٹا کی پیشگی پروسیسنگ پودوں کی سائنس میں مشین لیرننگ کی کامیابی کا بنیادی ستون ہے۔ خام زرعی ڈیٹا اکثر گندا، نامکمل، اور غیر مستحکم ہوتا ہے۔ یہ سبق آپ کو سیکھاتا ہے کہ خام ڈیٹا کو کیسے صاف، تجزیہ کے لیے تیار ڈیٹا سیٹ میں تبدیل کیا جائے۔

## کیوں پیشگی پروسیسنگ اہم ہے

**حقیقی دنیا کا منظر**: آپ کے پاس 100 ٹماٹر کے پودوں کے 90 دنوں کے لیے سینسر ڈیٹا ہے۔ ڈیٹا سیٹ میں:
- 15% مہیا نہیں کی گئی قدر (سینسر کی ناکامیاں)
- غیر معمولی اعداد (براہ راست سورج کی روشنی سے درجہ حرارت کے اچانک اتار چڑھاؤ)
- مختلف پیمانے (درجہ حرارت °C میں، نمی % میں)
- ماحولیاتی مداخلت سے شور

**پیشگی پروسیسنگ کے بغیر،** آپ کا ML ماڈل غلطیوں سے سیکھے گا، نہ کہ نمونوں سے۔

## زراعت میں عام ڈیٹا کے مسائل

### 1. مہیا نہیں کی گئی ڈیٹا

**وجوہات:**
- سینسر کی ناکامیاں یا منقطع ہونا
- موسم کی اسٹیشن کا نیچے جانا
- دستی ڈیٹا داخل کرنے میں غلطیاں
- نیٹ ورک کی کنیکٹیویٹی کے مسائل

**مثال:**
```python
import pandas as pd
import numpy as np

# مہیا نہیں کی گئی قدر کے ساتھ نمونہ ڈیٹا
data = pd.DataFrame({
    'plant_id': [1, 2, 3, 4, 5],
    'height_cm': [45.2, np.nan, 52.1, 48.3, np.nan],
    'leaf_count': [12, 15, np.nan, 14, 16],
    'soil_moisture': [0.35, 0.42, 0.38, np.nan, 0.40]
})

print("ہر کالم میں مہیا نہیں کی گئی اعداد:")
print(data.isnull().sum())
```

**آؤٹ پٹ:**
```
plant_id        0
height_cm       2
leaf_count      1
soil_moisture   1
```

### 2. غیر معمولی اعداد

**وجوہات:**
- سینسر کی کالبریشن کا خسارہ
- سینسروں کو جسمانی نقصان
- انتہائی موسم کے واقعات
- ڈیٹا داخل کرنے میں غلطیاں

**تشخیص:**
```python
import matplotlib.pyplot as plt

# درجہ حرارت کے ڈیٹا میں غیر معمولی عدد
درجہ_حرارت = [22, 23, 24, 22, 23, 95, 24, 23, 22]  # 95°C غیر معمولی عدد ہے

plt.boxplot(درجہ_حرارت)
plt.ylabel('درجہ حرارت (°C)')
plt.title('درجہ حرارت ڈیٹا میں غیر معمولی عدد')
plt.show()
```

### 3. مختلف پیمانے

**مسئلہ**: ML الگورتھم فیچر کی مقدار سے حساس ہوتے ہیں۔

**مثال:**
```python
data = pd.DataFrame({
    'درجہ_حرارت_سیلسیس': [22, 24, 26],      # سلسلہ: 20-30
    'نمی_فیصد': [65, 70, 75],         # سلسلہ: 40-100
    'مٹی_نائیٹروجن_ppm': [45, 50, 48]         # سلسلہ: 0-200
})

# مختلف سلسلے ماڈل کی تربیت کو متاثر کرتے ہیں!
```

## مہیا نہیں کی گئی ڈیٹا کو سنبھالنا

### حکمت عملی 1: مہیا نہیں کی گئی ڈیٹا کو ہٹانا

**جب استعمال کریں**: مہیا نہیں کی گئی ڈیٹا کی کم فیصد (5% سے کم)

```python
# مہیا نہیں کی گئی اعداد والی ساری قطاریں ہٹا دیں
data_clean = data.dropna()

# مہیا نہیں کی گئی اعداد کے ساتھ کالم کو ہٹا دیں
data_clean = data.dropna(axis=1, thresh=len(data)*0.7)  # 30% سے کم مہیا نہیں کی گئی ڈیٹا کے ساتھ رکھیں
```

### حکمت عملی 2: سادہ امپوٹیشن

**جب استعمال کریں**:随机 مہیا نہیں کی گئی ڈیٹا کے نمونے۔

```python
from sklearn.impute import SimpleImputer

# عددی ڈیٹا کے لیے اوسط امپوٹیشن
imputer = SimpleImputer(strategy='mean')
data[['height_cm', 'soil_moisture']] = imputer.fit_transform(
    data[['height_cm', 'soil_moisture']]
)

# زمرہ واری ڈیٹا کے لیے سب سے زیادہ تکرار کرنے والی امپوٹیشن
imputer_cat = SimpleImputer(strategy='most_frequent')
data[['plant_variety']] = imputer_cat.fit_transform(data[['plant_variety']])
```

**حکمت عملی:**
- `mean`: اوسط قدر (سادہ تقسیم کے لیے)
- `median`: درمیانی قدر (غیر معمولی اعداد سے محفوظ)
- `most_frequent`: تکرار (زمرہ واری ڈیٹا کے لیے)
- `constant`: مخصوص قدر (جیسے 0)

### حکمت عملی 3: آگے/پچھلے فیل

**جب استعمال کریں**: وقت کی سیریز ڈیٹا۔

```python
# آخری مشاہدہ آگے لے جائیں
data['soil_moisture'] = data['soil_moisture'].fillna(method='ffill')

# اگلی مشاہدہ کا استعمال کریں
data['soil_moisture'] = data['soil_moisture'].fillna(method='bfill')
```

### حکمت عملی 4: بین الزمانی

**جب استعمال کریں**: ہموار وقت کی سیریز ڈیٹا۔

```python
# لکیری بین الزمانی
data['height_cm'] = data['height_cm'].interpolate(method='linear')

# وقت کی بنیاد پر بین الزمانی
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.set_index('timestamp')
data['temperature'] = data['temperature'].interpolate(method='time')
```

### حکمت عملی 5: پیشگوئی امپوٹیشن

**جب استعمال کریں**: پیچیدہ نمونے، کافی ڈیٹا۔

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# دوسرے فیچرز کا استعمال مہیا نہیں کی گئی اعداد کی پیشگوئی کے لیے کریں
imputer = IterativeImputer(max_iter=10, random_state=42)
data_imputed = imputer.fit_transform(data)
```

## غیر معمولی اعداد کو سنبھالنا

### تشخصی کے तरीकے

#### 1. اعداد و شمار کے तरीकے (ز-اسکور)

```python
from scipy import stats
import numpy as np

def remove_outliers_zscore(data, column, threshold=3):
    """غیر معمولی اعداد کو ز-اسکور کے طریقہ سے ہٹائیں"""
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

# لاگو کریں
data_clean = remove_outliers_zscore(data, 'temperature', threshold=3)
```

#### 2. بین چہارم

```python
def remove_outliers_iqr(data, column):
    """غیر معمولی اعداد کو بین چہارم کے طریقہ سے ہٹائیں"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# لاگو کریں
data_clean = remove_outliers_iqr(data, 'soil_nitrogen')
```

#### 3. ڈومین کا علم

**بہترین نقطہ نظر**: زرعی مہارت کا استعمال کریں!

```python
# حقیقی حدود کا تعین پودوں کی سائنس کے مطابق کریں
def remove_outliers_domain(data):
    """غیر معمولی اعداد کو ڈومین کے علم کے طریقہ سے ہٹائیں"""
    return data[
        (data['temperature'] >= 5) & (data['temperature'] <= 45) &  # °C
        (data['humidity'] >= 20) & (data['humidity'] <= 100) &      # %
        (data['soil_moisture'] >= 0) & (data['soil_moisture'] <= 1) # تناسب
    ]
```

### غیر معمولی اعداد کو سنبھالنا (ہٹانا نہیں)

```python
# فیصد کے مطابق کپ کریں
data['temperature'] = data['temperature'].clip(
    lower=data['temperature'].quantile(0.05),
    upper=data['temperature'].quantile(0.95)
)

# لوگ تبدیلی اسکیمڈ ڈیٹا کے لیے
data['yield_log'] = np.log1p(data['yield'])
```

## فیچر سکیلنگ

### کیوں سکیل کریں?

کئی ML الگورتھم (SVM، نیورل نیٹ ورکس، K-Means) فیچر کی مقدار سے حساس ہوتے ہیں۔

### طریقہ 1: معیاری کرنا (ز-اسکور نارملائزیشن)

**صیغہ**: z = (x - μ) / σ

**نتیجہ**: اوسط = 0، معیار = 1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['temperature', 'humidity', 'nitrogen']])

print("اوسط:", scaled_data.mean(axis=0))  # ~[0, 0, 0]
print("معیار:", scaled_data.std(axis=0))    # ~[1, 1, 1]
```

**جب استعمال کریں**: فیچرز سادہ تقسیم کی پیروی کرتے ہیں

### طریقہ 2: من-مکس نارملائزیشن

**صیغہ**: `x_norm = (x - x_min) / (x_max - x_min)`

**نتیجہ**: سلسلہ = [0, 1]

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['temperature', 'humidity']])

print("کم سے کم:", scaled_data.min(axis=0))  # [0, 0]
print("زیادہ سے زیادہ:", scaled_data.max(axis=0))  # [1, 1]
```

**جب استعمال کریں**: مخصوص سلسلے کی ضرورت ہے، نیورل نیٹ ورکس

### طریقہ 3: روبسٹ سکیلنگ

**استعمال**: میڈین اور بین چہارم (غیر معمولی اعداد سے محفوظ)

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaled_data = scaler.fit_transform(data[['yield', 'plant_height']])
```

**جب استعمال کریں**: ڈیٹا میں غیر معمولی اعداد ہیں

## فیچر انجینئرنگ

### مفید فیچرز تخلیق کرنا

#### 1. وقت کی بنیاد پر فیچرز

```python
# ڈیٹ ٹائم میں تبدیل کریں
data['timestamp'] = pd.to_datetime(data['timestamp'])

# فیچرز نکالیں
data['day_of_year'] = data['timestamp'].dt.dayofyear
data['week_of_year'] = data['timestamp'].dt.isocalendar().week
data['month'] = data['timestamp'].dt.month
data['is_growing_season'] = data['month'].isin([4, 5, 6, 7, 8, 9])

# گروتھ ڈگری دن (GDD)
data['gdd'] = np.maximum(data['avg_temp'] - 10, 0)  # بیس ٹیمپ 10°C
```

#### 2. مجموعی فیچرز

```python
# رولنگ اوسط (شور کو ہموار کریں)
data['temp_7day_avg'] = data['temperature'].rolling(window=7).mean()
data['rainfall_30day_sum'] = data['rainfall'].rolling(window=30).sum()

# گروتھ ریٹ
data['height_growth_rate'] = data['height'].diff() / data['days'].diff()
```

#### 3. انٹرایکشن فیچرز

```python
# فیچرز کو ملائیں جو باہمی تعامل کرتے ہیں
data['heat_moisture_index'] = data['temperature'] * data['humidity'] / 100
data['light_temp_ratio'] = data['light_hours'] / (data['temperature'] + 1)

# پولی نومیئل فیچرز
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data[['nitrogen', 'phosphorus']])
```

#### 4. ڈومین مخصوص انڈیکس

```python
# نباتات کے انڈیکس (سیٹلائٹ/ڈرون امیجری سے)
data['ndvi'] = (data['nir'] - data['red']) / (data['nir'] + data['red'])
data['evi'] = 2.5 * (data['nir'] - data['red']) / (data['nir'] + 6*data['red'] - 7.5*data['blue'] + 1)

# تناؤ کے انڈیکس
data['water_stress'] = 1 - (data['soil_moisture'] / data['field_capacity'])
data['nutrient_stress'] = data['optimal_n'] - data['current_n']
```

## زمرہ واری متغیرات کو انکوڈ کرنا

### ون ہاٹ انکوڈنگ

**جب استعمال کریں**: زمرہ واری زمرے (کوئی ترتیب نہیں)

```python
# دستی
data_encoded = pd.get_dummies(data, columns=['variety', 'soil_type'])

# اسک لیرن کا استعمال کرتے ہوئے
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded = encoder.fit_transform(data[['variety']])
```

**مثال:**
```
variety_tomato    variety_pepper    variety_lettuce
      1                 0                 0
      0                 1                 0
      0                 0                 1
```

### لیبل انکوڈنگ

**جب استعمال کریں**: زمرہ واری زمرے (ترتیب ہے)

```python
from sklearn.preprocessing import LabelEncoder

# بیماری کی شدت: کم < درمیان < زیادہ
encoder = LabelEncoder()
data['severity_encoded'] = encoder.fit_transform(data['disease_severity'])

# آؤٹ پٹ: کم=0, درمیان=1, زیادہ=2
```

### ٹارگٹ انکوڈنگ

**جب استعمال کریں**: اعلیٰ کارڈینالٹی زمرہ واری فیچرز

```python
# اوسط انکوڈنگ (استعمال کرتے وقت احتیاط کریں - информیشن لیک ہو سکتا ہے!)
variety_means = data.groupby('variety')['yield'].mean()
data['variety_encoded'] = data['variety'].map(variety_means)
```

## غیر متوازن ڈیٹا کو سنبھالنا

### مسئلہ

```python
# بیماری کی تشخیص کا ڈیٹا سیٹ
print(data['disease'].value_counts())
# صحت مند:    9500 (95%)
# بیمار:    500 (5%)

# ماڈل ہر چیز کے لیے "صحت مند" کی پیشگوئی کرے گا!
```

### حل 1: ری سمپلنگ

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# اقلیتی کلاس کو اوور سمپل کریں
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# اکثریتی کلاس کو انڈر سمپل کریں
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
```

### حل 2: کلاس ویٹ

```python
from sklearn.ensemble import RandomForestClassifier

# خودکار طور پر کلاس ویٹ کو توازن کریں
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)
```

## مکمل پیشگی پروسیسنگ پائپ لائن

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# عددی کالم کے لیے پیشگی پروسیسنگ کا تعین کریں
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# زمرہ واری کالم کے لیے پیشگی پروسیسنگ کا تعین کریں
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# پیشگی پروسیسنگ کے مراحل کو ملائیں
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['temperature', 'humidity', 'nitrogen']),
        ('cat', categorical_transformer, ['variety', 'soil_type'])
    ])

# مکمل پائپ لائن کا تعین کریں
from sklearn.ensemble import RandomForestRegressor

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# تربیت دیں
full_pipeline.fit(X_train, y_train)

# پیشگوئی کریں (پیشگی پروسیسنگ خودکار طور پر ہوتی ہے!)
predictions = full_pipeline.predict(X_test)
```

## بہترین مشق

### 1. **اپنے ڈیٹا کو پہلے سمجھیں**
```python
# پیشگی پروسیسنگ سے پہلے کا جائزہ لیں
print(data.describe())
print(data.info())
data.hist(bins=50, figsize=(20,15))
plt.show()
```

### 2. **پیشگی پروسیسنگ سے پہلے تقسیم کریں**
```python
# غلط: پوری ڈیٹا پر فٹ کریں (ڈیٹا لیک!)
scaler.fit(data)

# درست: صرف تربیتی ڈیٹا پر فٹ کریں
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3. **سبھی مراحل کی دستاویز کریں**
```python
# پیشگی پروسیسنگ کے فیصلوں کو ٹریک کریں
preprocessing_log = {
    'missing_values': 'median imputation',
    'outliers': 'IQR method, removed 2%',
    'scaling': 'StandardScaler',
    'features_added': ['gdd', 'ndvi', 'temp_7day_avg'],
    'features_removed': ['sensor_id', 'field_notes']
}
```

### 4. **نتائج کی توثیق کریں**
```python
# پیشگی پروسیسنگ کے آؤٹ پٹ کو چیک کریں
assert not data_processed.isnull().any().any(), "اب بھی مہیا نہیں کی گئی اعداد موجود ہیں!"
assert data_processed['temperature'].min() >= -5, "غیر حقیقی درجہ حرارت"
print(f"ڈیٹا سیٹ کی شکل: {data_processed.shape}")
print(f"میموری کا استعمال: {data_processed.memory_usage().sum() / 1024**2:.2f} MB")
```

## حقیقی دنیا کا مثال: گندم کے ییلڈ کی پیشگوئی

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ڈیٹا لوڈ کریں
data = pd.read_csv('wheat_yield_data.csv')

# 1. مہیا نہیں کی گئی ڈیٹا کو سنبھالیں
imputer = SimpleImputer(strategy='median')
numeric_cols = ['temperature', 'rainfall', 'soil_n', 'soil_p', 'soil_k']
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# 2. غیر معمولی اعداد کو ہٹائیں
data = data[data['yield'] < data['yield'].quantile(0.99)]

# 3. فیچر انجینئرنگ
data['gdd'] = np.maximum(data['temperature'] - 10, 0)
data['rainfall_30d'] = data['rainfall'].rolling(30, min_periods=1).sum()
data['npk_ratio'] = data['soil_n'] / (data['soil_p'] + data['soil_k'] + 1)

# 4. ڈیٹا کو تقسیم کریں
features = ['temperature', 'rainfall', 'soil_n', 'soil_p', 'soil_k', 
            'gdd', 'rainfall_30d', 'npk_ratio']
X = data[features]
y = data['yield']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. فیچرز کو سکیل کریں
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. ماڈل کو تربیت دیں
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. ماڈل کی کارکردگی کا جائزہ لیں
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f} ٹن/ہیکٹر")
print(f"R²: {r2:.3f}")
```

## خلاصہ

| کام | طریقہ | استعمال |
|------|--------|----------|
| **مہیا نہیں کی گئی ڈیٹا** | میڈین/اوسط امپوٹیشن |随机 مہیا نہیں کی گئی ڈیٹا |
| | آگے/پچھلے فیل | وقت کی سیریز |
| | پیشگوئی امپوٹیشن | پیچیدہ نمونے |
| **غیر معمولی اعداد** | ز-اسکور | سادہ تقسیم |
| | بین چہارم | مائل تقسیم |
| | ڈومین کا علم | ہمیشہ ترجیح دی جانی چاہیے! |
| **سکیلنگ** | معیاری کرنا | سادہ تقسیم |
| | من-مکس نارملائزیشن | [0,1] سلسلے کی ضرورت |
| | روبسٹ سکیلنگ | ڈیٹا میں غیر معمولی اعداد |
| **زمرہ واری** | ون ہاٹ انکوڈنگ | زمرہ واری زمرے |
| | لیبل انکوڈنگ | زمرہ واری زمرے کی ترتیب |

## اگلا قدم

اب جب آپ پودوں کے ڈیٹا کو پیشگی پروسیسنگ کر سکتے ہیں، تو آپ کلاسیفیکیشن ماڈلز بنانے کے لیے تیار ہیں!

**آگے بڑھیں:** [پودوں کی کلاسیفیکیشن ماڈلز →](./classification-models)

---

**پودوں کی کامیابی کے لیے اچھی پیشگی پروسیسنگ = اچھے ماڈل!**