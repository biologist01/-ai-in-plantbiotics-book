---
sidebar_position: 1
---

# پودوں کی سائنس کے لیے مشین لرننگ کا تعارف

**مشین لرننگ (ML)** پودوں کی سائنس کو تبدیل کر رہی ہے جیسے کمپیوٹرز کو واضح پروگرامنگ کے بغیر ڈیٹا سے نمونے سیکھنے کی اجازت دی جائے۔ زراعت میں، ML ہمیں پودوں کے رویے کو سمجھنے، نتائج کی پیش گوئی کرنے، اور ڈیٹا پر مبنی فیصلے کرنے میں مدد کرتی ہے۔

## پودوں کی سائنس میں مشین لرننگ کیوں ضروری ہے؟

### ڈیٹا کی دھماکہ خیز افزائش

جدید زراعت ڈیٹا کی بڑی مقدار پیدا کرتی ہے:
- **سینسر نیٹورکس**: مٹی، موسم، پودوں کی صحت کی مسلسل نگرانی
- **امیجنگ سسٹم**: ہزاروں پودوں کی روزانہ تصاویر
- **جینومک ڈیٹا**: DNA سیکوئنسز کی لاکھوں تعداد
- **فیلڈ ٹرائلز**: فصلوں کی کارکردگی کے سالوں کا ڈیٹا

روایتی تجزیہ کے طریقے اس ڈیٹا کی مقدار سے ہم آہنگ نہیں ہو سکتے۔ مشین لرننگ کر سکتی ہے۔

### حقیقی دنیا کے اطلاقات

| اطلاق | ML تکنیک | اثر |
|-------|----------|-----|
| **فصل کی پیداوار کی پیش گوئی** | ریگریشن ماڈلز | فصل کی نقل و حمل اور مارکیٹ کی قیمتوں کی منصوبہ بندی |
| **بیماری کی تشخیص** | امیج کلاسفیکیشن | ابتدائی مداخلت، نقصانات کو کم کرنا |
| **قسم کی سلیکشن** | کلسٹرنگ | مقامی حالات کے مطابق فصلوں کا میل کھولنا |
| **نشوونما کی بہترین کاری** | ریئنفورسمنٹ لرننگ | کم سے کم وسائل کے ساتھ پیداوار کو زیادہ سے زیادہ کرنا |
| **جینومک سلیکشن** | ڈیپ لرننگ | بریڈنگ کو 10 گنا تیز کرنا |

## زراعت کے لیے بنیادی ML تصورات

### 1. سپروائزڈ لرننگ

**تعریف**: لیبل شدہ مثالوں سے سیکھنا تاکہ نئے ڈیٹا پر پیش گوئیاں کی جائیں۔

**پودوں کی سائنس کی مثالیں:**
- **کلاسفیکیشن**: کیا یہ پتی صحت مند ہے یا بیمار؟
- **ریگریشن**: موسم کی بنیاد پر فصل کی پیداوار کیا ہوگی؟

```python
# مثال: سینسر ڈیٹا سے پودوں کی صحت کی پیش گوئی
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ٹریننگ ڈیٹا: [درجہ حرارت, نمی, مٹی کی نمی]
X_train = np.array([
    [25, 65, 0.3],  # صحت مند پودے کی حالتیں
    [30, 40, 0.1],  # تناؤ والے پودے کی حالتیں
    [22, 70, 0.4],  # صحت مند
])

# لیبلز: 0 = بیمار، 1 = صحت مند
y_train = np.array([1, 0, 1])

# ماڈل ٹرین کریں
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# نئے ڈیٹا پر پیش گوئی
new_plant_data = np.array([[28, 55, 0.2]])
prediction = model.predict(new_plant_data)
probability = model.predict_proba(new_plant_data)

print(f"پیش گوئی: {'صحت مند' if prediction[0] == 1 else 'بیمار'}")
print(f"اعتماد: {probability[0][prediction[0]]:.2f}")
```

### 2. ان سپروائزڈ لرننگ

**تعریف**: ڈیٹا میں پوشیدہ نمونے اور ساخت تلاش کرنا بغیر لیبلز کے۔

**زرعی مثالیں:**
- پودوں کی اقسام کو گروپ کرنا
- غیر معمولی نمونے (آؤٹ لائرز) کی شناخت

```python
# مثال: پودوں کی اقسام کو کلسٹر کرنا
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# پودوں کی خصوصیات: اونچائی، پتیوں کی تعداد، پھل کی تعداد
plant_features = np.array([
    [50, 15, 8],   # قسم A
    [45, 12, 6],   # قسم A
    [80, 25, 15],  # قسم B
    [85, 22, 12],  # قسم B
    [60, 18, 10],  # قسم C
])

# 3 کلسٹرز میں گروپ کریں
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(plant_features)

# نتائج دکھائیں
for i, features in enumerate(plant_features):
    print(f"پودہ {i+1}: خصوصیات {features} -> کلسٹر {clusters[i]}")

# ویژولائزیشن
plt.scatter(plant_features[:, 0], plant_features[:, 1], c=clusters, cmap='viridis')
plt.xlabel('اونچائی (cm)')
plt.ylabel('پتیوں کی تعداد')
plt.title('پودوں کی اقسام کا کلسٹرنگ')
plt.show()
```

### 3. ریئنفورسمنٹ لرننگ

**تعریف**: ماحول کے ساتھ تعامل کے ذریعے سیکھنا تاکہ انعام کو زیادہ سے زیادہ کیا جائے۔

**زرعی اطلاقات:**
- آبپاشی کی بہترین کاری
- کھاد کی تقسیم

```python
# مثال: سمارٹ آبپاشی سسٹم
import random

class IrrigationAgent:
    def __init__(self):
        self.water_levels = [0.1, 0.2, 0.3, 0.4, 0.5]  # لیٹر فی گھنٹہ
        self.q_table = {}  # حالت -> عمل -> قیمت
    
    def get_state(self, soil_moisture, plant_health):
        """حالت کی نمائندگی"""
        return (round(soil_moisture, 1), plant_health)
    
    def choose_action(self, state, epsilon=0.1):
        """ایکشن منتخب کریں (epsilon-greedy)"""
        if random.random() < epsilon:
            return random.choice(self.water_levels)  # ریڈم
        
        # بہترین ایکشن
        if state not in self.q_table:
            self.q_table[state] = {level: 0 for level in self.water_levels}
        
        return max(self.q_table[state], key=self.q_table[state].get)
    
    def update_q_value(self, state, action, reward, next_state, alpha=0.1, gamma=0.9):
        """Q ویلیو اپ ڈیٹ کریں"""
        if state not in self.q_table:
            self.q_table[state] = {level: 0 for level in self.water_levels}
        if next_state not in self.q_table:
            self.q_table[next_state] = {level: 0 for level in self.water_levels}
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        self.q_table[state][action] = current_q + alpha * (
            reward + gamma * max_next_q - current_q
        )

# استعمال
agent = IrrigationAgent()

# مثال: مٹی کی نمی 0.2، پودے کی صحت اچھی
state = agent.get_state(0.2, 'good')
action = agent.choose_action(state)
print(f"مطلوبہ آبپاشی کی مقدار: {action} L/hr")
```

## ML پائپ لائن برائے زراعت

### 1. ڈیٹا اکٹھا کرنا

زرعی ڈیٹا کے ذرائع:
- **فیلڈ سینسرز**: مٹی کی نمی، درجہ حرارت، pH
- **ویذر اسٹیشنز**: بارش، ہوا، نمی
- **کیمرے**: پودوں کی تصاویر، ڈرون فوٹو
- **لیب آلات**: جینومک سیکوئنسرز، اسپیکٹرومٹرز

### 2. ڈیٹا پری پروسیسنگ

```python
# مثال: سینسر ڈیٹا کی کلیننگ
import pandas as pd
from sklearn.preprocessing import StandardScaler

# نمونہ ڈیٹا
data = pd.DataFrame({
    'temperature': [25, 30, None, 22, 28],
    'humidity': [65, 45, 70, None, 55],
    'soil_moisture': [0.3, 0.1, 0.4, 0.2, 0.35]
})

print("اصل ڈیٹا:")
print(data)

# مسنگ ویلیوز ہینڈل کریں
data_clean = data.fillna(data.mean())

# نارملائز کریں
scaler = StandardScaler()
numeric_cols = ['temperature', 'humidity', 'soil_moisture']
data_clean[numeric_cols] = scaler.fit_transform(data_clean[numeric_cols])

print("\nکلین اور نارملائزڈ ڈیٹا:")
print(data_clean)
```

### 3. فیچر انجینئرنگ

```python
# مثال: نئے فیچرز بنانا
def create_agricultural_features(df):
    df = df.copy()
    
    # ماحولیاتی اسٹریس انڈیکس
    df['stress_index'] = (
        (df['temperature'] - 25)**2 + 
        (df['humidity'] - 60)**2 + 
        (1 - df['soil_moisture'])**2
    ) / 3
    
    # پانی کی ضرورت
    df['water_deficit'] = np.where(
        df['soil_moisture'] < 0.3, 
        0.3 - df['soil_moisture'], 
        0
    )
    
    # موسمی کیٹیگری
    df['temp_category'] = pd.cut(
        df['temperature'], 
        bins=[0, 15, 25, 35, 50], 
        labels=['ٹھنڈا', 'معتدل', 'گرم', 'بہت گرم']
    )
    
    return df

# فیچرز شامل کریں
data_enhanced = create_agricultural_features(data_clean)
print("اضافی فیچرز کے ساتھ ڈیٹا:")
print(data_enhanced[['stress_index', 'water_deficit', 'temp_category']])
```

### 4. ماڈل ٹریننگ اور تشخیص

```python
# مثال: کراس ویلیڈیشن
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

# ماڈل اور ڈیٹا
model = RandomForestRegressor(n_estimators=100, random_state=42)
X = data_clean[['temperature', 'humidity']]
y = data_clean['soil_moisture']

# کراس ویلیڈیشن
scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error')
print(f"MAE سکورز: {-scores}")
print(f"اوسط MAE: {-scores.mean():.3f}")

# مکمل ٹریننگ
model.fit(X, y)
predictions = model.predict(X)
mae = mean_absolute_error(y, predictions)
print(f"ٹریننگ MAE: {mae:.3f}")
```

## چیلنجز اور حل

### ڈیٹا کوالٹی مسائل

**چیلنجز:**
- شور (noise) سینسرز سے
- مسنگ ڈیٹا فیلڈ میں
- غیر متوازن ڈیٹا (زیادہ صحت مند پودے)

**حل:**
```python
# مثال: آؤٹ لائرز ہٹانا
from scipy import stats

def remove_outliers(df, columns, threshold=3):
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        df_clean = df_clean[z_scores < threshold]
    return df_clean

# آؤٹ لائرز ہٹائیں
data_no_outliers = remove_outliers(data_clean, ['temperature', 'humidity'])
print(f"آؤٹ لائرز ہٹانے سے پہلے: {len(data_clean)} ریکارڈز")
print(f"آؤٹ لائرز ہٹانے کے بعد: {len(data_no_outliers)} ریکارڈز")
```

### اوور فٹنگ

**چیلنجز:**
- پیچیدہ ماڈلز فیلڈ ڈیٹا پر اچھی پرفارمنس نہیں کرتے
- محدود ٹریننگ ڈیٹا

**حل:**
```python
# مثال: ریگولرائزیشن
from sklearn.linear_model import Ridge

# مختلف الفا ویلیوز آزمائیں
alphas = [0.1, 1.0, 10.0, 100.0]
best_alpha = None
best_score = float('inf')

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    scores = cross_val_score(ridge_model, X, y, cv=3, 
                           scoring='neg_mean_absolute_error')
    mean_score = -scores.mean()
    
    if mean_score < best_score:
        best_score = mean_score
        best_alpha = alpha

print(f"بہترین الفا: {best_alpha}, MAE: {best_score:.3f}")
```

## اگلے اقدامات

- [ڈیٹا پری پروسیسنگ](/docs/module-1/data-preprocessing)
- [ریگریشن ماڈلز](/docs/module-1/regression-models)

## خلاصہ

مشین لرننگ پودوں کی سائنس کو تبدیل کر رہی ہے:
- سپروائزڈ لرننگ پیش گوئیاں کے لیے
- ان سپروائزڈ لرننگ نمونے دریافت کرنے کے لیے
- ریئنفورسمنٹ لرننگ بہترین کاری کے لیے

زرعی چیلنجز کے لیے عملی ML تکنیک سیکھیں!