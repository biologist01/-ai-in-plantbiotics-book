---
sidebar_position: 3
---

# ماڈیول 1: پودوں کے تجزیہ کے لیے کلاسفیکیشن ماڈلز

## پودوں کی بائیوٹیکنالوجی میں کلاسفیکیشن کا تعارف 🌱

پودوں کی بائیوٹیکنالوجی میں مصنوعی ذہانت (AI) کے اطلاق نے اس فیلڈ کو تبدیل کر دیا ہے، جس سے پودوں کی اقسام، بیماری کی تشخیص، اور صحت کی حیثیت کی پیش گوئی کا زیادہ درست اور موثر تجزیہ ممکن ہوا ہے۔ کلاسفیکیشن ماڈلز اس انقلاب کا ایک اہم حصہ ہیں، جو ریسرچرز اور کسانوں کو فصل کے انتظام، کیڑوں کے کنٹرول، اور پیداوار کی بہترین کاری کے بارے میں معلومات پر مبنی فیصلے کرنے کی اجازت دیتے ہیں۔ اس ماڈیول میں، ہم کلاسفیکیشن ماڈلز کی دنیا میں گہرائی سے جائیں گے، ان کے اطلاقات، بنیادی تصورات، اور پودوں کی بائیوٹیکنالوجی میں عملی نفاذ کو دریافت کریں گے۔

### حقیقی دنیا کی تحریک

ایک ایسا منظر نامے کا تصور کریں جہاں ایک کسان اپنی گندم کی فصل کو متاثر کرنے والی بیماری کو تیزی سے پہچان سکتا ہے، جس سے وقت پر مداخلت اور پیداوار کے نقصان کو کم کیا جا سکتا ہے۔ یا، ایک ریسرچر کا تصور کریں جو ایک دور دراز علاقے میں پودوں کی اقسام کو درست طریقے سے کلاسفائی کر سکتا ہے، جس سے نئی اقسام کی دریافت اور تحفظ کے اقدامات آسان ہو جاتے ہیں۔ یہ منظر نامے اب کلاسفیکیشن ماڈلز کی طاقت کی بدولت پودوں کی بائیوٹیکنالوجی میں ممکن ہیں۔

## زراعت میں بنیادی تصورات: کلاسفیکیشن

کلاسفیکیشن ماڈلز کو وسیع طور پر دو اقسام میں تقسیم کیا جا سکتا ہے: بائنری اور ملٹی کلاس کلاسفیکیشن۔

### بائنری کلاسفیکیشن

بائنری کلاسفیکیشن میں دو کلاسز میں سے ایک کی پیش گوئی شامل ہے، جیسے صحت مند بمقابلہ بیمار پودے یا گھاس بمقابلہ فصل۔ اس قسم کی کلاسفیکیشن کا عام طور پر بیماری کی تشخیص اور گھاس کے انتظام میں استعمال ہوتا ہے۔

### ملٹی کلاس کلاسفیکیشن

ملٹی کلاس کلاسفیکیشن میں متعدد کلاسز میں سے ایک کی پیش گوئی شامل ہے، جیسے مختلف پودوں کی اقسام یا بیماری کی اقسام۔ اس قسم کی کلاسفیکیشن پودوں کی اقسام کی پہچان اور بیماری کی تشخیص میں مفید ہے۔

## پودوں کی کلاسفیکیشن کے لیے ڈیسیژن ٹریز اور رینڈم فارسٹس

ڈیسیژن ٹریز اور رینڈم فارسٹس پودوں کی بائیوٹیکنالوجی میں کلاسفیکیشن کے کاموں کے لیے استعمال ہونے والے مقبول مشین لرننگ الگورتھم ہیں۔

### ڈیسیژن ٹریز

ڈیسیژن ٹریز سادہ، لیکن طاقتور ماڈلز ہیں جو ڈیٹا کو فیچر ویلیوز کی بنیاد پر بار بار چھوٹے سب سیٹس میں تقسیم کرتے ہوئے کام کرتے ہیں۔ وہ آسانی سے تشریح کیے جا سکتے ہیں اور بائنری اور ملٹی کلاس دونوں کلاسفیکیشن کے کاموں کو ہینڈل کر سکتے ہیں۔

```python
# ضروری لائبریریز امپورٹ کریں
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# پودوں کے ڈیٹا کی مثال (مصنوعی)
np.random.seed(42)
n_samples = 200

# فیچرز: پتی کی لمبائی، پتی کی چوڑائی، پھولوں کی تعداد، اونچائی
plant_features = np.random.rand(n_samples, 4) * 10

# لیبلز: 0 = ٹماٹر، 1 = گندم، 2 = مکئی
plant_labels = np.random.choice([0, 1, 2], n_samples)

# ڈیٹا کو ٹریننگ اور ٹیسٹنگ سیٹس میں تقسیم کریں
X_train, X_test, y_train, y_test = train_test_split(
    plant_features, plant_labels, test_size=0.2, random_state=42
)

# ڈیسیژن ٹری کلاسفائیر ٹرین کریں
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# ماڈل کی تشخیص کریں
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"ڈیسیژن ٹری درستگی: {accuracy:.2f}")

# تفصیلی رپورٹ
print("کلاسفیکیشن رپورٹ:")
print(classification_report(y_test, y_pred, target_names=['ٹماٹر', 'گندم', 'مکئی']))
```

### رینڈم فارسٹس

رینڈم فارسٹس متعدد ڈیسیژن ٹریز کا مجموعہ ہیں جو مل کر پیش گوئی کرتے ہیں۔ یہ تکنیک اوور فٹنگ کو کم کرتی ہے اور عام طور پر ڈیسیژن ٹریز سے بہتر کارکردگی دکھاتی ہے۔

```python
from sklearn.ensemble import RandomForestClassifier

# رینڈم فارسٹ کلاسفائیر
rf_clf = RandomForestClassifier(
    n_estimators=100,  # درختوں کی تعداد
    max_depth=5,       # درخت کی زیادہ سے زیادہ گہرائی
    random_state=42
)

rf_clf.fit(X_train, y_train)

# پیش گوئی اور تشخیص
rf_pred = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"رینڈم فارسٹ درستگی: {rf_accuracy:.2f}")

# فیچر اہمیت
feature_names = ['پتی کی لمبائی', 'پتی کی چوڑائی', 'پھولوں کی تعداد', 'اونچائی']
feature_importance = pd.DataFrame({
    'فیچر': feature_names,
    'اہمیت': rf_clf.feature_importances_
}).sort_values('اہمیت', ascending=False)

print("فیچر اہمیت:")
print(feature_importance)
```

## سپورٹ ویکٹر مشینز (SVM) برائے پودوں کی کلاسفیکیشن

SVM ایک طاقتور کلاسفیکیشن الگورتھم ہے جو ڈیٹا پوائنٹس کو کلاسز میں تقسیم کرنے کے لیے ایک ہائپر پلین تلاش کرتا ہے۔ یہ پیچیدہ غیر لینئر تعلقات کو ہینڈل کرنے کے لیے کرنل فنکشنز استعمال کرتا ہے۔

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ڈیٹا اسکیل کریں (SVM کے لیے ضروری)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM کلاسفائیر
svm_clf = SVC(
    kernel='rbf',      # Radial Basis Function کرنل
    C=1.0,            # ریگولرائزیشن پارامیٹر
    gamma='scale',    # کرنل کو ایفیسینٹ
    random_state=42
)

svm_clf.fit(X_train_scaled, y_train)

# پیش گوئی اور تشخیص
svm_pred = svm_clf.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)

print(f"SVM درستگی: {svm_accuracy:.2f}")
```

## پودوں کی بیماری کی تشخیص کے لیے کلاسفیکیشن

### مثال: پتی کی بیماری کی شناخت

```python
# پتی کی بیماری کے ڈیٹا کی مثال
leaf_data = pd.DataFrame({
    'red_pixels': np.random.uniform(0, 1, 150),
    'green_pixels': np.random.uniform(0, 1, 150),
    'blue_pixels': np.random.uniform(0, 1, 150),
    'texture_roughness': np.random.uniform(0, 10, 150),
    'spot_size': np.random.uniform(0, 5, 150),
    'disease': np.random.choice(['صحت مند', 'زنگ', 'جھلسا', 'پاؤڈری ملڈیو'], 150)
})

# فیچرز اور لیبلز
X_leaf = leaf_data[['red_pixels', 'green_pixels', 'blue_pixels', 
                    'texture_roughness', 'spot_size']]
y_leaf = leaf_data['disease']

# لیبل کو عددی میں تبدیل کریں
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_leaf_encoded = le.fit_transform(y_leaf)

# ٹرین/ٹیسٹ تقسیم
X_train_leaf, X_test_leaf, y_train_leaf, y_test_leaf = train_test_split(
    X_leaf, y_leaf_encoded, test_size=0.2, random_state=42
)

# رینڈم فارسٹ سے ٹرین کریں
disease_clf = RandomForestClassifier(n_estimators=100, random_state=42)
disease_clf.fit(X_train_leaf, y_train_leaf)

# پیش گوئی
disease_pred = disease_clf.predict(X_test_leaf)
disease_accuracy = accuracy_score(y_test_leaf, disease_pred)

print(f"بیماری کی تشخیص درستگی: {disease_accuracy:.2f}")

# نئے پتے پر پیش گوئی کی مثال
new_leaf = pd.DataFrame({
    'red_pixels': [0.8],
    'green_pixels': [0.3],
    'blue_pixels': [0.2],
    'texture_roughness': [7.5],
    'spot_size': [3.2]
})

prediction = disease_clf.predict(new_leaf)
predicted_disease = le.inverse_transform(prediction)
print(f"پتے کی بیماری: {predicted_disease[0]}")
```

## نیورل نیٹورکس اور ڈیپ لرننگ برائے پودوں کی کلاسفیکیشن

ڈیپ لرننگ پیچیدہ پیٹرنز کو سیکھنے کے لیے متعدد پرتوں والے نیورل نیٹورکس استعمال کرتا ہے۔ یہ امیج کلاسفیکیشن جیسے کاموں کے لیے خاص طور پر مفید ہے۔

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# ملٹی کلاس کے لیے لیبلز کو one-hot انکوڈ کریں
y_train_cat = to_categorical(y_train_leaf, num_classes=4)
y_test_cat = to_categorical(y_test_leaf, num_classes=4)

# سادہ نیورل نیٹورک بنائیں
def create_plant_classifier(input_dim, num_classes):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ماڈل بنائیں اور ٹرین کریں
nn_model = create_plant_classifier(X_train_leaf.shape[1], 4)

history = nn_model.fit(
    X_train_leaf, y_train_cat,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

# تشخیص
nn_loss, nn_accuracy = nn_model.evaluate(X_test_leaf, y_test_cat, verbose=0)
print(f"نیورل نیٹورک درستگی: {nn_accuracy:.2f}")

# تربیت کی تاریخ پلٹ کریں
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='ٹریننگ درستگی')
plt.plot(history.history['val_accuracy'], label='ویلیڈیشن درستگی')
plt.title('ماڈل درستگی')
plt.xlabel('ایپوک')
plt.ylabel('درستگی')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='ٹریننگ نقصان')
plt.plot(history.history['val_loss'], label='ویلیڈیشن نقصان')
plt.title('ماڈل نقصان')
plt.xlabel('ایپوک')
plt.ylabel('نقصان')
plt.legend()

plt.tight_layout()
plt.show()
```

## کلاسفیکیشن میٹرکس اور ماڈل کی تشخیص

کلاسفیکیشن ماڈلز کی کارکردگی کا اندازہ کرنے کے لیے، ہم مختلف میٹرکس استعمال کرتے ہیں۔

### کنفیوژن میٹرکس

کنفیوژن میٹرکس حقیقی اور پیش گوئی کردہ کلاسز کا موازنہ دکھاتا ہے۔

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# کنفیوژن میٹرکس بنائیں
cm = confusion_matrix(y_test_leaf, disease_pred)

# کنفیوژن میٹرکس دکھائیں
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=le.classes_
)
disp.plot(cmap='Blues')
plt.title('کنفیوژن میٹرکس - پودوں کی بیماری')
plt.show()

# میٹرکس ویلیوز پرنٹ کریں
print("کنفیوژن میٹرکس:")
print(cm)
```

### پریسیژن، ریکال، اور F1 اسکور

```python
from sklearn.metrics import precision_recall_fscore_support

# ہر کلاس کے لیے میٹرکس حساب کریں
precision, recall, f1, support = precision_recall_fscore_support(
    y_test_leaf, disease_pred, average=None
)

# نتائج دکھائیں
for i, class_name in enumerate(le.classes_):
    print(f"{class_name}:")
    print(f"  پریسیژن: {precision[i]:.2f}")
    print(f"  ریکال: {recall[i]:.2f}")
    print(f"  F1 اسکور: {f1[i]:.2f}")
    print(f"  سپورٹ: {support[i]}")
    print()

# میکرو اور میکرو اوسط
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_test_leaf, disease_pred, average='macro'
)

print("میکرو اوسط:")
print(f"  پریسیژن: {macro_precision:.2f}")
print(f"  ریکال: {macro_recall:.2f}")
print(f"  F1 اسکور: {macro_f1:.2f}")
```

## کلاسفیکیشن چیلنجز اور حل

### غیر متوازن ڈیٹا

زرعی ڈیٹاسیٹ اکثر غیر متوازن ہوتے ہیں (زیادہ صحت مند پودے، کم بیمار پودے)۔

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# SMOTE سے اوور سیمیپلنگ
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_leaf, y_train_leaf)

print(f"اصل ٹریننگ ڈیٹا: {X_train_leaf.shape[0]} نمونے")
print(f"ریسیمپلڈ ڈیٹا: {X_resampled.shape[0]} نمونے")

# ریسیمپلڈ ڈیٹا پر ماڈل ٹرین کریں
resampled_clf = RandomForestClassifier(n_estimators=100, random_state=42)
resampled_clf.fit(X_resampled, y_resampled)

resampled_pred = resampled_clf.predict(X_test_leaf)
resampled_accuracy = accuracy_score(y_test_leaf, resampled_pred)

print(f"ریسیمپلڈ ماڈل درستگی: {resampled_accuracy:.2f}")
```

### اوور فٹنگ کو روکنا

```python
# کراس ویلیڈیشن
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    disease_clf, X_leaf, y_leaf_encoded, cv=5, scoring='accuracy'
)

print(f"کراس ویلیڈیشن درستگی سکورز: {cv_scores}")
print(f"اوسط درستگی: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# ہائپر پارامیٹر ٹیوننگ
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_leaf, y_train_leaf)

print(f"بہترین پارامیٹرز: {grid_search.best_params_}")
print(f"بہترین درستگی: {grid_search.best_score_:.2f}")
```

## ریئل ورلڈ ایپلیکیشنز

### پودوں کی اقسام کی خودکار شناخت

```python
def identify_plant_species(features):
    """
    پودوں کی اقسام کی شناخت
    """
    # یہاں حقیقی ماڈل استعمال کریں
    species = ['ٹماٹر', 'گندم', 'مکئی', 'چاول', 'سورج مکھی']
    
    # سادہ منطق (حقیقی دنیا میں ٹرینڈ ماڈل استعمال کریں)
    if features[0] > 0.7:  # پتی کی لمبائی
        return 'سورج مکھی'
    elif features[1] > 0.6:  # پتی کی چوڑائی
        return 'گندم'
    elif features[2] > 7:  # پھولوں کی تعداد
        return 'ٹماٹر'
    else:
        return 'مکئی'

# مثال
sample_features = [0.5, 0.4, 5, 15]  # لمبائی، چوڑائی، پھول، اونچائی
identified_species = identify_plant_species(sample_features)
print(f"پودہ کی قسم: {identified_species}")
```

### بیماری کی ابتدائی تشخیص سسٹم

```python
def early_disease_detection(sensor_data, image_features):
    """
    بیماری کی ابتدائی تشخیص
    """
    # سینسر ڈیٹا اور امیج فیچرز کا امتزاج
    combined_features = sensor_data + image_features
    
    # بیماری کی شدت کی پیش گوئی (0-1)
    severity_score = min(1.0, sum(combined_features) / len(combined_features))
    
    if severity_score < 0.3:
        status = 'صحت مند'
        action = 'کوئی کارروائی نہیں'
    elif severity_score < 0.6:
        status = 'مشکوک'
        action = 'نگرانی جاری رکھیں'
    else:
        status = 'بیمار'
        action = 'فوری علاج شروع کریں'
    
    return {
        'severity_score': severity_score,
        'status': status,
        'recommended_action': action
    }

# مثال
sensor_readings = [0.8, 0.6, 0.9]  # نمی، درجہ حرارت، روشنی
image_analysis = [0.7, 0.4, 0.8]  # رنگ، ٹکسچر، سپاٹس

diagnosis = early_disease_detection(sensor_readings, image_analysis)
print(f"تشخیص: {diagnosis['status']}")
print(f"شدت: {diagnosis['severity_score']:.2f}")
print(f"کارروائی: {diagnosis['recommended_action']}")
```

## خلاصہ

کلاسفیکیشن ماڈلز پودوں کی بائیوٹیکنالوجی میں ایک ضروری ٹول ہیں:

- **ڈیسیژن ٹریز**: سادہ اور تشریح پذیر
- **رینڈم فارسٹس**: زیادہ درست اور مستحکم
- **SVM**: پیچیدہ تعلقات کے لیے
- **نیورل نیٹورکس**: امیج کلاسفیکیشن کے لیے

کلاسفیکیشن ماڈلز کا صحیح استعمال فصل کے انتظام کو بہتر بنا سکتا ہے اور زرعی کارکردگی کو بڑھا سکتا ہے۔

## اگلے اقدامات

- [ریگریشن ماڈلز](/docs/module-1/regression-models)
- [ٹائم سیریز تجزیہ](/docs/module-1/time-series)

## عملی مشق

1. اپنے پودوں کے ڈیٹا پر کلاسفیکیشن ماڈل ٹرین کریں
2. مختلف کلاسفیکیشن تکنیکوں کا موازنہ کریں
3. کنفیوژن میٹرکس کا تجزیہ کریں اور غلط پیش گوئیاں کی وجوہات دریافت کریں