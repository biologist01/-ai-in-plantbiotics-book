---
sidebar_position: 2
---

# تصویر کا حصول اور پری پروسیسنگ

## تعارف

پودوں کے تجزیے کے لیے تصویری پری پروسیسنگ بہت اہم ہے۔ اس سبق میں آپ فلٹرنگ، سیگمنٹیشن، فیچر ایکسٹریکشن، اور پس منظر ہٹانے کی تکنیکیں سیکھیں گے 🌱۔

## شور کم کرنا

```python
import cv2
import numpy as np

# تصویر لوڈ کریں
img = cv2.imread('noisy_leaf.jpg')

# گاؤسین بلر
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# میڈین فلٹر (نمک اور مرچ شور کے لیے)
median = cv2.medianBlur(img, 5)

# بائی لیٹرل فلٹر (کناروں کو محفوظ رکھتا ہے)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
```

## پس منظر ہٹانا

```python
import cv2
import numpy as np

def remove_background(image_path):
    img = cv2.imread(image_path)
    
    # گرے اسکیل میں تبدیل کریں
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # تھریشولڈنگ
    _, thresh = cv2.threshold(gray, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # کونٹورز تلاش کریں
    contours, _ = cv2.findContours(thresh, 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    # سب سے بڑا کونٹور (پتا)
    largest = max(contours, key=cv2.contourArea)
    
    # ماسک بنائیں
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    
    # نتیجہ
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

# استعمال
clean_leaf = remove_background('leaf_with_background.jpg')
```

## رنگ پر مبنی سیگمنٹیشن

```python
def color_segmentation(img):
    # HSV میں تبدیل کریں
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # صحت مند پتے (سبز)
    healthy_mask = cv2.inRange(hsv, 
                                np.array([35, 40, 40]), 
                                np.array([85, 255, 255]))
    
    # بیمار پتے (بھورا/پیلا)
    diseased_mask = cv2.inRange(hsv, 
                                 np.array([10, 40, 40]), 
                                 np.array([35, 255, 255]))
    
    return healthy_mask, diseased_mask
```

## مورفولوجیکل آپریشنز

```python
# کرنل بنائیں
kernel = np.ones((5, 5), np.uint8)

# ایروژن - چھوٹے شور ہٹائیں
erosion = cv2.erode(mask, kernel, iterations=1)

# ڈائلیشن - سوراخ بھریں
dilation = cv2.dilate(mask, kernel, iterations=1)

# اوپننگ (ایروژن + ڈائلیشن)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# کلوزنگ (ڈائلیشن + ایروژن)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

## فیچر ایکسٹریکشن

```python
def extract_features(img, mask):
    # رنگ کے اعدادوشمار
    masked = cv2.bitwise_and(img, img, mask=mask)
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    
    # ہسٹوگرام
    hist = cv2.calcHist([hsv], [0, 1], mask, [50, 60], [0, 180, 0, 256])
    
    # کونٹور فیچرز
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # شیپ فیکٹر
    circularity = 4 * np.pi * area / (perimeter ** 2)
    
    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'histogram': hist.flatten()
    }
```

## عملی پروجیکٹ: پتے کا رقبہ ناپنا

```python
def measure_leaf_area(image_path, pixels_per_cm=100):
    img = cv2.imread(image_path)
    
    # سیگمنٹیشن
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([95, 255, 255]))
    
    # شور ہٹائیں
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # پکسل گنیں
    leaf_pixels = cv2.countNonZero(mask)
    
    # رقبہ حساب کریں (cm²)
    area_cm2 = leaf_pixels / (pixels_per_cm ** 2)
    
    return area_cm2

# استعمال
area = measure_leaf_area('leaf.jpg')
print(f"پتے کا رقبہ: {area:.2f} cm²")
```

## خلاصہ

| تکنیک | مقصد |
|-------|------|
| بلرنگ | شور کم کرنا |
| تھریشولڈنگ | پس منظر ہٹانا |
| مورفولوجی | شیپ صاف کرنا |
| کونٹورز | شکل نکالنا |

## اگلے اقدامات

- [ڈیپ لرننگ CNN](/docs/module-2/deep-learning-cnn) - بیماری کا پتہ لگانا
