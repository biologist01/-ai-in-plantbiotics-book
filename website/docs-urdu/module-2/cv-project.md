---
sidebar_position: 5
---

# منی پروجیکٹ: خودکار پودوں کی فینوٹائپنگ

## تعارف

اس پروجیکٹ میں آپ ایک مکمل خودکار فینوٹائپنگ سسٹم بنائیں گے جو پودے کی اونچائی، پتے کا رقبہ، رنگ تجزیہ، اور نشوونما کی ٹریکنگ کرے گا 🌱۔

## پروجیکٹ کا جائزہ

سسٹم کی خصوصیات:
- پودے کی خودکار سیگمنٹیشن
- اونچائی اور چوڑائی کی پیمائش
- پتوں کا رقبہ حساب کرنا
- صحت کا تجزیہ (رنگ کی بنیاد پر)
- وقت کے ساتھ نشوونما کی ٹریکنگ

## مرحلہ 1: پودے کی سیگمنٹیشن

```python
import cv2
import numpy as np

class PlantSegmenter:
    def __init__(self):
        self.lower_green = np.array([25, 40, 40])
        self.upper_green = np.array([95, 255, 255])
    
    def segment(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        
        # شور ہٹائیں
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def get_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        return sorted(contours, key=cv2.contourArea, reverse=True)
```

## مرحلہ 2: پیمائش نکالنا

```python
class PlantMeasurer:
    def __init__(self, pixels_per_cm=50):
        self.pixels_per_cm = pixels_per_cm
    
    def measure_height(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        height_cm = h / self.pixels_per_cm
        return height_cm
    
    def measure_width(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        width_cm = w / self.pixels_per_cm
        return width_cm
    
    def measure_leaf_area(self, mask):
        leaf_pixels = cv2.countNonZero(mask)
        area_cm2 = leaf_pixels / (self.pixels_per_cm ** 2)
        return area_cm2
    
    def count_leaves(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        # چھوٹے کونٹورز فلٹر کریں
        leaves = [c for c in contours if cv2.contourArea(c) > 500]
        return len(leaves)
```

## مرحلہ 3: رنگ تجزیہ

```python
class ColorAnalyzer:
    def analyze_health(self, image, mask):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # صرف ماسک والے علاقے کا تجزیہ
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        
        # سبز پکسلز کا فیصد
        green_mask = cv2.inRange(masked_hsv, 
                                  np.array([35, 40, 40]), 
                                  np.array([85, 255, 255]))
        
        # پیلے پکسلز (تناؤ کی علامت)
        yellow_mask = cv2.inRange(masked_hsv, 
                                   np.array([20, 40, 40]), 
                                   np.array([35, 255, 255]))
        
        total = cv2.countNonZero(mask)
        green = cv2.countNonZero(green_mask)
        yellow = cv2.countNonZero(yellow_mask)
        
        health_score = (green / total) * 100 if total > 0 else 0
        stress_score = (yellow / total) * 100 if total > 0 else 0
        
        return {
            'health_score': health_score,
            'stress_score': stress_score
        }
```

## مرحلہ 4: مکمل سسٹم

```python
class PlantPhenotyper:
    def __init__(self, pixels_per_cm=50):
        self.segmenter = PlantSegmenter()
        self.measurer = PlantMeasurer(pixels_per_cm)
        self.color_analyzer = ColorAnalyzer()
    
    def analyze(self, image_path):
        image = cv2.imread(image_path)
        
        # سیگمنٹیشن
        mask = self.segmenter.segment(image)
        contours = self.segmenter.get_contours(mask)
        
        if not contours:
            return None
        
        main_plant = contours[0]
        
        # پیمائش
        height = self.measurer.measure_height(main_plant)
        width = self.measurer.measure_width(main_plant)
        leaf_area = self.measurer.measure_leaf_area(mask)
        leaf_count = self.measurer.count_leaves(mask)
        
        # صحت تجزیہ
        health = self.color_analyzer.analyze_health(image, mask)
        
        return {
            'height_cm': round(height, 2),
            'width_cm': round(width, 2),
            'leaf_area_cm2': round(leaf_area, 2),
            'leaf_count': leaf_count,
            'health_score': round(health['health_score'], 1),
            'stress_score': round(health['stress_score'], 1)
        }

# استعمال
phenotyper = PlantPhenotyper(pixels_per_cm=50)
results = phenotyper.analyze('plant_day1.jpg')
print(f"اونچائی: {results['height_cm']} cm")
print(f"پتوں کا رقبہ: {results['leaf_area_cm2']} cm²")
print(f"صحت سکور: {results['health_score']}%")
```

## مرحلہ 5: نشوونما ٹریکنگ

```python
import pandas as pd
from datetime import datetime

class GrowthTracker:
    def __init__(self):
        self.history = []
    
    def record(self, measurements, date=None):
        if date is None:
            date = datetime.now()
        
        measurements['date'] = date
        self.history.append(measurements)
    
    def get_growth_rate(self):
        if len(self.history) < 2:
            return None
        
        df = pd.DataFrame(self.history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # روزانہ نشوونما کی شرح
        df['height_change'] = df['height_cm'].diff()
        df['days'] = df['date'].diff().dt.days
        df['growth_rate'] = df['height_change'] / df['days']
        
        return df['growth_rate'].mean()
    
    def plot_growth(self):
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(self.history)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot(df['date'], df['height_cm'], 'g-o')
        axes[0, 0].set_title('اونچائی')
        axes[0, 0].set_ylabel('cm')
        
        axes[0, 1].plot(df['date'], df['leaf_area_cm2'], 'b-o')
        axes[0, 1].set_title('پتوں کا رقبہ')
        axes[0, 1].set_ylabel('cm²')
        
        axes[1, 0].plot(df['date'], df['leaf_count'], 'r-o')
        axes[1, 0].set_title('پتوں کی تعداد')
        
        axes[1, 1].plot(df['date'], df['health_score'], 'g-o')
        axes[1, 1].set_title('صحت سکور')
        axes[1, 1].set_ylabel('%')
        
        plt.tight_layout()
        plt.savefig('growth_analysis.png')
        plt.show()
```

## خلاصہ

اس پروجیکٹ میں آپ نے سیکھا:
- پودے کی خودکار سیگمنٹیشن
- جسمانی پیمائش نکالنا
- صحت کا تجزیہ
- نشوونما کی ٹریکنگ

## اگلے اقدامات

- [جینومکس کا تعارف](/docs/module-3/genomics-intro) - AI اور جینز
