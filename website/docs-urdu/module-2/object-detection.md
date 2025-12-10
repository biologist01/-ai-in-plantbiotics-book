---
sidebar_position: 4
---

# پھلوں اور پھولوں کے لیے آبجیکٹ ڈیٹیکشن

## تعارف

آبجیکٹ ڈیٹیکشن ماڈلز پھلوں کی گنتی، پھولوں کا پتہ لگانے، اور خودکار کٹائی کے لیے استعمال ہوتے ہیں۔ اس سبق میں YOLO اور Faster R-CNN سیکھیں گے 🍅۔

## آبجیکٹ ڈیٹیکشن بمقابلہ کلاسیفیکیشن

| ٹاسک | آؤٹ پٹ |
|------|--------|
| کلاسیفیکیشن | کلاس لیبل |
| ڈیٹیکشن | بائنڈنگ باکسز + لیبلز |
| سیگمنٹیشن | پکسل لیول ماسک |

## YOLOv8 کے ساتھ ڈیٹیکشن

```python
from ultralytics import YOLO

# ماڈل لوڈ کریں
model = YOLO('yolov8n.pt')

# تصویر پر پیش گوئی
results = model('tomatoes.jpg')

# نتائج دکھائیں
results[0].show()

# باکسز نکالیں
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    confidence = box.conf[0]
    class_id = box.cls[0]
    print(f"کلاس: {class_id}, اعتماد: {confidence:.2f}")
```

## کسٹم ڈیٹاسیٹ پر ٹریننگ

### ڈیٹا کی تیاری (YOLO فارمیٹ)

```yaml
# dataset.yaml
path: ./data
train: images/train
val: images/val

names:
  0: tomato_ripe
  1: tomato_unripe
  2: tomato_flower
```

### اینوٹیشن (labels/image.txt)
```
# class x_center y_center width height (normalized)
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.15
```

### ماڈل ٹرین کریں

```python
from ultralytics import YOLO

# نیا ماڈل بنائیں
model = YOLO('yolov8n.pt')

# ٹرین کریں
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='tomato_detector'
)

# بہترین ماڈل لوڈ کریں
best_model = YOLO('runs/detect/tomato_detector/weights/best.pt')
```

## پھلوں کی گنتی

```python
def count_fruits(image_path, model):
    results = model(image_path)
    
    counts = {}
    for box in results[0].boxes:
        class_name = results[0].names[int(box.cls[0])]
        counts[class_name] = counts.get(class_name, 0) + 1
    
    return counts

# استعمال
model = YOLO('tomato_detector.pt')
fruit_counts = count_fruits('field.jpg', model)
print(f"پکے ٹماٹر: {fruit_counts.get('tomato_ripe', 0)}")
print(f"کچے ٹماٹر: {fruit_counts.get('tomato_unripe', 0)}")
```

## پختگی کا پتہ لگانا

```python
import cv2
import numpy as np

def detect_ripeness(image_path, model):
    results = model(image_path)
    img = cv2.imread(image_path)
    
    ripe = 0
    unripe = 0
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        
        # رنگ کے مطابق باکس
        if class_id == 0:  # پکا
            color = (0, 255, 0)  # سبز
            ripe += 1
        else:  # کچا
            color = (0, 0, 255)  # سرخ
            unripe += 1
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    return img, ripe, unripe

# استعمال
result_img, ripe, unripe = detect_ripeness('tomatoes.jpg', model)
print(f"پکے: {ripe}, کچے: {unripe}")
cv2.imwrite('detected.jpg', result_img)
```

## ویڈیو پر ڈیٹیکشن

```python
import cv2
from ultralytics import YOLO

model = YOLO('fruit_detector.pt')

cap = cv2.VideoCapture('farm_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    annotated = results[0].plot()
    
    cv2.imshow('پھلوں کا پتہ لگانا', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## خلاصہ

| ماڈل | رفتار | درستگی |
|------|-------|--------|
| YOLOv8n | تیز ترین | اچھی |
| YOLOv8m | متوسط | بہتر |
| YOLOv8x | سست | بہترین |

## اگلے اقدامات

- [CV پروجیکٹ](/docs/module-2/cv-project) - فینوٹائپنگ سسٹم
