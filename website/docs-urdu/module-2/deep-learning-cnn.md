---
sidebar_position: 3
---

# پودوں کی بیماری کا پتہ لگانے کے لیے ڈیپ لرننگ

## تعارف

کنولیوشنل نیورل نیٹ ورکس (CNNs) پودوں کی بیماریوں کی درجہ بندی کے لیے انتہائی مؤثر ہیں۔ اس سبق میں آپ ٹرانسفر لرننگ اور ڈیٹا آگمنٹیشن سیکھیں گے 🌿۔

## CNN کی بنیادیں

CNN تین اہم پرتوں پر مشتمل ہے:
- **کنولیوشن**: فیچرز نکالنا
- **پولنگ**: سائز کم کرنا
- **فلی کنیکٹڈ**: درجہ بندی

## TensorFlow/Keras کے ساتھ CNN

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# سادہ CNN ماڈل
model = models.Sequential([
    # پہلی کنولیوشن پرت
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # دوسری کنولیوشن پرت
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # تیسری کنولیوشن پرت
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # فلیٹن اور ڈینس پرتیں
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 بیماریاں
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

## ٹرانسفر لرننگ

پہلے سے تربیت یافتہ ماڈلز استعمال کریں:

```python
from tensorflow.keras.applications import ResNet50, EfficientNetB0

# ResNet50 لوڈ کریں
base_model = ResNet50(weights='imagenet', 
                       include_top=False, 
                       input_shape=(224, 224, 3))

# بیس ماڈل فریز کریں
base_model.trainable = False

# نیا ماڈل بنائیں
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(38, activation='softmax')  # PlantVillage کلاسز
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

## ڈیٹا آگمنٹیشن

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# آگمنٹیشن سیٹ اپ
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ٹریننگ ڈیٹا لوڈ کریں
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

## مکمل ٹریننگ پائپ لائن

```python
# ماڈل ٹرین کریں
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)

# نتائج پلاٹ کریں
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='ٹریننگ')
plt.plot(history.history['val_accuracy'], label='ویلیڈیشن')
plt.title('ماڈل کی درستگی')
plt.xlabel('ایپاک')
plt.ylabel('درستگی')
plt.legend()
plt.show()
```

## Grad-CAM کے ساتھ تشریح

```python
import numpy as np
import tensorflow as tf

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()
```

## عملی پروجیکٹ: بیماری کی شناخت

```python
# ماڈل لوڈ کریں
model = tf.keras.models.load_model('plant_disease_model.h5')

# تصویر تیار کریں
def predict_disease(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    
    disease_names = ['صحت مند', 'زنگ', 'پاؤڈری مِلڈیو', 'لیف سپاٹ', ...]
    
    return disease_names[class_idx], confidence

# استعمال
disease, conf = predict_disease('test_leaf.jpg')
print(f"بیماری: {disease}, اعتماد: {conf:.2%}")
```

## خلاصہ

| ماڈل | فوائد | نقصانات |
|------|-------|---------|
| سادہ CNN | سمجھنے میں آسان | کم درستگی |
| ResNet | گہرا نیٹ ورک | زیادہ وسائل |
| EfficientNet | بہترین کارکردگی | پیچیدہ |

## اگلے اقدامات

- [آبجیکٹ ڈیٹیکشن](/docs/module-2/object-detection) - پھلوں کی گنتی
