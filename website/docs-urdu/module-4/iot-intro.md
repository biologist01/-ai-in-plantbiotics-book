---
sidebar_position: 1
---

# IoT اور سمارٹ زراعت کا تعارف

## تعارف

انٹرنیٹ آف تھنگز (IoT) زراعت میں انقلاب لا رہا ہے۔ سینسرز، ڈیٹا، اور AI مل کر پریسیژن فارمنگ کو ممکن بناتے ہیں 🌱📡۔

## IoT کیا ہے؟

IoT آلات کا نیٹورک ہے جو:
- سینسرز سے ڈیٹا جمع کرتا ہے
- انٹرنیٹ پر بھیجتا ہے
- کلاؤڈ میں تجزیہ کرتا ہے
- خودکار فیصلے کرتا ہے

## زرعی IoT آرکیٹیکچر

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   سینسرز    │ -> │   گیٹ وے   │ -> │   کلاؤڈ    │
└─────────────┘    └─────────────┘    └─────────────┘
      │                  │                   │
   مٹی، موسم        ڈیٹا اکٹھا          ML ماڈلز
   پودے، آب         پروسیسنگ            ڈیش بورڈ
```

## عام زرعی سینسرز

| سینسر | پیمائش | استعمال |
|-------|--------|--------|
| DHT22 | درجہ حرارت، نمی | ماحول |
| مٹی نمی | مٹی کی نمی | آبپاشی |
| NDVI | پودوں کی صحت | فصل |
| pH | مٹی کی تیزابیت | زرخیزی |

## پہلا IoT سیٹ اپ

### ہارڈویئر

```python
# Raspberry Pi / ESP32 کے ساتھ سینسر پڑھنا

import time

class MockSensor:
    """
    ٹیسٹنگ کے لیے مصنوعی سینسر
    """
    def __init__(self, name):
        self.name = name
    
    def read_temperature(self):
        import random
        return 20 + random.uniform(-5, 10)
    
    def read_humidity(self):
        import random
        return 60 + random.uniform(-20, 20)

# سینسر آبجیکٹ
sensor = MockSensor("DHT22")

# ریڈنگز
while True:
    temp = sensor.read_temperature()
    humidity = sensor.read_humidity()
    
    print(f"درجہ حرارت: {temp:.1f}°C")
    print(f"نمی: {humidity:.1f}%")
    print("-" * 30)
    
    time.sleep(5)  # 5 سیکنڈ انتظار
```

### مٹی سینسر

```python
class SoilMoistureSensor:
    def __init__(self, pin=None):
        self.pin = pin
        self.dry_value = 1023
        self.wet_value = 300
    
    def read_raw(self):
        # ہارڈویئر سے پڑھیں (مصنوعی)
        import random
        return random.randint(300, 1023)
    
    def read_percentage(self):
        raw = self.read_raw()
        percentage = (self.dry_value - raw) / (self.dry_value - self.wet_value)
        return max(0, min(100, percentage * 100))

# استعمال
soil_sensor = SoilMoistureSensor()
moisture = soil_sensor.read_percentage()
print(f"مٹی کی نمی: {moisture:.1f}%")
```

## ڈیٹا لاگنگ

```python
import json
from datetime import datetime

class DataLogger:
    def __init__(self, filename='sensor_data.json'):
        self.filename = filename
        self.data = []
    
    def log(self, readings):
        """
        ریڈنگز محفوظ کریں
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'readings': readings
        }
        self.data.append(entry)
        
        # فائل میں لکھیں
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get_recent(self, n=10):
        """
        حالیہ ریڈنگز حاصل کریں
        """
        return self.data[-n:]

# استعمال
logger = DataLogger()

readings = {
    'temperature': 25.5,
    'humidity': 65.2,
    'soil_moisture': 45.8
}

logger.log(readings)
print("ڈیٹا محفوظ ہو گیا!")
```

## MQTT پروٹوکول

```python
# MQTT سے ڈیٹا بھیجنا

import json

class SimpleMQTT:
    """
    سادہ MQTT کلائنٹ (ڈیمو)
    """
    def __init__(self, broker, port=1883):
        self.broker = broker
        self.port = port
        self.connected = False
    
    def connect(self):
        print(f"{self.broker}:{self.port} سے جڑ رہے ہیں...")
        self.connected = True
        print("کامیابی سے جڑ گئے!")
    
    def publish(self, topic, message):
        if not self.connected:
            print("پہلے کنیکٹ کریں!")
            return
        
        if isinstance(message, dict):
            message = json.dumps(message)
        
        print(f"ٹاپک: {topic}")
        print(f"پیغام: {message}")
        print("بھیج دیا گیا!")

# استعمال
mqtt = SimpleMQTT("broker.example.com")
mqtt.connect()

sensor_data = {
    'device_id': 'farm_sensor_01',
    'temperature': 26.5,
    'humidity': 58.3,
    'soil_moisture': 42.1
}

mqtt.publish('farm/sensors/field1', sensor_data)
```

## ریئل ٹائم ڈیش بورڈ

```python
# Streamlit ڈیش بورڈ (demo)

import random
from datetime import datetime, timedelta

def generate_sensor_data(hours=24):
    """
    ڈیمو ڈیٹا بنائیں
    """
    data = []
    base_time = datetime.now() - timedelta(hours=hours)
    
    for i in range(hours * 12):  # ہر 5 منٹ
        timestamp = base_time + timedelta(minutes=i*5)
        
        entry = {
            'timestamp': timestamp.isoformat(),
            'temperature': 20 + 10 * (0.5 + 0.5 * 
                          (timestamp.hour / 24)) + random.uniform(-2, 2),
            'humidity': 60 + random.uniform(-15, 15),
            'soil_moisture': 50 + random.uniform(-10, 10)
        }
        data.append(entry)
    
    return data

# ڈیٹا بنائیں
sensor_history = generate_sensor_data()
print(f"کل ریکارڈز: {len(sensor_history)}")
print(f"آخری ریڈنگ: {sensor_history[-1]}")
```

## الرٹس اور نوٹیفیکیشنز

```python
class AlertSystem:
    def __init__(self):
        self.thresholds = {
            'temperature_high': 35,
            'temperature_low': 5,
            'soil_moisture_low': 20,
            'humidity_high': 90
        }
        self.alerts = []
    
    def check(self, readings):
        """
        الرٹس چیک کریں
        """
        current_alerts = []
        
        if readings.get('temperature', 0) > self.thresholds['temperature_high']:
            current_alerts.append('⚠️ درجہ حرارت بہت زیادہ!')
        
        if readings.get('temperature', 100) < self.thresholds['temperature_low']:
            current_alerts.append('❄️ درجہ حرارت بہت کم!')
        
        if readings.get('soil_moisture', 100) < self.thresholds['soil_moisture_low']:
            current_alerts.append('💧 مٹی کو پانی چاہیے!')
        
        self.alerts.extend(current_alerts)
        return current_alerts

# استعمال
alert_system = AlertSystem()

test_readings = {'temperature': 38, 'soil_moisture': 15}
alerts = alert_system.check(test_readings)

for alert in alerts:
    print(alert)
```

## خلاصہ

- IoT زراعت کو ڈیجیٹل بناتا ہے
- سینسرز ڈیٹا جمع کرتے ہیں
- ڈیٹا لاگنگ ضروری ہے
- الرٹس وقت پر خبردار کرتے ہیں

## اگلے اقدامات

- [سینسر نیٹورکس](/docs/module-4/sensor-networks)
