---
sidebar_position: 4
---

# سمارٹ آبپاشی سسٹم

## تعارف

سمارٹ آبپاشی پانی کی بچت اور فصل کی بہتر پیداوار کے لیے AI استعمال کرتی ہے۔ سینسرز، موسم ڈیٹا، اور ML سے خودکار آبپاشی کا فیصلہ کریں 💧🌱۔

## روایتی بمقابلہ سمارٹ آبپاشی

| پہلو | روایتی | سمارٹ |
|------|--------|-------|
| پانی کا استعمال | زیادہ | 30-50% کم |
| محنت | دستی | خودکار |
| درستگی | کم | زیادہ |
| فصل کی صحت | متغیر | بہتر |

## سینسر ڈیٹا جمع کرنا

```python
import numpy as np
from datetime import datetime, timedelta

class IrrigationSensors:
    def __init__(self, field_id):
        self.field_id = field_id
        self.sensors = {}
    
    def read_soil_moisture(self, depth='shallow'):
        """
        مٹی کی نمی پڑھیں
        """
        base = 50 if depth == 'shallow' else 60
        return base + np.random.uniform(-20, 20)
    
    def read_weather(self):
        """
        موسم کی معلومات
        """
        return {
            'temperature': 25 + np.random.uniform(-5, 10),
            'humidity': 60 + np.random.uniform(-20, 20),
            'wind_speed': 5 + np.random.uniform(0, 10),
            'solar_radiation': 500 + np.random.uniform(-200, 200)
        }
    
    def read_all(self):
        """
        تمام سینسرز پڑھیں
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'field_id': self.field_id,
            'soil_moisture_shallow': self.read_soil_moisture('shallow'),
            'soil_moisture_deep': self.read_soil_moisture('deep'),
            'weather': self.read_weather()
        }

# استعمال
sensors = IrrigationSensors('F001')
reading = sensors.read_all()
print(f"نمی (اوپر): {reading['soil_moisture_shallow']:.1f}%")
print(f"موسم: {reading['weather']}")
```

## Evapotranspiration حساب

```python
def calculate_eto(temperature, humidity, wind_speed, solar_radiation):
    """
    حوالہ بخارات-تعرق (FAO Penman-Monteith کا سادہ ورژن)
    """
    # سادہ فارمولا
    es = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))  # سیچوریشن
    ea = es * humidity / 100  # حقیقی
    vpd = es - ea  # ویپر پریشر ڈیفیسٹ
    
    # ETo (mm/day)
    eto = 0.0023 * (temperature + 17.8) * (solar_radiation / 2.45) ** 0.5 * vpd
    
    return max(0, eto)

def calculate_etc(eto, kc):
    """
    فصل بخارات-تعرق
    kc: فصل کا عامل
    """
    return eto * kc

# استعمال
weather = reading['weather']
eto = calculate_eto(
    weather['temperature'],
    weather['humidity'],
    weather['wind_speed'],
    weather['solar_radiation']
)

# گندم کے لیے
kc_wheat = 1.15
etc = calculate_etc(eto, kc_wheat)

print(f"ETo: {eto:.2f} mm/day")
print(f"ETc (گندم): {etc:.2f} mm/day")
```

## آبپاشی فیصلہ ماڈل

```python
class IrrigationDecisionModel:
    def __init__(self, crop_type='wheat'):
        self.crop_type = crop_type
        self.thresholds = {
            'wheat': {'low': 30, 'optimal': 50, 'high': 70},
            'rice': {'low': 60, 'optimal': 80, 'high': 95},
            'cotton': {'low': 25, 'optimal': 45, 'high': 65}
        }
        self.kc_values = {
            'wheat': 1.15,
            'rice': 1.2,
            'cotton': 1.15
        }
    
    def get_recommendation(self, soil_moisture, eto, forecast=None):
        """
        آبپاشی کی سفارش
        """
        threshold = self.thresholds[self.crop_type]
        
        # موجودہ حالت
        if soil_moisture < threshold['low']:
            status = 'critical'
            irrigate = True
        elif soil_moisture < threshold['optimal']:
            status = 'low'
            irrigate = True
        elif soil_moisture > threshold['high']:
            status = 'high'
            irrigate = False
        else:
            status = 'optimal'
            irrigate = False
        
        # پانی کی مقدار
        if irrigate:
            kc = self.kc_values[self.crop_type]
            etc = eto * kc
            water_needed = max(0, threshold['optimal'] - soil_moisture)
            # mm میں تبدیل
            irrigation_mm = water_needed * 0.3 + etc
        else:
            irrigation_mm = 0
        
        return {
            'status': status,
            'irrigate': irrigate,
            'water_mm': round(irrigation_mm, 1),
            'message': self._get_message(status, irrigation_mm)
        }
    
    def _get_message(self, status, water):
        messages = {
            'critical': f'⚠️ فوری آبپاشی! {water} mm',
            'low': f'💧 آبپاشی کریں: {water} mm',
            'optimal': '✅ نمی بہترین ہے',
            'high': '🌊 پانی کافی ہے'
        }
        return messages[status]

# استعمال
model = IrrigationDecisionModel('wheat')
recommendation = model.get_recommendation(
    soil_moisture=reading['soil_moisture_shallow'],
    eto=eto
)

print(f"\n{recommendation['message']}")
print(f"آبپاشی: {'ہاں' if recommendation['irrigate'] else 'نہیں'}")
```

## ML سے پیش گوئی

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def create_training_data(n_samples=1000):
    """
    ٹریننگ ڈیٹا بنائیں
    """
    np.random.seed(42)
    
    data = {
        'soil_moisture': np.random.uniform(10, 90, n_samples),
        'temperature': np.random.uniform(15, 40, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'eto': np.random.uniform(2, 8, n_samples),
        'days_since_irrigation': np.random.randint(0, 7, n_samples)
    }
    
    # لیبل: آبپاشی کریں یا نہیں
    data['irrigate'] = (
        (data['soil_moisture'] < 40) | 
        ((data['soil_moisture'] < 55) & (data['days_since_irrigation'] > 3))
    ).astype(int)
    
    return pd.DataFrame(data)

# ڈیٹا بنائیں
import pandas as pd
df = create_training_data()

# ماڈل
features = ['soil_moisture', 'temperature', 'humidity', 'eto', 'days_since_irrigation']
X = df[features]
y = df['irrigate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"ماڈل درستگی: {accuracy:.2%}")

# پیش گوئی
new_data = pd.DataFrame({
    'soil_moisture': [35],
    'temperature': [28],
    'humidity': [55],
    'eto': [5],
    'days_since_irrigation': [2]
})

prediction = clf.predict(new_data)[0]
probability = clf.predict_proba(new_data)[0][1]

print(f"\nپیش گوئی: {'آبپاشی کریں' if prediction else 'انتظار کریں'}")
print(f"امکان: {probability:.1%}")
```

## شیڈولنگ سسٹم

```python
from datetime import datetime, timedelta

class IrrigationScheduler:
    def __init__(self, field_id):
        self.field_id = field_id
        self.schedule = []
        self.history = []
    
    def add_event(self, date, duration_minutes, zone):
        """
        آبپاشی ایونٹ شامل کریں
        """
        event = {
            'id': len(self.schedule) + 1,
            'date': date,
            'duration': duration_minutes,
            'zone': zone,
            'status': 'scheduled'
        }
        self.schedule.append(event)
        return event
    
    def get_upcoming(self, days=7):
        """
        آنے والے ایونٹس
        """
        now = datetime.now()
        end = now + timedelta(days=days)
        
        upcoming = [
            e for e in self.schedule 
            if e['status'] == 'scheduled' and 
               now <= e['date'] <= end
        ]
        
        return sorted(upcoming, key=lambda x: x['date'])
    
    def auto_schedule(self, soil_moisture, eto, forecast):
        """
        خودکار شیڈولنگ
        """
        now = datetime.now()
        
        # فیصلہ کریں
        model = IrrigationDecisionModel('wheat')
        rec = model.get_recommendation(soil_moisture, eto)
        
        if rec['irrigate']:
            # بہترین وقت (صبح سویرے)
            next_morning = now.replace(hour=5, minute=0, second=0)
            if next_morning <= now:
                next_morning += timedelta(days=1)
            
            # مدت حساب کریں (1mm = ~10 minutes)
            duration = int(rec['water_mm'] * 10)
            
            event = self.add_event(next_morning, duration, 'zone1')
            return event
        
        return None

# استعمال
scheduler = IrrigationScheduler('F001')

# خودکار شیڈول
event = scheduler.auto_schedule(
    soil_moisture=35,
    eto=5,
    forecast=None
)

if event:
    print(f"✅ شیڈول: {event['date']} - {event['duration']} منٹ")
```

## والو کنٹرول

```python
class ValveController:
    def __init__(self, zones=4):
        self.zones = {f'zone{i}': False for i in range(1, zones+1)}
        self.flow_rate = 10  # لیٹر/منٹ
    
    def open_valve(self, zone):
        """
        والو کھولیں
        """
        if zone in self.zones:
            self.zones[zone] = True
            print(f"✅ {zone} والو کھل گیا")
            return True
        return False
    
    def close_valve(self, zone):
        """
        والو بند کریں
        """
        if zone in self.zones:
            self.zones[zone] = False
            print(f"❌ {zone} والو بند ہو گیا")
            return True
        return False
    
    def get_status(self):
        """
        حالت دیکھیں
        """
        return self.zones.copy()
    
    def irrigate(self, zone, duration_minutes):
        """
        آبپاشی کریں
        """
        self.open_valve(zone)
        
        # پانی کی مقدار
        water_liters = self.flow_rate * duration_minutes
        
        print(f"💧 {zone} میں {duration_minutes} منٹ آبپاشی")
        print(f"💧 کل پانی: {water_liters} لیٹر")
        
        # حقیقی میں یہاں انتظار ہوگا
        # time.sleep(duration_minutes * 60)
        
        self.close_valve(zone)
        return water_liters

# استعمال
controller = ValveController()
water_used = controller.irrigate('zone1', 30)
print(f"\nاستعمال شدہ پانی: {water_used} لیٹر")
```

## مکمل سسٹم

```python
class SmartIrrigationSystem:
    def __init__(self, field_id):
        self.field_id = field_id
        self.sensors = IrrigationSensors(field_id)
        self.scheduler = IrrigationScheduler(field_id)
        self.controller = ValveController()
        self.model = IrrigationDecisionModel('wheat')
    
    def run_cycle(self):
        """
        ایک سائیکل چلائیں
        """
        # سینسر پڑھیں
        reading = self.sensors.read_all()
        
        # ETo حساب کریں
        weather = reading['weather']
        eto = calculate_eto(
            weather['temperature'],
            weather['humidity'],
            weather['wind_speed'],
            weather['solar_radiation']
        )
        
        # فیصلہ کریں
        recommendation = self.model.get_recommendation(
            reading['soil_moisture_shallow'],
            eto
        )
        
        # رپورٹ
        report = {
            'timestamp': reading['timestamp'],
            'soil_moisture': reading['soil_moisture_shallow'],
            'eto': eto,
            'recommendation': recommendation
        }
        
        # آبپاشی کریں اگر ضرورت ہو
        if recommendation['irrigate']:
            duration = int(recommendation['water_mm'] * 10)
            self.controller.irrigate('zone1', duration)
            report['irrigated'] = True
            report['water_used'] = duration * 10
        else:
            report['irrigated'] = False
        
        return report

# استعمال
system = SmartIrrigationSystem('F001')
report = system.run_cycle()

print(f"\n📊 رپورٹ:")
print(f"  نمی: {report['soil_moisture']:.1f}%")
print(f"  ETo: {report['eto']:.2f} mm/day")
print(f"  {report['recommendation']['message']}")
```

## خلاصہ

- سمارٹ آبپاشی پانی بچاتی ہے
- سینسرز اور ML مل کر فیصلے کرتے ہیں
- خودکار شیڈولنگ آسان ہے
- والو کنٹرول ضروری ہے

## اگلے اقدامات

- [کیپسٹون پروجیکٹ](/docs/module-4/capstone-project)
