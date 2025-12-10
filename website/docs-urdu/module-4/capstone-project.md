---
sidebar_position: 5
---

# کیپسٹون پروجیکٹ: سمارٹ فارم ڈیش بورڈ

## منصوبے کا جائزہ

اس فائنل پروجیکٹ میں ہم ایک مکمل سمارٹ فارمنگ ڈیش بورڈ بنائیں گے جو تمام ماڈیولز کو یکجا کرے 🌾🤖📊۔

## مقاصد

- سینسر ڈیٹا ویژولائزیشن
- ML سے پیش گوئی
- خودکار آبپاشی کنٹرول
- الرٹس اور نوٹیفیکیشنز

## پروجیکٹ ڈھانچہ

```
smart_farm/
├── app.py              # مین ایپلیکیشن
├── sensors.py          # سینسر ماڈیول
├── ml_models.py        # ML ماڈلز
├── irrigation.py       # آبپاشی کنٹرول
├── database.py         # ڈیٹا سٹوریج
├── alerts.py           # الرٹس سسٹم
└── templates/
    └── dashboard.html  # UI
```

## سینسر ماڈیول

```python
# sensors.py

import numpy as np
from datetime import datetime
from dataclasses import dataclass

@dataclass
class SensorReading:
    timestamp: str
    temperature: float
    humidity: float
    soil_moisture: float
    light_intensity: float
    ndvi: float

class FarmSensors:
    def __init__(self, farm_id):
        self.farm_id = farm_id
        self.calibration = {
            'temperature_offset': 0,
            'humidity_offset': 0
        }
    
    def read(self) -> SensorReading:
        """
        تمام سینسرز سے پڑھیں
        """
        now = datetime.now()
        hour = now.hour
        
        # دن/رات کے مطابق
        day_factor = np.sin(np.pi * hour / 12) if 6 <= hour <= 18 else 0
        
        return SensorReading(
            timestamp=now.isoformat(),
            temperature=25 + 10 * day_factor + np.random.uniform(-2, 2),
            humidity=60 - 20 * day_factor + np.random.uniform(-5, 5),
            soil_moisture=50 + np.random.uniform(-15, 15),
            light_intensity=day_factor * 80000 + np.random.uniform(0, 5000),
            ndvi=0.5 + 0.3 * day_factor + np.random.uniform(-0.1, 0.1)
        )
    
    def read_history(self, hours=24):
        """
        تاریخی ڈیٹا (ڈیمو)
        """
        readings = []
        for i in range(hours * 12):  # ہر 5 منٹ
            readings.append({
                'time': i * 5,
                'temperature': 25 + 8 * np.sin(np.pi * i / 144) + np.random.uniform(-1, 1),
                'humidity': 60 - 15 * np.sin(np.pi * i / 144) + np.random.uniform(-3, 3),
                'soil_moisture': 50 + np.random.uniform(-10, 10)
            })
        return readings
```

## ML ماڈلز

```python
# ml_models.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

class CropHealthPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self._train_demo()
    
    def _train_demo(self):
        """
        ڈیمو ٹریننگ
        """
        np.random.seed(42)
        n = 500
        
        X = np.random.rand(n, 4)  # ndvi, soil, temp, humidity
        y = (X[:, 0] > 0.4).astype(int)  # صحت مند اگر NDVI > 0.4
        
        self.model.fit(X, y)
    
    def predict(self, ndvi, soil_moisture, temperature, humidity):
        """
        فصل کی صحت پیش گوئی
        """
        X = np.array([[ndvi, soil_moisture/100, temperature/50, humidity/100]])
        prob = self.model.predict_proba(X)[0][1]
        
        if prob > 0.8:
            status = 'بہترین 🌿'
        elif prob > 0.5:
            status = 'اچھی 🌱'
        else:
            status = 'توجہ چاہیے ⚠️'
        
        return {
            'health_score': round(prob * 100, 1),
            'status': status
        }

class YieldPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self._train_demo()
    
    def _train_demo(self):
        np.random.seed(42)
        n = 500
        
        X = np.random.rand(n, 5)
        y = 30 + 20 * X[:, 0] + 15 * X[:, 1] + np.random.normal(0, 5, n)
        
        self.model.fit(X, y)
    
    def predict(self, features):
        """
        پیداوار پیش گوئی
        """
        X = np.array([features])
        prediction = self.model.predict(X)[0]
        
        return {
            'yield_estimate': round(prediction, 1),
            'unit': 'من/ایکڑ',
            'confidence_interval': (round(prediction * 0.9, 1), round(prediction * 1.1, 1))
        }

class DiseaseDetector:
    def __init__(self):
        self.diseases = ['صحت مند', 'زنگ', 'جھلسا', 'پاؤڈری ملڈیو']
        self.model = RandomForestClassifier(n_estimators=100)
        self._train_demo()
    
    def _train_demo(self):
        np.random.seed(42)
        n = 500
        
        X = np.random.rand(n, 3)
        y = np.random.randint(0, 4, n)
        
        self.model.fit(X, y)
    
    def detect(self, ndvi, humidity, temperature):
        """
        بیماری کی شناخت
        """
        X = np.array([[ndvi, humidity/100, temperature/50]])
        pred = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        
        return {
            'disease': self.diseases[pred],
            'confidence': round(max(probs) * 100, 1),
            'all_probabilities': {
                self.diseases[i]: round(p * 100, 1) 
                for i, p in enumerate(probs)
            }
        }
```

## آبپاشی ماڈیول

```python
# irrigation.py

from datetime import datetime, timedelta

class SmartIrrigation:
    def __init__(self):
        self.zones = {
            'zone1': {'status': 'off', 'last_irrigation': None},
            'zone2': {'status': 'off', 'last_irrigation': None},
            'zone3': {'status': 'off', 'last_irrigation': None},
        }
        self.history = []
    
    def calculate_water_need(self, soil_moisture, temperature, humidity):
        """
        پانی کی ضرورت حساب کریں
        """
        # بنیادی ضرورت
        base_need = max(0, 60 - soil_moisture) * 0.5
        
        # موسم کی ایڈجسٹمنٹ
        temp_factor = 1 + (temperature - 25) * 0.02
        humidity_factor = 1 - (humidity - 50) * 0.01
        
        water_mm = base_need * temp_factor * humidity_factor
        
        return max(0, round(water_mm, 1))
    
    def get_recommendation(self, soil_moisture, temperature, humidity):
        """
        آبپاشی کی سفارش
        """
        water_need = self.calculate_water_need(soil_moisture, temperature, humidity)
        
        if soil_moisture < 30:
            urgency = 'فوری ⚠️'
            action = 'ابھی آبپاشی کریں'
        elif soil_moisture < 45:
            urgency = 'جلد 💧'
            action = '2-4 گھنٹے میں آبپاشی'
        elif water_need > 0:
            urgency = 'عام 🌱'
            action = 'آج آبپاشی تجویز'
        else:
            urgency = 'کوئی ضرورت نہیں ✅'
            action = 'پانی کافی ہے'
        
        return {
            'water_need_mm': water_need,
            'urgency': urgency,
            'action': action,
            'duration_minutes': int(water_need * 10)
        }
    
    def irrigate(self, zone, duration_minutes):
        """
        آبپاشی کریں
        """
        if zone not in self.zones:
            return {'success': False, 'error': 'زون نہیں ملا'}
        
        self.zones[zone]['status'] = 'irrigating'
        
        # ریکارڈ
        event = {
            'zone': zone,
            'start_time': datetime.now().isoformat(),
            'duration': duration_minutes,
            'water_liters': duration_minutes * 10  # 10 L/min
        }
        self.history.append(event)
        
        self.zones[zone]['status'] = 'off'
        self.zones[zone]['last_irrigation'] = datetime.now()
        
        return {'success': True, 'event': event}
    
    def get_schedule(self, days=7):
        """
        آبپاشی شیڈول
        """
        schedule = []
        base_date = datetime.now()
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            schedule.append({
                'date': date.strftime('%Y-%m-%d'),
                'day': ['پیر', 'منگل', 'بدھ', 'جمعرات', 'جمعہ', 'ہفتہ', 'اتوار'][date.weekday()],
                'recommended_time': '05:30 صبح',
                'zone': f'zone{(i % 3) + 1}'
            })
        
        return schedule
```

## الرٹس سسٹم

```python
# alerts.py

from datetime import datetime
from enum import Enum

class AlertLevel(Enum):
    INFO = 'معلومات'
    WARNING = 'خبردار'
    CRITICAL = 'اہم'

class AlertSystem:
    def __init__(self):
        self.alerts = []
        self.thresholds = {
            'temperature_high': 40,
            'temperature_low': 5,
            'soil_moisture_low': 25,
            'soil_moisture_high': 85,
            'humidity_high': 95
        }
    
    def check(self, reading):
        """
        الرٹس چیک کریں
        """
        new_alerts = []
        
        # درجہ حرارت
        if reading.temperature > self.thresholds['temperature_high']:
            new_alerts.append(self._create_alert(
                AlertLevel.CRITICAL,
                '🌡️ درجہ حرارت بہت زیادہ',
                f"موجودہ: {reading.temperature:.1f}°C"
            ))
        elif reading.temperature < self.thresholds['temperature_low']:
            new_alerts.append(self._create_alert(
                AlertLevel.CRITICAL,
                '❄️ ٹھنڈ کا خطرہ',
                f"موجودہ: {reading.temperature:.1f}°C"
            ))
        
        # مٹی کی نمی
        if reading.soil_moisture < self.thresholds['soil_moisture_low']:
            new_alerts.append(self._create_alert(
                AlertLevel.WARNING,
                '💧 مٹی خشک ہے',
                f"نمی: {reading.soil_moisture:.1f}%"
            ))
        
        # نمی
        if reading.humidity > self.thresholds['humidity_high']:
            new_alerts.append(self._create_alert(
                AlertLevel.WARNING,
                '💨 زیادہ نمی - بیماری کا خطرہ',
                f"نمی: {reading.humidity:.1f}%"
            ))
        
        self.alerts.extend(new_alerts)
        return new_alerts
    
    def _create_alert(self, level, title, description):
        return {
            'timestamp': datetime.now().isoformat(),
            'level': level.value,
            'title': title,
            'description': description,
            'acknowledged': False
        }
    
    def get_active_alerts(self):
        """
        فعال الرٹس
        """
        return [a for a in self.alerts if not a['acknowledged']]
    
    def acknowledge(self, alert_index):
        """
        الرٹ تسلیم کریں
        """
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index]['acknowledged'] = True
```

## مین ایپلیکیشن

```python
# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ماڈیولز امپورٹ کریں (اوپر کی کلاسز)
# from sensors import FarmSensors
# from ml_models import CropHealthPredictor, YieldPredictor, DiseaseDetector
# from irrigation import SmartIrrigation
# from alerts import AlertSystem

def create_dashboard():
    """
    ڈیش بورڈ بنائیں
    """
    st.set_page_config(page_title="سمارٹ فارم ڈیش بورڈ", layout="wide")
    st.title("🌾 سمارٹ فارم مانیٹرنگ ڈیش بورڈ")
    
    # سائیڈبار
    st.sidebar.header("⚙️ ترتیبات")
    farm_id = st.sidebar.selectbox("فارم منتخب کریں", ['فارم A', 'فارم B', 'فارم C'])
    
    # اجزاء بنائیں
    sensors = FarmSensors(farm_id)
    health_predictor = CropHealthPredictor()
    yield_predictor = YieldPredictor()
    disease_detector = DiseaseDetector()
    irrigation = SmartIrrigation()
    alert_system = AlertSystem()
    
    # موجودہ ریڈنگز
    reading = sensors.read()
    
    # الرٹس چیک کریں
    new_alerts = alert_system.check(reading)
    
    # میٹرکس
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌡️ درجہ حرارت", f"{reading.temperature:.1f}°C")
    with col2:
        st.metric("💧 مٹی کی نمی", f"{reading.soil_moisture:.1f}%")
    with col3:
        st.metric("💨 ہوا کی نمی", f"{reading.humidity:.1f}%")
    with col4:
        st.metric("🌿 NDVI", f"{reading.ndvi:.2f}")
    
    # فصل کی صحت
    st.subheader("🌱 فصل کی صحت")
    health = health_predictor.predict(
        reading.ndvi, reading.soil_moisture,
        reading.temperature, reading.humidity
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.progress(health['health_score'] / 100)
        st.write(f"حالت: {health['status']}")
    
    with col2:
        disease = disease_detector.detect(
            reading.ndvi, reading.humidity, reading.temperature
        )
        st.write(f"بیماری کی جانچ: {disease['disease']}")
        st.write(f"اعتماد: {disease['confidence']}%")
    
    # آبپاشی
    st.subheader("💧 آبپاشی کی سفارش")
    rec = irrigation.get_recommendation(
        reading.soil_moisture, reading.temperature, reading.humidity
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("پانی کی ضرورت", f"{rec['water_need_mm']} mm")
    with col2:
        st.write(f"**فوری پن:** {rec['urgency']}")
    with col3:
        st.write(f"**عمل:** {rec['action']}")
    
    # چارٹس
    st.subheader("📊 24 گھنٹے کا ڈیٹا")
    history = sensors.read_history(24)
    df = pd.DataFrame(history)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['temperature'], name='درجہ حرارت'))
    fig.add_trace(go.Scatter(y=df['humidity'], name='نمی'))
    fig.add_trace(go.Scatter(y=df['soil_moisture'], name='مٹی کی نمی'))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # الرٹس
    if new_alerts:
        st.subheader("⚠️ الرٹس")
        for alert in new_alerts:
            st.warning(f"{alert['title']}: {alert['description']}")
    
    # شیڈول
    st.subheader("📅 آبپاشی شیڈول")
    schedule = irrigation.get_schedule()
    st.table(pd.DataFrame(schedule))

if __name__ == "__main__":
    create_dashboard()
```

## چلانے کا طریقہ

```bash
# ضروریات انسٹال کریں
pip install streamlit pandas plotly scikit-learn numpy

# ایپ چلائیں
streamlit run app.py
```

## خلاصہ

اس پروجیکٹ میں ہم نے سیکھا:

- ✅ سینسر ڈیٹا انٹیگریشن
- ✅ ML ماڈلز برائے پیش گوئی
- ✅ خودکار آبپاشی فیصلے
- ✅ الرٹس سسٹم
- ✅ ڈیش بورڈ ڈیزائن

## مزید بہتری

- موبائل ایپ
- ڈرون انٹیگریشن
- مارکیٹ پیش گوئی
- بلاک چین ٹریس ایبیلٹی

---

🎉 **مبارک ہو!** آپ نے کورس مکمل کیا۔

پلانٹ بائیوٹیکنالوجی میں AI کا سفر جاری رکھیں! 🌾🤖
