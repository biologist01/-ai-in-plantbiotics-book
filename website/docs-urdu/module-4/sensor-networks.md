---
sidebar_position: 2
---

# سینسر نیٹورکس اور ڈیٹا مینجمنٹ

## تعارف

زرعی IoT میں متعدد سینسرز کا نیٹورک ضروری ہے۔ یہاں ہم سیکھیں گے کہ سینسر نیٹورک کیسے بنایا اور منظم کیا جائے 📡🌾۔

## نیٹورک ٹاپولوجی

```
         ┌───────────────────┐
         │    کلاؤڈ سرور    │
         └─────────┬─────────┘
                   │
         ┌─────────┴─────────┐
         │     گیٹ وے      │
         └─────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───┴───┐     ┌───┴───┐     ┌───┴───┐
│سینسر 1│     │سینسر 2│     │سینسر 3│
└───────┘     └───────┘     └───────┘
```

## سینسر نوڈ کلاس

```python
from datetime import datetime
import json
import uuid

class SensorNode:
    def __init__(self, node_id=None, location=None, sensors=None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.location = location or {'lat': 0, 'lon': 0, 'field': 'unknown'}
        self.sensors = sensors or ['temperature', 'humidity', 'soil_moisture']
        self.battery = 100.0
        self.last_reading = None
    
    def read_sensors(self):
        """
        تمام سینسرز سے پڑھیں
        """
        import random
        
        readings = {
            'node_id': self.node_id,
            'timestamp': datetime.now().isoformat(),
            'location': self.location,
            'battery': self.battery,
            'data': {}
        }
        
        for sensor in self.sensors:
            if sensor == 'temperature':
                readings['data'][sensor] = 25 + random.uniform(-5, 10)
            elif sensor == 'humidity':
                readings['data'][sensor] = 60 + random.uniform(-20, 20)
            elif sensor == 'soil_moisture':
                readings['data'][sensor] = 50 + random.uniform(-20, 20)
            elif sensor == 'light':
                readings['data'][sensor] = random.randint(0, 100000)
        
        # بیٹری کم ہو رہی ہے
        self.battery -= 0.1
        self.last_reading = readings
        
        return readings

# نوڈز بنائیں
nodes = [
    SensorNode('N001', {'lat': 31.5, 'lon': 74.3, 'field': 'A1'}),
    SensorNode('N002', {'lat': 31.5, 'lon': 74.31, 'field': 'A2'}),
    SensorNode('N003', {'lat': 31.51, 'lon': 74.3, 'field': 'B1'}),
]

# ریڈنگز لیں
for node in nodes:
    reading = node.read_sensors()
    print(f"نوڈ {node.node_id}: {reading['data']}")
```

## گیٹ وے

```python
import time
from collections import deque

class Gateway:
    def __init__(self, gateway_id):
        self.gateway_id = gateway_id
        self.nodes = {}
        self.data_buffer = deque(maxlen=1000)
        self.is_connected = False
    
    def register_node(self, node):
        """
        نیا نوڈ رجسٹر کریں
        """
        self.nodes[node.node_id] = node
        print(f"✅ نوڈ {node.node_id} رجسٹر ہو گیا")
    
    def collect_data(self):
        """
        تمام نوڈز سے ڈیٹا جمع کریں
        """
        collected = []
        
        for node_id, node in self.nodes.items():
            try:
                reading = node.read_sensors()
                self.data_buffer.append(reading)
                collected.append(reading)
            except Exception as e:
                print(f"❌ نوڈ {node_id} سے ڈیٹا نہیں آیا: {e}")
        
        return collected
    
    def get_buffer_summary(self):
        """
        بفر کا خلاصہ
        """
        return {
            'total_readings': len(self.data_buffer),
            'nodes_count': len(self.nodes),
            'oldest': self.data_buffer[0]['timestamp'] if self.data_buffer else None,
            'newest': self.data_buffer[-1]['timestamp'] if self.data_buffer else None
        }

# گیٹ وے بنائیں
gateway = Gateway('GW001')

# نوڈز رجسٹر کریں
for node in nodes:
    gateway.register_node(node)

# ڈیٹا جمع کریں
data = gateway.collect_data()
print(f"\nکل ریڈنگز: {len(data)}")
print(f"بفر سمری: {gateway.get_buffer_summary()}")
```

## ڈیٹا بیس سٹوریج

```python
import sqlite3
from datetime import datetime

class SensorDatabase:
    def __init__(self, db_path='sensors.db'):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                temperature REAL,
                humidity REAL,
                soil_moisture REAL,
                battery REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                location_lat REAL,
                location_lon REAL,
                field TEXT,
                registered_at TEXT
            )
        ''')
        
        self.conn.commit()
    
    def insert_reading(self, reading):
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO readings 
            (node_id, timestamp, temperature, humidity, soil_moisture, battery)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            reading['node_id'],
            reading['timestamp'],
            reading['data'].get('temperature'),
            reading['data'].get('humidity'),
            reading['data'].get('soil_moisture'),
            reading['battery']
        ))
        
        self.conn.commit()
    
    def get_node_history(self, node_id, limit=100):
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT * FROM readings 
            WHERE node_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (node_id, limit))
        
        return cursor.fetchall()
    
    def get_field_average(self, field):
        """
        فیلڈ کی اوسط ریڈنگز
        """
        # سادہ مثال
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT AVG(temperature), AVG(humidity), AVG(soil_moisture)
            FROM readings
        ''')
        return cursor.fetchone()

# استعمال
db = SensorDatabase(':memory:')  # میموری میں

# ڈیٹا سٹور کریں
for reading in data:
    db.insert_reading(reading)

print("✅ ڈیٹا محفوظ ہو گیا")

# اوسط دیکھیں
avg = db.get_field_average('A1')
print(f"اوسط درجہ حرارت: {avg[0]:.1f}°C")
```

## ڈیٹا ایگریگیشن

```python
import numpy as np
from collections import defaultdict

class DataAggregator:
    def __init__(self):
        self.data = defaultdict(list)
    
    def add_reading(self, reading):
        node_id = reading['node_id']
        self.data[node_id].append(reading)
    
    def get_statistics(self, node_id):
        """
        نوڈ کے اعدادوشمار
        """
        readings = self.data[node_id]
        
        if not readings:
            return None
        
        temps = [r['data'].get('temperature', 0) for r in readings]
        humidity = [r['data'].get('humidity', 0) for r in readings]
        
        return {
            'node_id': node_id,
            'count': len(readings),
            'temperature': {
                'min': np.min(temps),
                'max': np.max(temps),
                'mean': np.mean(temps),
                'std': np.std(temps)
            },
            'humidity': {
                'min': np.min(humidity),
                'max': np.max(humidity),
                'mean': np.mean(humidity),
                'std': np.std(humidity)
            }
        }
    
    def get_network_summary(self):
        """
        پورے نیٹورک کا خلاصہ
        """
        all_temps = []
        all_humidity = []
        
        for node_id, readings in self.data.items():
            all_temps.extend([r['data'].get('temperature', 0) for r in readings])
            all_humidity.extend([r['data'].get('humidity', 0) for r in readings])
        
        return {
            'total_nodes': len(self.data),
            'total_readings': sum(len(r) for r in self.data.values()),
            'avg_temperature': np.mean(all_temps),
            'avg_humidity': np.mean(all_humidity)
        }

# استعمال
aggregator = DataAggregator()

# ڈیٹا شامل کریں
for reading in data:
    aggregator.add_reading(reading)

# اعدادوشمار دیکھیں
for node in nodes:
    stats = aggregator.get_statistics(node.node_id)
    print(f"\nنوڈ {node.node_id}:")
    print(f"  اوسط درجہ حرارت: {stats['temperature']['mean']:.1f}°C")
    print(f"  اوسط نمی: {stats['humidity']['mean']:.1f}%")

print(f"\nنیٹورک سمری: {aggregator.get_network_summary()}")
```

## ڈیٹا کوالٹی چیکنگ

```python
class DataQualityChecker:
    def __init__(self):
        self.valid_ranges = {
            'temperature': (-10, 60),
            'humidity': (0, 100),
            'soil_moisture': (0, 100)
        }
    
    def check_reading(self, reading):
        """
        ریڈنگ کی کوالٹی چیک کریں
        """
        issues = []
        
        for sensor, value in reading['data'].items():
            if sensor in self.valid_ranges:
                min_val, max_val = self.valid_ranges[sensor]
                
                if value < min_val or value > max_val:
                    issues.append({
                        'sensor': sensor,
                        'value': value,
                        'issue': 'رینج سے باہر'
                    })
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def clean_data(self, readings):
        """
        خراب ڈیٹا ہٹائیں
        """
        clean = []
        rejected = []
        
        for reading in readings:
            check = self.check_reading(reading)
            
            if check['valid']:
                clean.append(reading)
            else:
                rejected.append((reading, check['issues']))
        
        return clean, rejected

# استعمال
checker = DataQualityChecker()

# ٹیسٹ ریڈنگ
test_data = [
    {'node_id': 'N001', 'data': {'temperature': 25, 'humidity': 60}},
    {'node_id': 'N002', 'data': {'temperature': 150, 'humidity': 200}},  # خراب
]

clean, rejected = checker.clean_data(test_data)
print(f"✅ درست: {len(clean)}")
print(f"❌ مسترد: {len(rejected)}")
```

## خلاصہ

- سینسر نوڈز ڈیٹا جمع کرتے ہیں
- گیٹ وے ڈیٹا کو کلاؤڈ بھیجتا ہے
- ڈیٹا بیس سٹوریج ضروری ہے
- کوالٹی چیکنگ اہم ہے

## اگلے اقدامات

- [پیداوار پیش گوئی](/docs/module-4/yield-prediction)
