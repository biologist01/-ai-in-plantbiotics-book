---
sidebar_position: 2
---

# جینومک سیکوینسز کے لیے ڈیپ لرننگ

## تعارف

ڈیپ لرننگ جینومک سیکوینس تجزیے میں استعمال ہو رہی ہے، بشمول پروموٹر کی پیش گوئی، سپلائس سائٹ ڈیٹیکشن، اور جین فائنڈنگ 🧬۔

## DNA سیکوینس انکوڈنگ

### ون-ہاٹ انکوڈنگ

```python
import numpy as np

def one_hot_encode(sequence):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((len(sequence), 4))
    
    for i, nucleotide in enumerate(sequence):
        if nucleotide in mapping:
            encoded[i, mapping[nucleotide]] = 1
    
    return encoded

# مثال
seq = "ATGCGATC"
encoded = one_hot_encode(seq)
print(encoded.shape)  # (8, 4)
```

### K-mer انکوڈنگ

```python
from collections import Counter

def kmer_encoding(sequence, k=3):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)

# استعمال
seq = "ATGCGATCGATCG"
kmers = kmer_encoding(seq, k=3)
print(kmers)  # {'ATG': 1, 'TGC': 1, 'GCG': 1, ...}
```

## CNN سے موٹف ڈسکوری

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_dna_cnn(seq_length=1000):
    model = models.Sequential([
        # پہلی کنولیوشن - موٹف تلاش کریں
        layers.Conv1D(64, 15, activation='relu', 
                      input_shape=(seq_length, 4)),
        layers.MaxPooling1D(2),
        
        # دوسری کنولیوشن
        layers.Conv1D(128, 10, activation='relu'),
        layers.MaxPooling1D(2),
        
        # تیسری کنولیوشن
        layers.Conv1D(256, 5, activation='relu'),
        layers.GlobalMaxPooling1D(),
        
        # ڈینس پرتیں
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # پروموٹر یا نہیں
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# ماڈل بنائیں
model = build_dna_cnn()
model.summary()
```

## LSTM سے سیکوینس ماڈلنگ

```python
def build_dna_lstm(seq_length=500):
    model = models.Sequential([
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=True),
            input_shape=(seq_length, 4)
        ),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model
```

## پروموٹر کی پیش گوئی

```python
# ڈیٹا تیار کریں
def prepare_promoter_data(fasta_file, labels_file):
    sequences = []
    labels = []
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        encoded = one_hot_encode(seq)
        sequences.append(encoded)
    
    labels = pd.read_csv(labels_file)['is_promoter'].values
    
    return np.array(sequences), labels

# ماڈل ٹرین کریں
X, y = prepare_promoter_data('sequences.fasta', 'labels.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = build_dna_cnn(seq_length=X.shape[1])
model.fit(X_train, y_train, 
          epochs=20, 
          batch_size=32,
          validation_data=(X_test, y_test))
```

## جین ایکسپریشن پیش گوئی

```python
def build_expression_predictor(seq_length=2000):
    # پروموٹر سیکوینس سے ایکسپریشن لیول پیش گوئی
    model = models.Sequential([
        layers.Conv1D(64, 20, activation='relu', input_shape=(seq_length, 4)),
        layers.MaxPooling1D(4),
        layers.Conv1D(128, 10, activation='relu'),
        layers.MaxPooling1D(4),
        layers.Conv1D(256, 5, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='linear')  # ایکسپریشن لیول
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

## ٹرانسفارمر ماڈلز

```python
import tensorflow as tf

class DNATransformer(tf.keras.Model):
    def __init__(self, seq_length, num_heads=4, d_model=64):
        super().__init__()
        
        self.embedding = layers.Dense(d_model)
        self.pos_encoding = self.positional_encoding(seq_length, d_model)
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model
        )
        
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_model * 4, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.classifier = layers.Dense(1, activation='sigmoid')
    
    def call(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoding
        
        attn_output = self.attention(x, x)
        x = x + attn_output
        
        ffn_output = self.ffn(x)
        x = x + ffn_output
        
        x = tf.reduce_mean(x, axis=1)
        return self.classifier(x)
```

## خلاصہ

| ماڈل | استعمال | فوائد |
|------|--------|-------|
| CNN | موٹف ڈسکوری | مقامی پیٹرن |
| LSTM | سیکوینس ماڈلنگ | لمبی دوری |
| ٹرانسفارمر | جینوم وائڈ | توجہ میکانزم |

## اگلے اقدامات

- [CRISPR AI](/docs/module-3/crispr-ai) - جین ایڈیٹنگ
