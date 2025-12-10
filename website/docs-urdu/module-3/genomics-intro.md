---
sidebar_position: 1
---

# پودوں کے جینومکس میں AI کا تعارف

## تعارف

مصنوعی ذہانت پودوں کی جینومکس ریسرچ کو تیز کر رہی ہے، DNA سیکوینسنگ تجزیے سے لے کر خصوصیات کی پیش گوئی تک۔ اس ماڈیول میں آپ جینومک ڈیٹا فارمیٹس اور بنیادی بائیو انفارمیٹکس سیکھیں گے 🧬۔

## جینومکس کی بنیادیں

### DNA، جینز، اور خصوصیات

```
DNA → RNA → پروٹین → خصوصیت
```

- **DNA**: جینیاتی معلومات کا ذخیرہ
- **جینز**: DNA کے فعال حصے
- **خصوصیات (Traits)**: مشاہدہ کی جانے والی خصوصیات

## جینومک ڈیٹا فارمیٹس

### FASTA فارمیٹ

```
>gene_id gene_name
ATGCGATCGATCGATCGATCG
ATCGATCGATCGATCGATCGA
```

### FASTQ فارمیٹ (کوالٹی سکورز کے ساتھ)

```
@read_id
ATGCGATCGATCGATCG
+
IIIIIIIIIIIIIIIII
```

### VCF فارمیٹ (ویریئنٹس)

```
#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO
chr1    12345   rs123   A       G       30      PASS    DP=100
```

## Python میں جینومک ڈیٹا

```python
from Bio import SeqIO
from Bio.Seq import Seq

# FASTA فائل پڑھیں
for record in SeqIO.parse("genes.fasta", "fasta"):
    print(f"جین: {record.id}")
    print(f"لمبائی: {len(record.seq)}")
    print(f"GC مواد: {gc_content(record.seq):.2f}%")

# GC مواد حساب کریں
def gc_content(sequence):
    g = sequence.count('G')
    c = sequence.count('C')
    return (g + c) / len(sequence) * 100
```

## AI کے استعمال

| استعمال | تفصیل |
|--------|-------|
| ویریئنٹ کالنگ | SNPs اور انڈیلز کا پتہ |
| جین کی پیش گوئی | کوڈنگ ریجنز تلاش کرنا |
| فنکشن پیش گوئی | پروٹین کا کام |
| GWAS | خصوصیت-جین تعلق |

## جینوٹائپ سے فینوٹائپ

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# جینومک ڈیٹا لوڈ کریں
genotypes = pd.read_csv('genotypes.csv')  # SNP ڈیٹا
phenotypes = pd.read_csv('phenotypes.csv')  # خصوصیات

# ماڈل ٹرین کریں
X = genotypes.values
y = phenotypes['yield'].values

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# فیچر اہمیت
importance = model.feature_importances_
top_snps = np.argsort(importance)[-10:]
print("سب سے اہم SNPs:", top_snps)
```

## GWAS (Genome-Wide Association Study)

```python
import numpy as np
from scipy import stats

def gwas_analysis(genotypes, phenotypes):
    """
    ہر SNP کے لیے خصوصیت کے ساتھ تعلق چیک کریں
    """
    n_snps = genotypes.shape[1]
    p_values = []
    
    for i in range(n_snps):
        snp = genotypes[:, i]
        
        # لکیری ریگریشن
        slope, intercept, r, p, se = stats.linregress(snp, phenotypes)
        p_values.append(p)
    
    return np.array(p_values)

# استعمال
p_values = gwas_analysis(genotypes.values, phenotypes['height'].values)

# اہم SNPs (p < 0.05 / n_snps for Bonferroni correction)
threshold = 0.05 / len(p_values)
significant = np.where(p_values < threshold)[0]
print(f"اہم SNPs: {len(significant)}")
```

## بائیو انفارمیٹکس ٹولز

### BLAST سے مماثلت تلاش کریں

```python
from Bio.Blast import NCBIWWW, NCBIXML

def blast_search(sequence):
    result = NCBIWWW.qblast("blastn", "nt", sequence)
    records = NCBIXML.parse(result)
    
    for record in records:
        for alignment in record.alignments[:5]:
            print(f"ہٹ: {alignment.title}")
            print(f"سکور: {alignment.hsps[0].score}")
```

## خلاصہ

- جینومکس پودوں کی بہتری کی بنیاد ہے
- AI جینومک ڈیٹا کا تجزیہ تیز کر رہا ہے
- GWAS خصوصیات سے جینز کا تعلق تلاش کرتا ہے

## اگلے اقدامات

- [سیکوینس تجزیہ](/docs/module-3/sequence-analysis) - ڈیپ لرننگ
