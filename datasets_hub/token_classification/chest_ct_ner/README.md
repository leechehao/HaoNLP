---
configs:
- config_name: chest_ct_ner
  data_files:
  - split: train
    path: "data/train.conll"
  - split: valid
    path: "data/valid.conll"
  - split: test
    path: "data/test.conll"
---
# Chest CT 影像文字報告實體識別資料集
## 儲存庫結構
```
chest_ct_ner/
├── README.md
├── data/
│   ├── train.conll
│   ├── valid.conll
│   └── test.conll
└── chest_ct_ner.py
```

## 資料格式
### CoNLL 格式的文字檔
CoNLL 格式是一種文字檔，每行有一個 `token`，句子之間由空行分隔。一行中的第一個字應該是 `token`，而最後一個字應該是 `label`。
以下列兩句為例：
+ A small right renal cyst .
+ No lung consolidation
```
A O
small B-Size
right B-Location
renal I-Location
cyst B-Finding
. O

No B-Certainty
lung B-Location
consolidation B-Finding
```

## 使用方式
```python
from datasets import load_dataset

dataset  = load_dataset("chest_ct_ner")
# DatasetDict({
#     train: Dataset({
#         features: ['id', 'tokens', 'ner_tags'],
#         num_rows: 142
#     })
#     validation: Dataset({
#         features: ['id', 'tokens', 'ner_tags'],
#         num_rows: 21
#     })
#     test: Dataset({
#         features: ['id', 'tokens', 'ner_tags'],
#         num_rows: 41
#     })
# })
```
