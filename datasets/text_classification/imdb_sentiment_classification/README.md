---
configs:
- config_name: imdb_sentiment_classification
  data_files:
  - split: train
    path: "data/train.csv"
  - split: validation
    path: "data/validation.csv"
  - split: test
    path: "data/test.csv"
---

# Chest CT 影像文字報告實體識別資料集
## 儲存庫結構
```
imdb_sentiment_classification/
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── imdb_sentiment_classification.py
└── README.md
```

## 資料格式
### csv 檔
csv 檔中每一列包含一段文本與其對應之標籤(neg: 0 / pos: 1)。
參考例子：
```
texts,labels

"Story of a man who has unnatural feelings for a pig. Starts out with a opening scene that is a terrific example of absurd comedy. A formal orchestra audience is turned into an insane, violent mob by the crazy chantings of it's singers. Unfortunately it stays absurd the WHOLE time with no general narrative eventually making it just too off putting. Even those from the era should be turned off. The cryptic dialogue would make Shakespeare seem easy to a third grader. On a technical level it's better than you might think with some good cinematography by future great Vilmos Zsigmond. Future stars Sally Kirkland and Frederic Forrest can be seen briefly.",neg

"If you haven't already seen this movie of Mary-Kate and Ashley's, then all I can say is: ""What Are You Waiting For!?"". This is yet another terrific and wonderful movie by the fraternal twins that we all know and love so much! It's fun, romantic, exciting and absolutely breath-taking (scenery-wise)! Of course; as always, Mary-Kate and Ashley are the main scenery here anyway! Would any true fan want it any other way? Of course not! Anyway; it's a great movie in every sense of the word, so if you haven't already seen it then you just have to now! I mean right now too! So what are you waiting for? I promise that you won't be disappointed! Sincerely, Rick Morris",pos
```

## 使用方式
```python
from datasets import load_dataset

dataset  = load_dataset("imdb_sentiment_classification")
# DatasetDict({
#     train: Dataset({
#         features: ['texts', 'labels'],
#         num_rows: 25000
#     })
#     validation: Dataset({
#         features: ['texts', 'labels'],
#         num_rows: 15000
#     })
#     test: Dataset({
#         features: ['texts', 'labels'],
#         num_rows: 10000
#     })
# })
```
