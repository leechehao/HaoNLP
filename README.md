# HaoNLP
HaoNLP 結合了 PyTorch Lightning 和 Hydra 的強大功能，同時也整合了 MLflow。HaoNLP 旨在提供一個簡化和優化的框架，讓機器學習項目的開發變得更加高效和靈活。通過利用 PyTorch Lightning 的輕量級封裝，它能夠簡化複雜模型的訓練過程，同時保持代碼的可讀性和可擴展性。此外，Hydra 的集成為項目配置提供了強大而靈活的管理，使得調整實驗參數變得簡單快捷。透過整合 MLflow，HaoNLP 為模型的版本控制、實驗追蹤以及成果分享提供了強有力的支援，從而使用戶能夠輕鬆管理和追蹤整個機器學習的生命週期。

HaoNLP 的目標是為研究人員和開發人員提供一個一站式的解決方案，通過減少必要的樣板代碼和提供更靈活的配置選項，來加速機器學習模型的開發和實驗過程。我相信，通過結合這兩個庫的優勢，並利用 MLflow 強大的模型管理和實驗追蹤功能，HaoNLP 能夠幫助用戶更有效地進行機器學習模型的開發、訓練和評估。

## :floppy_disk: Installation
在本地環境 `git clone` 整個 HaoNLP 專案，再執行 `pip install` 安裝 **`haonlp`** 套件：
```
git clone https://github.com/leechehao/HaoNLP.git
pip install ./HaoNLP
```
接著將 HaoNLP 資料夾中的 `src` 路徑添加至 **`PYTHONPATH`** 環境變數（開發階段）：
```bash
export PYTHONPATH="/path/to/HaoNLP/src:$PYTHONPATH"
```
:bulb: HaoNLP 預設的 Hydra Config Path 是指向 `src/haonlp_conf/` 位置。若需添加或修改 YAML 配置文件，請將它們放置在此目錄下。如果您希望使用非預設的配置路徑，可以通過設定環境變數 **`HAONLP_CONFIG_PATH`**（建議選項）來指定一個新的路徑，將配置和套件模組分開。（可以將 `src/haonlp_conf/` 目錄的內容複製一份到由 **`HAONLP_CONFIG_PATH`** 環境變數指定的路徑，以進行後續的修改。）

:notebook: 當所有 YAML 配置文件及模組都開發完畢，請重新安裝 **`haonlp`** 套件，此時就可以從 **`PYTHONPATH`** 環境變數移除 `/path/to/HaoNLP/src` 路徑：
```bash
pip install /path/to/HaoNLP
```

## :rocket: Quick Start
目前提供的任務類型：
+ Text Classification
+ Token Classification
+ Boundary Detection

### :books: 定義自己的資料集
根據任務類型符合對應的資料格式，以下示範 Token Classification。

首先創建自己的資料集儲存庫（Dataset Repository）並符合 datasets [官方文件](https://huggingface.co/docs/datasets/repository_structure)的規範：
```
custom_dataset/
├── README.md
├── data/
│   ├── train.conll
│   ├── valid.conll
│   └── test.conll
└── custom_dataset.py
```
Token Classification 的資料格式採用 CoNLL format：

將文本（`4. Mild pleural effusion on the left side .`）轉換成 CoNLL format。
```
4. O
Mild B-Attribute
pleural B-Finding
effusion I-Finding
on O
the O
left B-Location
side I-Location
. O

5. O
No B-Certainty
...
```
每一行都是一個 word，後面跟著其 label 種類，而不同文本之間用換行隔開。label 的格式採用 [**BIO format**](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))。

### :hammer_and_wrench: 新增 YAML 至 Hydra Config Path
在 `src/haonlp_conf/dataset/token_classification/` 資料夾下新增 `custom_dataset.yaml`（預設情況）：
```yaml
# @package dataset
defaults:
  - token_classification/default
  
dataset_name:
  - /path/to/your/custom/dataset/repository
```
接著在 `src/haonlp_conf/experiment/` 資料夾下新增 `custom_dataset_exp_1.yaml`（預設情況）：
```yaml
# @package _global_
defaults:
  - override /dataset: token_classification/custom_dataset
  - override /task: token_classification
  - _self_

task:
  monitor: val_f1
  mode: max
trainer:
  logger:
    experiment_name: Custom-Task
    run_name: run_1
    tracking_uri: http://localhost:5000
```
需要指定 **`dataset`** 和 **`task`** 這兩個 package 的 yaml，除此之外，還要設定 5 個 keys：
+ `task.monitor`：要監控的指標。用來選擇在驗證集上表現最好的 model checkpoint。
+ `task.mode`：最大化或最小化 `task.monitor`。可選的值為｛`min`, `max`｝。
+ `trainer.logger.experiment_name`：MLflow 實驗的名稱。
+ `trainer.logger.run_name`：MLflow run 的名稱。
+ `trainer.logger.tracking_uri`：MLflow 追蹤伺服器的 URI。預設的 `host` 為 `127.0.0.1`，`port` 為 `5000`。

### 訓練模型
```bash
# If you have already installed the `haonlp` package
haonlp-train +experiment=custom_dataset_exp_1

# If you git clone repository
python src/haonlp/cli/train.py +experiment=custom_dataset_exp_1
```

### 評估模型
需要將剛剛的 **`custom_dataset_exp_1.yaml`** 中添加 **`run_id`** 屬性，其值表示要評估的模型在 MLflow 中的 Run ID。
```yaml
...
    run_name: run_1
    tracking_uri: http://localhost:5000
    
run_id: 0855734d2553483997cd2023fa7eedad
```


```bash
# If you have already installed the `haonlp` package
haonlp-test +experiment=custom_dataset_exp_1

# If you git clone repository
python src/haonlp/cli/test.py +experiment=custom_dataset_exp_1
```

### 模型推理
執行 `haonlp-predict --inputs=<INPUTS> --tracking_uri=<TRACKING_URI> --run_id=<RUN_ID>` 指令：

```bash
# If you have already installed the `haonlp` package
haonlp-predict --inputs='New nodule at right lower lung.' --tracking_uri=http://localhost:5000 --run_id=0855734d2553483997cd2023fa7eedad

# If you git clone repository
python src/haonlp/cli/predict.py --inputs='New nodule at right lower lung.' --tracking_uri=http://localhost:5000 --run_id=0855734d2553483997cd2023fa7eedad
```