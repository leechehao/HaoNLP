import argparse
import json

import mlflow
import datasets
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_uri", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--inputs", type=str)
    parser.add_argument("--input_file", type=str, help="CSV 檔案格式。")
    parser.add_argument("--output_file", default="output.json", type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    model = mlflow.pytorch.load_model(f"runs:/{args.run_id}/model")
    model.eval()

    if args.inputs is not None:
        print(args.inputs)
        print(model.hf_predict(args.inputs))
    elif (args.input_file is not None) and (args.key is not None):
        outputs = []
        dataset = datasets.Dataset.from_csv(args.input_file)
        key_dataset = KeyDataset(dataset, args.key)
        for data, out in zip(key_dataset, tqdm(model.hf_predict(key_dataset, batch_size=args.batch_size), total=len(dataset))):
            outputs.append({"text": data, "results": out})
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4, default=convert_float)
    else:
        raise RuntimeError("請至少給定 inputs 或 input_file 及 key。")


def convert_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    return obj


if __name__ == "__main__":
    # python winlp/cli/predict.py --inputs='New nodule at right lower lung.' --tracking_uri=/home/bryant/MyMLOps/exp --run_id
    # python winlp/cli/predict.py --input_file=gg.csv --key=text --batch_size=16 --tracking_uri=/home/bryant/MyMLOps/exp --run_id
    main()
