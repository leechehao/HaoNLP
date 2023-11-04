import argparse

import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, required=True)
    parser.add_argument("--tracking_uri", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    model = mlflow.pytorch.load_model(f"runs:/{args.run_id}/model")

    print(model.hf_predict(args.inputs))


if __name__ == "__main__":
    # python winlp/cli/predict.py --inputs='New nodule at right lower lung.' --tracking_uri=/home/bryant/MyMLOps/exp --run_id
    main()
