from setuptools import setup, find_packages


setup(
    name="winlp",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "winlp-train=winlp.cli.train:main",
            "winlp-test=winlp.cli.test:main",
            "winlp-predict=winlp.cli.predict:main",
        ],
    },
    python_requires=">=3.10",
    install_requires=[
        "hydra-core==1.3.2",
        "mlflow==2.8.1",
        "torch==2.1.1",
        "lightning==2.1.2",
        "datasets==2.15.0",
        "transformers==4.35.2",
        "onnx==1.15.0",
        "onnxruntime-gpu==1.16.3",
        "boto3==1.33.6",
    ],
)
