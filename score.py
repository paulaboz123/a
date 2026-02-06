name: albot-aml-v3
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pytorch
  - torchvision
  - pytorch-cuda=12.4
  - numpy=1.26.*
  - pandas=2.2.*
  - scikit-learn=1.4.*
  - scipy=1.11.*
  - matplotlib=3.8.*
  - seaborn=0.13.*
  - pip:
      - mlflow>=2.10
      - azure-ai-ml>=1.15
      - azure-identity>=1.16
      - azureml-inference-server-http>=1.2
      - transformers>=4.38
      - imbalanced-learn

name: albot-aml-v3
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - numpy=1.26.*
  - pandas=2.2.*
  - scikit-learn=1.4.*
  - scipy=1.11.*
  - matplotlib=3.8.*
  - seaborn=0.13.*
  - regex
  - joblib
  - pillow
  - psutil
  - protobuf<5
  - flask
  - fsspec
  - pip:
      # torch/torchvision instaluj w Dockerfile (patrz niżej), żeby kontrolować wariant CUDA
      - mlflow>=2.10
      - azure-ai-ml>=1.15
      - azure-identity>=1.16
      - azureml-inference-server-http>=1.2
      # legacy azureml-* tylko jeśli naprawdę musisz:
      # - azureml-core==...
      # - azureml-defaults==...
      # - azureml-mlflow==...
      - transformers>=4.38
      - imbalanced-learn
