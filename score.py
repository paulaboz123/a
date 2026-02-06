name: albot-data-v2
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip>=24
  - regex>=2024.5
  - numpy=1.26.*
  - pandas=2.2.*
  - openpyxl=3.1.*
  - fsspec>=2024.2
  - ftfy>=6.2
  - pip:
      - azure-ai-ml>=1.22,<2
      - azure-identity>=1.16,<2
mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest
