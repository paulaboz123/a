# Azure ML Batch Endpoint (1 endpoint, 3 deployments) — E2E repo template

This repo is a **complete, runnable template** for deploying an Azure ML **Batch Endpoint** with **3 deployments** (e.g., `en`, `pl`, `de`) via **Azure DevOps YAML pipeline**, and then **submitting a Batch job** (so the analysis actually runs).

## What this repo does
1. Installs Azure ML CLI extension + tools (incl. `yq`)
2. Creates/updates a **Batch Endpoint**
3. Creates/updates **3 Batch Deployments** on that endpoint
4. Submits a **Batch job** via `az ml batch-endpoint invoke` (job submission — not HTTP)

## Directory layout
```
azure-pipelines/
  deploy-batch-endpoint.yml
src/deployment/batch_endpoint/
  dev/
    templates/
      endpoint.template.yml
      deployment-en.template.yml
      deployment-pl.template.yml
      deployment-de.template.yml
    scoring/
      score.py
    sample-data/
      input.csv
  test/ (same structure)
  prod/ (same structure)
```

## Required pipeline inputs (parameters)
- `environment`: `dev` | `test` | `prod`
- `serviceConnection`: Azure DevOps service connection with access to AML workspace + RG
- `resourceGroup`: AML resource group
- `amlWorkspace`: AML workspace name
- `bu`: your BU / product identifier (used for naming)
- `batchCompute`: **AmlCompute cluster name** (required for batch deployments)
- `jobDeploymentName`: which deployment to run job with (e.g. `en-model`, `pl-model`, `de-model`)
- `jobInput`: input data URI
- `jobOutput`: output data URI

### Input/Output URI examples
Use AzureML datastore URIs (recommended):
- `jobInput`:  `azureml://datastores/workspaceblobstore/paths/batch-input/dev/`
- `jobOutput`: `azureml://datastores/workspaceblobstore/paths/batch-output/dev/`

You can also pass a public HTTPS/SAS URI, depending on your setup.

## Notes
- **No default deployment** is configured on the endpoint. You choose the deployment at job submission time (`--deployment-name`).
- Online-specific concepts like **traffic split** and “HTTP invoke key” do not apply to Batch.
- If your scoring logic is currently online-style (request/response), adapt it to Batch: `run(mini_batch: List[str])`.

## Quick local sanity check (optional)
From the repo root (with `az` logged-in):
```bash
az extension add -n ml -y
az configure --defaults group=<RG> workspace=<WS>

# deploy dev endpoint + deployments (renders templates to _rendered/)
cd src/deployment/batch_endpoint/dev
bash ./render_and_deploy.sh \
  --bu albot \
  --endpoint-name albot-batch-dev \
  --compute gpu-cluster
```

The pipeline already does rendering + deploy + invoke; this script is just for local iteration.
