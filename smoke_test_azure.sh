#!/usr/bin/env bash
# smoke_test_azure.sh
# Invokes a batch endpoint to create a job. Requires az login/context already set
# and Azure ML defaults (group/workspace) configured or passed via --resource-group/--workspace.
#
# Example:
#   ./smoke_test_azure.sh -e albot-endpoint-dev -d de-model \
#     -i azureml://datastores/workspaceblobstore/paths/batch/input/ \
#     -o azureml://datastores/workspaceblobstore/paths/batch/output/
#
set -euo pipefail

while getopts "e:d:i:o:" opt; do
  case "$opt" in
    e) ENDPOINT_NAME="$OPTARG" ;;
    d) DEPLOYMENT_NAME="$OPTARG" ;;
    i) INPUT="$OPTARG" ;;
    o) OUTPUT="$OPTARG" ;;
    *) exit 1 ;;
  esac
done

: "${ENDPOINT_NAME:?missing -e endpoint name}"
: "${DEPLOYMENT_NAME:?missing -d deployment name}"
: "${INPUT:?missing -i input uri}"
: "${OUTPUT:?missing -o output uri}"

JOB_NAME="$(az ml batch-endpoint invoke \
  -n "$ENDPOINT_NAME" \
  --deployment-name "$DEPLOYMENT_NAME" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --query name -o tsv)"

echo "Created job: $JOB_NAME"
az ml job show -n "$JOB_NAME" --query status -o tsv
az ml job wait -n "$JOB_NAME" --status Completed --timeout 7200
az ml job show -n "$JOB_NAME" --query status -o tsv
