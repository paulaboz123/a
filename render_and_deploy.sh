#!/usr/bin/env bash
set -euo pipefail

# Local helper (optional). The pipeline already renders+deploys.
# Usage:
#   bash ./render_and_deploy.sh --bu albot --endpoint-name albot-batch-dev --compute gpu-cluster

BU=""
ENDPOINT_NAME=""
COMPUTE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bu) BU="$2"; shift 2 ;;
    --endpoint-name) ENDPOINT_NAME="$2"; shift 2 ;;
    --compute) COMPUTE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$BU" || -z "$ENDPOINT_NAME" || -z "$COMPUTE" ]]; then
  echo "Missing required args."
  exit 1
fi

BU_LOWER="$(echo "$BU" | tr '[:upper:]' '[:lower:]')"
WORKDIR="$(pwd)"
RENDERDIR="$WORKDIR/_rendered"
mkdir -p "$RENDERDIR"

sed -e "s/__ENDPOINT_NAME__/${ENDPOINT_NAME}/g" "$WORKDIR/templates/endpoint.template.yml" > "$RENDERDIR/endpoint-${BU_LOWER}.yml"

for lang in en pl de; do
  sed     -e "s/__DEPLOYMENT_NAME__/${lang}-model/g"     -e "s/__ENDPOINT_NAME__/${ENDPOINT_NAME}/g"     -e "s#__CODE_DIR__#${WORKDIR}/scoring#g"     -e "s/__MODEL_ASSET__/azureml:${lang}@latest/g"     "$WORKDIR/templates/deployment-${lang}.template.yml" > "$RENDERDIR/deployment-${BU_LOWER}-${lang}.yml"
done

az ml batch-endpoint create -f "$RENDERDIR/endpoint-${BU_LOWER}.yml" || az ml batch-endpoint update -f "$RENDERDIR/endpoint-${BU_LOWER}.yml"

for lang in en pl de; do
  file="$RENDERDIR/deployment-${BU_LOWER}-${lang}.yml"
  name="$(yq -r '.name' "$file")"
  az ml batch-deployment create -f "$file" --set compute="azureml:${COMPUTE}" || az ml batch-deployment update -f "$file" --set compute="azureml:${COMPUTE}"
done

echo "Done."
