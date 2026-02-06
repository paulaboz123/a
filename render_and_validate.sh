#!/usr/bin/env bash
set -euo pipefail

# Local helper: validate that YAML files exist for env and BU.
ENV="${1:-dev}"
BU="${2:-albot}"

DIR="src/deployment/batch_endpoint/${ENV}"
echo "Checking ${DIR} for BU=${BU}"

ls -la "${DIR}"
test -f "${DIR}/endpoint-${BU}.yml"
test -f "${DIR}/deployment-${BU}-de.yml"
test -f "${DIR}/deployment-${BU}-pl.yml"
test -f "${DIR}/deployment-${BU}-en.yml"

echo "OK"
