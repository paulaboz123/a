- task: AzureCLI@2
  name: deploy_batch_endpoint
  displayName: Deploy Batch Endpoint
  inputs:
    azureSubscription: ${{ parameters.serviceConnection }}
    scriptType: bash
    scriptLocation: inlineScript
    failOnStandardError: false
    workingDirectory: deployment/${{ parameters.environment }}
    inlineScript: |
      set -euo pipefail

      az configure --defaults workspace=${{ parameters.amlWorkspace }} group=${{ parameters.resourceGroup }}

      ENDPOINT_FILE="endpoint-${{ lower(parameters.bu) }}.yml"   # jeśli u Ciebie endpoint.yml -> zmień na "endpoint.yml"
      ENDPOINT_NAME=$(yq -r '.name' "$ENDPOINT_FILE")

      echo "BATCH ENDPOINT: $ENDPOINT_NAME"

      if az ml batch-endpoint show -n "$ENDPOINT_NAME" >/dev/null 2>&1; then
        echo "Update endpoint..."
        az ml batch-endpoint update -f "$ENDPOINT_FILE"
      else
        echo "Create endpoint..."
        az ml batch-endpoint create -f "$ENDPOINT_FILE"
      fi


////////////////////////////




- task: AzureCLI@2
  name: deploy_batch_deployments
  displayName: Deploy Batch Deployments (de/pl/en)
  inputs:
    azureSubscription: ${{ parameters.serviceConnection }}
    scriptType: bash
    scriptLocation: inlineScript
    failOnStandardError: false
    workingDirectory: deployment/${{ parameters.environment }}
    inlineScript: |
      set -euo pipefail

      az configure --defaults workspace=${{ parameters.amlWorkspace }} group=${{ parameters.resourceGroup }}

      MODEL_FILES=("model-de.yml" "model-pl.yml" "model-en.yml")

      for MODEL_FILE in "${MODEL_FILES[@]}"; do
        DEPLOYMENT_NAME=$(yq -r '.name' "$MODEL_FILE")
        ENDPOINT_NAME=$(yq -r '.endpoint_name' "$MODEL_FILE")

        echo "BATCH DEPLOYMENT: $DEPLOYMENT_NAME (endpoint=$ENDPOINT_NAME)"

        if az ml batch-deployment show -e "$ENDPOINT_NAME" -n "$DEPLOYMENT_NAME" >/dev/null 2>&1; then
          echo "Update deployment..."
          az ml batch-deployment update -f "$MODEL_FILE"
        else
          echo "Create deployment..."
          az ml batch-deployment create -f "$MODEL_FILE"
        fi
      done



//////////////////////////



- task: AzureCLI@2
  name: smoke_test_invoke
  displayName: Smoke test - invoke batch endpoint (run job)
  inputs:
    azureSubscription: ${{ parameters.serviceConnection }}
    scriptType: bash
    scriptLocation: inlineScript
    failOnStandardError: false
    workingDirectory: deployment/${{ parameters.environment }}
    inlineScript: |
      set -euo pipefail

      az configure --defaults workspace=${{ parameters.amlWorkspace }} group=${{ parameters.resourceGroup }}

      ENDPOINT_FILE="endpoint-${{ lower(parameters.bu) }}.yml"   # albo endpoint.yml
      ENDPOINT_NAME=$(yq -r '.name' "$ENDPOINT_FILE")

      # wybierasz deployment który chcesz testować (np. de)
      DEPLOYMENT_NAME="AL-de"  # <- zmień jeśli u Ciebie inaczej

      # musisz podać input/output jako azureml://... (folder z plikami json)
      INPUT_URI="${{ parameters.batchInputUri }}"    # dodaj parametr/zmienną w swoim pipeline
      OUTPUT_URI="${{ parameters.batchOutputUri }}"  # dodaj parametr/zmienną w swoim pipeline

      echo "Invoke job..."
      JOB_NAME=$(az ml batch-endpoint invoke \
        -n "$ENDPOINT_NAME" \
        --deployment-name "$DEPLOYMENT_NAME" \
        --input "$INPUT_URI" \
        --output "$OUTPUT_URI" \
        --query name -o tsv)

      echo "JOB: $JOB_NAME"

      az ml job wait -n "$JOB_NAME" --status Completed --timeout 7200

      az ml job show -n "$JOB_NAME" --query status -o tsv
