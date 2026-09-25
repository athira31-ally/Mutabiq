#!/usr/bin/env bash
# Turn on the Azure AI services for the live app:
#   - Azure AI Document Intelligence: reads a listing page saved as PDF (text + permit QR codes)
#   - Azure AI Vision (Read OCR):     reads printed permit numbers on photos
# Reuses what the resource group already has (a Document Intelligence / Vision resource, or a multi-service
# "AIServices" resource, which serves both from one endpoint) and only creates what is missing.
# Without these the app still works, using local Tesseract / zxing.
#
#   bash scripts/enable_azure_ai.sh
#   SKU=S0 bash scripts/enable_azure_ai.sh     # if a new resource is needed and the free tier (F0) is taken
set -euo pipefail

RG=rg-trakheesi-demo
APP=trakheesi-api
LOCATION=${LOCATION:-uaenorth}
SKU=${SKU:-F0}
SUFFIX=$(az account show --query id -o tsv | tr -d '-' | cut -c1-6)

# Clean up resources a failed attempt left behind (state "Failed"), including their soft-deleted copy.
for name in $(az cognitiveservices account list -g "$RG" --query "[?properties.provisioningState=='Failed'].name" -o tsv); do
  loc=$(az cognitiveservices account show -n "$name" -g "$RG" --query location -o tsv)
  echo ">> Removing failed resource $name"
  az cognitiveservices account delete -n "$name" -g "$RG" -o none
  az cognitiveservices account purge -n "$name" -g "$RG" -l "$loc" -o none || true
done

existing() {  # first working resource in the RG whose kind is in the given list (in order of preference)
  local kind name
  for kind in "$@"; do
    name=$(az cognitiveservices account list -g "$RG" \
      --query "[?kind=='$kind' && properties.provisioningState=='Succeeded'].name | [0]" -o tsv)
    if [ -n "$name" ]; then echo "$name"; return; fi
  done
}

ensure() {  # ensure <new-name> <kind to create> <acceptable kinds...>
  local new=$1 create_kind=$2; shift 2
  local name
  name=$(existing "$@")
  if [ -z "$name" ]; then
    echo ">> Creating $new ($create_kind, $SKU, $LOCATION)" >&2
    az cognitiveservices account create -n "$new" -g "$RG" -l "$LOCATION" --kind "$create_kind" --sku "$SKU" \
      --custom-domain "$new" --yes -o none
    name=$new
  else
    echo ">> Using existing $name for $create_kind" >&2
  fi
  echo "$name"
}

DI=$(ensure "trakheesi-docintel-$SUFFIX" FormRecognizer FormRecognizer AIServices CognitiveServices)
VISION=$(ensure "trakheesi-vision-$SUFFIX" ComputerVision ComputerVision AIServices CognitiveServices)

DI_ENDPOINT=$(az cognitiveservices account show -n "$DI" -g "$RG" --query properties.endpoint -o tsv)
DI_KEY=$(az cognitiveservices account keys list -n "$DI" -g "$RG" --query key1 -o tsv)
VISION_ENDPOINT=$(az cognitiveservices account show -n "$VISION" -g "$RG" --query properties.endpoint -o tsv)
VISION_KEY=$(az cognitiveservices account keys list -n "$VISION" -g "$RG" --query key1 -o tsv)

echo ">> Storing the keys as Container App secrets and wiring the env vars"
az containerapp secret set -n "$APP" -g "$RG" --secrets "docintel-key=$DI_KEY" "vision-key=$VISION_KEY" -o none
az containerapp update -n "$APP" -g "$RG" -o none --set-env-vars \
  "AZURE_DOCINTEL_ENDPOINT=$DI_ENDPOINT" "AZURE_DOCINTEL_KEY=secretref:docintel-key" \
  "AZURE_VISION_ENDPOINT=$VISION_ENDPOINT" "AZURE_VISION_KEY=secretref:vision-key"

echo
echo "Done. Document Intelligence: $DI ($DI_ENDPOINT)"
echo "      AI Vision:             $VISION ($VISION_ENDPOINT)"
echo "The app restarts with both on; a PDF check then shows 'read from PDF (azure-document-intelligence)'."
