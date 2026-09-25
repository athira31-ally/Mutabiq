#!/usr/bin/env bash
# Turn on the Azure AI services for the live app:
#   - Azure AI Document Intelligence: reads a listing page saved as PDF (text + permit QR codes)
#   - Azure AI Vision (Read OCR):     reads printed permit numbers on photos
# Both start on the free tier (F0). Without them the app still works, using local Tesseract / zxing.
#
#   bash scripts/enable_azure_ai.sh
#   SKU=S0 bash scripts/enable_azure_ai.sh              # if your subscription already has a free one of either kind
#   LOCATION=westeurope bash scripts/enable_azure_ai.sh # if a service isn't offered in UAE North
set -euo pipefail

RG=rg-trakheesi-demo
APP=trakheesi-api
LOCATION=${LOCATION:-uaenorth}
SKU=${SKU:-F0}
SUFFIX=$(az account show --query id -o tsv | tr -d '-' | cut -c1-6)
DI=trakheesi-docintel-$SUFFIX
VISION=trakheesi-vision-$SUFFIX

create() {
  local name=$1 kind=$2
  if az cognitiveservices account show -n "$name" -g "$RG" >/dev/null 2>&1; then
    echo ">> $name already exists"
  else
    echo ">> Creating $name ($kind, $SKU, $LOCATION)"
    az cognitiveservices account create -n "$name" -g "$RG" -l "$LOCATION" --kind "$kind" --sku "$SKU" \
      --custom-domain "$name" --yes -o none
  fi
}

create "$DI" FormRecognizer
create "$VISION" ComputerVision

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
echo "Done. Document Intelligence: $DI_ENDPOINT"
echo "      AI Vision:             $VISION_ENDPOINT"
echo "The app restarts with both on; the result panel shows 'read from PDF (azure-document-intelligence)'."
