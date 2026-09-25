#!/usr/bin/env bash
# Deploys the Mutabiq API to Azure Container Apps,
# wired up to a real Azure AI Vision resource for OCR.
#
# Read DEPLOY_AZURE.md first — this script is the "just run it" version of
# the same steps that file explains one at a time. Every step is echoed as
# it runs so you can see exactly what's happening and copy any single
# command out if you'd rather run it by hand.
#
# Usage:
#   cd projects/01-compliance-detector
#   az login
#   ./scripts/azure_deploy.sh
#
# Safe to re-run: every az command below is idempotent (create-or-update),
# so running this twice won't create duplicate resources.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — override any of these by setting the env var before running, e.g.
#   RESOURCE_GROUP=rg-my-demo LOCATION=westeurope ./scripts/azure_deploy.sh
# ---------------------------------------------------------------------------
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-trakheesi-demo}"
LOCATION="${LOCATION:-uaenorth}"
VISION_LOCATION="${VISION_LOCATION:-$LOCATION}"
VISION_NAME="${VISION_NAME:-trakheesi-vision}"
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-trakheesi-env}"
APP_NAME="${APP_NAME:-trakheesi-api}"

echo "=================================================================="
echo " Mutabiq — Azure deploy"
echo "   Resource group : $RESOURCE_GROUP"
echo "   Location       : $LOCATION"
echo "   Vision resource: $VISION_NAME"
echo "   Container App  : $APP_NAME"
echo "=================================================================="

# ---------------------------------------------------------------------------
# Step 0 — sanity checks
# ---------------------------------------------------------------------------
if ! command -v az >/dev/null 2>&1; then
  echo "Azure CLI ('az') isn't installed. See DEPLOY_AZURE.md, Prerequisites." >&2
  exit 1
fi

if ! az account show >/dev/null 2>&1; then
  echo "Not logged in. Run 'az login' first." >&2
  exit 1
fi

echo ""
echo "--> Logged in as:"
az account show --query "{name:name, id:id}" -o table

echo ""
echo "--> Making sure the containerapp CLI extension is installed..."
az extension add --name containerapp --upgrade -y --only-show-errors

echo ""
echo "--> Registering the Azure resource providers this deploy needs"
echo "    (one-time per subscription; instant if already registered)..."
for provider in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry Microsoft.CognitiveServices; do
  az provider register --namespace "$provider" --wait
done

# ---------------------------------------------------------------------------
# Step 1 — resource group
# ---------------------------------------------------------------------------
echo ""
echo "--> Step 1/5: creating resource group '$RESOURCE_GROUP'..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" -o table

# ---------------------------------------------------------------------------
# Step 2 — Azure AI Vision resource (OCR backend), free F0 tier
# ---------------------------------------------------------------------------
echo ""
echo "--> Step 2/5: creating Azure AI Vision resource '$VISION_NAME' (F0 = free tier)..."
if ! az cognitiveservices account show --name "$VISION_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az cognitiveservices account create \
    --name "$VISION_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --kind ComputerVision \
    --sku F0 \
    --location "$VISION_LOCATION" \
    --yes \
    -o table
else
  echo "    already exists, skipping creation."
fi

VISION_ENDPOINT=$(az cognitiveservices account show \
  --name "$VISION_NAME" --resource-group "$RESOURCE_GROUP" \
  --query "properties.endpoint" -o tsv)
VISION_KEY=$(az cognitiveservices account keys list \
  --name "$VISION_NAME" --resource-group "$RESOURCE_GROUP" \
  --query "key1" -o tsv)

echo "    endpoint: $VISION_ENDPOINT"
echo "    key:      ${VISION_KEY:0:6}************ (fetched, not printed in full)"

# ---------------------------------------------------------------------------
# Step 3 — build the image from source and deploy it as a Container App
# ---------------------------------------------------------------------------
echo ""
echo "--> Step 3/5: building the image from source and deploying (this is the"
echo "    slow step — cloud build via ACR Tasks, typically 3-6 minutes)..."
az containerapp up \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --environment "$ENVIRONMENT_NAME" \
  --source . \
  --ingress external \
  --target-port 8000

# ---------------------------------------------------------------------------
# Step 4 — wire the Vision credentials in as secrets, keep min replicas at 0
# ---------------------------------------------------------------------------
echo ""
echo "--> Step 4/5: setting Azure Vision credentials as container secrets..."
az containerapp secret set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --secrets "vision-endpoint=$VISION_ENDPOINT" "vision-key=$VISION_KEY"

az containerapp update \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --min-replicas 0 \
  --max-replicas 2 \
  --set-env-vars \
    "AZURE_VISION_ENDPOINT=secretref:vision-endpoint" \
    "AZURE_VISION_KEY=secretref:vision-key" \
  -o none

# ---------------------------------------------------------------------------
# Step 5 — print the live URL and a ready-to-run test command
# ---------------------------------------------------------------------------
APP_URL=$(az containerapp show \
  --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" -o tsv)

echo ""
echo "=================================================================="
echo " Deployed."
echo ""
echo " Live URL:   https://$APP_URL"
echo " Health:     curl https://$APP_URL/health"
echo " Docs (UI):  https://$APP_URL/docs"
echo ""
echo " Try it for real:"
echo "   curl -X POST https://$APP_URL/check-listing \\"
echo "     -F listing_id=demo-1 -F agent_id=agent-1 \\"
echo "     -F claimed_permit_number=DLD-73920 \\"
echo "     -F images=@/path/to/any/listing/photo.jpg"
echo ""
echo " Note: min-replicas is 0, so the first request after idle time will"
echo " take a few seconds (cold start) while it spins the container back up."
echo ""
echo " When you're done demoing for a while, run scripts/azure_teardown.sh"
echo " to delete everything and stop billing."
echo "=================================================================="
