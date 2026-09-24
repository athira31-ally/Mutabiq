#!/usr/bin/env bash
# Put the Trakheesi Compliance Detector live on Azure Container Apps, using the image GitHub Actions
# builds (ghcr.io) - no ACR / ACR Tasks needed. Safe to re-run: creates the app once, then just updates it.
#
#   bash scripts/deploy_live.sh
#
# Reuses the existing Container Apps environment (your subscription allows one per region).
# Scales to zero when idle, so it costs ~nothing between demos.
set -euo pipefail

RG=${RESOURCE_GROUP:-rg-trakheesi-demo}
ENV_NAME=${ENVIRONMENT_NAME:-trakheesi-env}
APP=${APP_NAME:-trakheesi-api}
IMAGE=${IMAGE:-ghcr.io/athira31-ally/trakheesi-api:latest}
exists() { "$@" -o none >/dev/null 2>&1; }

echo ">> Container Apps environment $ENV_NAME ($RG)"
exists az containerapp env show -n "$ENV_NAME" -g "$RG" || { echo "!! environment $ENV_NAME not found in $RG"; exit 1; }

if exists az containerapp show -n "$APP" -g "$RG"; then
  echo ">> Updating $APP to $IMAGE"
  az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" \
    --set-env-vars IMAGE_SHA=$(git rev-parse --short HEAD 2>/dev/null || date +%s) -o none
else
  echo ">> Creating $APP from $IMAGE"
  az containerapp create -n "$APP" -g "$RG" --environment "$ENV_NAME" --image "$IMAGE" \
    --target-port 8000 --ingress external --min-replicas 0 --max-replicas 1 \
    --cpu 1.0 --memory 2.0Gi -o none
fi

URL=https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo ""
echo "Live:  $URL          (demo page - the first load after idle takes ~20 s)"
echo "API:   $URL/docs"
echo "Check: curl -s $URL/health"
