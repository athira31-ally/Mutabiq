#!/usr/bin/env bash
# Put Mutabiq live on Azure Container Apps, using the image GitHub Actions
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

# Wait for GitHub Actions to finish building the image for the commit you're on - deploying before the
# build is done silently ships the previous image.
SHA=$(git rev-parse HEAD 2>/dev/null || true)
if [ -n "$SHA" ] && command -v gh >/dev/null 2>&1; then
  RUN_ID=""
  for _ in $(seq 1 12); do
    RUN_ID=$(gh run list --workflow docker.yml --commit "$SHA" --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null || true)
    [ -n "$RUN_ID" ] && break
    echo ">> Waiting for the docker build of ${SHA:0:7} to start..."; sleep 5
  done
  if [ -n "$RUN_ID" ]; then
    echo ">> Waiting for the docker build of ${SHA:0:7} (run $RUN_ID)"
    gh run watch "$RUN_ID" --exit-status >/dev/null || { echo "!! The docker build failed - not deploying. See: gh run view $RUN_ID --log-failed"; exit 1; }
    echo ">> Image for ${SHA:0:7} is built"
  else
    echo "!! No docker build found for ${SHA:0:7} (not pushed yet?) - deploying whatever :latest is"
  fi
fi

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
