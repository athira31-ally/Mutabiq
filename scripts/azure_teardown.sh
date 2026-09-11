#!/usr/bin/env bash
# Deletes everything azure_deploy.sh created, in one shot, so nothing keeps
# billing. Deleting the resource group deletes the Container App, its
# environment, the auto-created Container Registry, and the Vision resource
# together — that's the entire cost surface of this project.
#
# Usage:
#   ./scripts/azure_teardown.sh
#
# Run this between interview prep sessions if you want to be extra sure
# nothing is left running — you can always redeploy in a few minutes with
# azure_deploy.sh when you need the live demo again.

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-trakheesi-demo}"

echo "This will permanently delete resource group '$RESOURCE_GROUP' and"
echo "everything in it (Container App, its environment, the container"
echo "registry, and the Vision resource)."
read -r -p "Type the resource group name to confirm: " confirm

if [[ "$confirm" != "$RESOURCE_GROUP" ]]; then
  echo "Names didn't match — aborting, nothing deleted."
  exit 1
fi

az group delete --name "$RESOURCE_GROUP" --yes --no-wait
echo "Deletion started (--no-wait). It'll finish in the background over the"
echo "next few minutes. Check progress any time with:"
echo "  az group show --name $RESOURCE_GROUP"
echo "(once that command errors 'not found', it's fully gone)."
