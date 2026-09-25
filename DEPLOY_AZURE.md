# Deploying Mutabiq to Azure — step by step

## Prerequisites

1. **An Azure account.** Free signup at azure.com if you don't have one — needs a card on file (for identity verification) but the free tier above genuinely doesn't charge it for this project.
2. **Azure CLI installed.**
   ```bash
   brew install azure-cli
   ```
3. **Logged in:**
   ```bash
   az login
   ```
   This opens a browser, you sign in, and the CLI remembers you.
4. **The project itself, already working locally** — if you haven't been through `GETTING_STARTED.md` yet, do that first (specifically Part 1, Steps 1–2, so `models/watermark_yolov8n.onnx` exists). It's already committed to the repo, so you likely don't need to retrain — just confirm the file is there:
   ```bash
   ls -lh models/watermark_yolov8n.onnx
   ```

## The one-command version

```bash
cd projects/01-compliance-detector
az login
./scripts/azure_deploy.sh
```



## Step by step, explained

### Step 1 — Resource group

```bash
az group create --name rg-trakheesi-demo --location uaenorth
```

A resource group is just a folder Azure uses to keep related things together — the container app, its registry, the Vision resource all go inside it. The entire point of doing this is that **cleanup becomes one command**: delete the resource group, and everything inside it is deleted with it. No hunting down five separate resources later.

`uaenorth` is Azure's UAE North region (Dubai). If it ever rejects a resource type as unsupported in that region (Container Apps support varies by region over time), switch to `westeurope` — for a portfolio project the physical region doesn't change what you can say about it; "designed for UAE deployment, provisioned in the nearest supported Azure region" is a fine, honest sentence if that comes up.

### Step 2 — The Azure AI Vision resource (real OCR)

```bash
az cognitiveservices account create \
  --name trakheesi-vision \
  --resource-group rg-trakheesi-demo \
  --kind ComputerVision \
  --sku F0 \
  --location uaenorth \
  --yes
```

This provisions an actual Azure AI Vision endpoint. It's the cloud service `AzureVisionOCR.extract_text()` in `src/ocr_permit.py` calls — the code was written and ready from the start, it just had nothing real to talk to in the sandbox this was built in. `F0` is the free SKU (subject to Azure's one-free-resource-per-subscription-per-region-and-kind limit — if you already have another F0 Computer Vision resource somewhere, either reuse it or switch this one to `S1`, which is pay-per-call and cheap: about $1 per 1,000 images).

Two things come out of this resource that the app needs:

```bash
az cognitiveservices account show --name trakheesi-vision --resource-group rg-trakheesi-demo \
  --query "properties.endpoint" -o tsv

az cognitiveservices account keys list --name trakheesi-vision --resource-group rg-trakheesi-demo \
  --query "key1" -o tsv
```

The **endpoint** is the URL to call; the **key** authenticates you as the owner of that resource (like a password scoped to this one service). Both get wired into the running container in Step 4 — never put the key directly in code or commit it to git.

### Step 3 — Build and deploy

```bash
az containerapp up \
  --name trakheesi-api \
  --resource-group rg-trakheesi-demo \
  --location uaenorth \
  --environment trakheesi-env \
  --source . \
  --ingress external \
  --target-port 8000
```

This single command does four things, in order:

1. **Builds the Docker image from `Dockerfile` in the cloud** using ACR Tasks — Azure spins up a build machine, runs `docker build` there, and throws the machine away when it's done. You don't need Docker installed locally for this to work (though you can also `docker build` locally and push yourself if you'd rather — same Dockerfile either way).
2. **Creates a Container Registry** to store that built image (this is the ~$5/month piece from the cost section above).
3. **Creates a Container Apps Environment** (`trakheesi-env`) — think of this as the secure network boundary your app runs inside; you could put multiple container apps in one environment later and they'd share logging/networking.
4. **Creates the Container App itself** and points a public HTTPS URL at port 8000 (`--target-port 8000`, matching `EXPOSE 8000` / `uvicorn --port 8000` in the Dockerfile), with `--ingress external` meaning that URL is reachable from the open internet, not just from inside Azure.

This is the slow step — 3 to 6 minutes, mostly the cloud build.

### Step 4 — Wiring in the Vision credentials

```bash
az containerapp secret set --name trakheesi-api --resource-group rg-trakheesi-demo \
  --secrets "vision-endpoint=$VISION_ENDPOINT" "vision-key=$VISION_KEY"

az containerapp update --name trakheesi-api --resource-group rg-trakheesi-demo \
  --min-replicas 0 --max-replicas 2 \
  --set-env-vars \
    "AZURE_VISION_ENDPOINT=secretref:vision-endpoint" \
    "AZURE_VISION_KEY=secretref:vision-key"
```

Azure Container Apps has two ways to pass configuration in: plain environment variables (visible to anyone who can run `az containerapp show`), and **secrets**, which are encrypted at rest and only referenced by name (`secretref:vision-key`) rather than exposed in plaintext anywhere. The Vision key is a credential, so it goes in as a secret, not a plain env var — the same instinct as never committing an API key to git.

Once these two env vars (`AZURE_VISION_ENDPOINT`, `AZURE_VISION_KEY`) exist in the container's environment, the app doesn't need a code change or redeploy to start using them — go back and look at `get_backend("auto")` in `src/ocr_permit.py`: it already checks `os.environ.get("AZURE_VISION_KEY")` and picks `AzureVisionOCR` automatically when it's set, falling back to `TesseractOCR` otherwise. That's the whole reason the pluggable-backend design existed from the start — this moment is what it was for.

`--min-replicas 0` is what makes the "scales to zero" cost story real: with nobody hitting the endpoint, Azure eventually shuts the container down completely (zero cost while stopped), and starts a new one on the next incoming request. The tradeoff is a **cold start** — the first request after idle time takes a few extra seconds while the container boots and loads the ONNX model into memory. `--max-replicas 2` caps how many copies it can scale *up* to under load, which for a demo just prevents a runaway cost surprise.

### Step 5 — Test it

```bash
APP_URL=$(az containerapp show --name trakheesi-api --resource-group rg-trakheesi-demo \
  --query "properties.configuration.ingress.fqdn" -o tsv)

curl "https://$APP_URL/health"
```

Should return `{"status":"ok"}`. Then the real thing:

```bash
curl -X POST "https://$APP_URL/check-listing" \
  -F listing_id=demo-1 -F agent_id=agent-1 \
  -F claimed_permit_number=DLD-73920 \
  -F images=@/path/to/any/listing/photo.jpg
```



