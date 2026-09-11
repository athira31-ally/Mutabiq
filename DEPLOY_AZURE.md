# Deploying the Compliance Detector to Azure — step by step

This turns the project from "code that runs on my laptop" into a real service with a public HTTPS URL, running on Azure, using Azure's own OCR API instead of the local Tesseract fallback. That's the difference between the current resume bullets and being able to say "deployed on Azure" and actually hand someone a link.

I can't run this against your Azure account for you — it's your subscription, your login, your money (small as it is). What's here is everything to do it yourself in about 20 minutes: the Dockerfile, a script that runs every `az` command in order, and this guide explaining what each step actually does, so you're not just pasting commands blind — you should be able to explain any part of this in an interview.

## What you'll end up with

- A live HTTPS endpoint (`https://trakheesi-api.<random>.azurecontainerapps.io`) that anyone can hit, including an interviewer.
- Real Azure AI Vision doing the OCR (not the local Tesseract fallback) — the `AzureVisionOCR` backend that was written but never exercised against a real Azure resource.
- The container scales to zero when nobody's using it and spins back up on the next request — realistic, cost-aware architecture, not a machine left running 24/7.

## Cost — read this before you start

Being upfront about this since it's coming out of your own account:

- **Azure AI Vision, F0 tier: free.** 5,000 OCR calls/month, 20/minute. A portfolio demo won't get near that.
- **Azure Container Apps, Consumption plan: free grant.** 180,000 vCPU-seconds and 360,000 GiB-seconds per month, plus 2 million requests, before anything is billed. Scaling to zero when idle (which the script sets up) means you're only burning this grant while someone's actually hitting the endpoint.
- **Azure Container Registry, Basic tier: NOT free — about $0.167/day, roughly $5/month.** This gets created automatically by the deploy command to hold your built Docker image. It's the one real recurring cost here, and it bills whether or not the app is being used, because it's just storing the image.
- **Log Analytics workspace: free up to 5 GB/month** of log ingestion. Not a concern at this scale.

**Bottom line:** roughly $5/month if you leave it deployed continuously, effectively $0 if you tear it down between uses. A teardown script is included — it takes one command and a couple of minutes to delete everything. Deploying again later takes about 5 minutes. If you're only using this for interviews/demos over the next few weeks, spin it up before you need it and tear it down after — that's a completely normal way to run a portfolio project, not a shortcut.

If you've just signed up for Azure, you likely also have a $200 / 30-day free credit on top of all this, which comfortably covers it either way.

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

That's genuinely it — the rest of this document is explaining what that script does, step by step, so none of it is a black box.

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

You can also just open `https://$APP_URL/docs` in a browser — FastAPI auto-generates an interactive Swagger page from the route definitions in `src/api.py`, where you can upload an image and hit the endpoint from the browser with no `curl` needed. That page alone is a good thing to have open in an interview.

## Optional: CI/CD with GitHub Actions

`.github/workflows/ci-compliance-detector.yml` is already active and needs nothing from you — it runs the 19 tests on every push to the repo (once it's pushed to GitHub). That alone is a legitimate "CI" line on the resume.

`.github/workflows/deploy-compliance-detector.yml` goes one step further — automatically redeploy to Azure on every push to `main`. It's optional and does nothing until you set up one-time federated login (OIDC), which lets GitHub Actions authenticate to Azure *without* a long-lived password sitting in a GitHub secret:

```bash
az ad app create --display-name trakheesi-github-deploy

# note the appId it prints, then:
APP_ID=<the appId from above>
az ad sp create --id "$APP_ID"

az ad app federated-credential create --id "$APP_ID" --parameters '{
  "name": "github-main-branch",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:YOUR-GITHUB-USERNAME/uae-proptech-portfolio:ref:refs/heads/main",
  "audiences": ["api://AzureADTokenExchange"]
}'

az role assignment create --assignee "$APP_ID" --role Contributor \
  --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/rg-trakheesi-demo"
```

Then add three repo secrets under GitHub → Settings → Secrets and variables → Actions: `AZURE_CLIENT_ID` (the `appId`), `AZURE_TENANT_ID` and `AZURE_SUBSCRIPTION_ID` (both from `az account show`). After that, every push to `main` that touches `projects/01-compliance-detector/` redeploys automatically.

This is genuinely optional — running `./scripts/azure_deploy.sh` by hand whenever you push an update is a completely normal way to operate a portfolio project, and honestly the simpler story to explain in an interview if asked how you deploy it.

## Troubleshooting

- **"Location is not supported for this resource type"** — switch `LOCATION` (and `VISION_LOCATION`) to `westeurope` and re-run; the script is idempotent.
- **"F0 is not available in this region/subscription"** on the Vision resource — you already have a free-tier Vision resource elsewhere on the subscription (only one is allowed). Either reuse its endpoint/key, or change `--sku F0` to `--sku S1` in the script (pay-per-call, ~$1/1,000 images — negligible for a demo).
- **First request after a while returns slowly or times out** — that's the cold start described above; wait a few seconds and retry.
- **`/check-listing` returns 503 "Watermark model not found"** — the model didn't make it into the image. Confirm `models/watermark_yolov8n.onnx` is committed and not excluded anywhere, and that `.dockerignore` isn't ignoring the `models/` folder.
- **`az` commands fail with a permissions/subscription error** — you may be logged into the wrong subscription if you have more than one. Check with `az account list -o table` and switch with `az account set --subscription "<name or id>"`.

## When you're done for now

```bash
./scripts/azure_teardown.sh
```

Deletes the resource group and everything in it. Redeploying later is `./scripts/azure_deploy.sh` again — a few minutes, not a rebuild from scratch.

## Once it's live

Come back and tell me the live URL — I'll fold it into the portfolio README, the project README's status line, and give you the updated resume bullet with "deployed to Azure Container Apps" in it, backed by a link that actually works. Claiming that before it's live and reachable isn't something worth doing — this way it's true the moment it's on your resume.
