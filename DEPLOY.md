# Deploying the AP Copilot demo to Cloud Run (private, Google sign-in)

Phase 2 runbook. Every command below was checked against the current Google docs (links at the end) and, where gcloud 562.0.0 was available locally, against `--help`. Nothing here has been run: this repo change is code only.

Target: project `palamo-demos`, region `us-west1`, service `ap-copilot-demo`. Access is Google sign-in through Identity-Aware Proxy (IAP), restricted to `halamo@palamo.ai` and `rpalma@palamo.ai`. The service is never public.

## What you are deploying

- The dashboard (`web/server.py`) with `AP_DEMO_MODE=1`: no cost/ROI banner, and the fixed notice "Test build · synthetic data only · results are not financial advice".
- The document path: a visitor uploads a PDF/PNG/JPEG (max 5 MB) or picks a synthetic corpus sample, and sees reader, LLM extraction, GL coding with reasons, validation, the Human Gate and the decision trail. Uploaded files live in a private temp dir and are deleted as soon as the Intake node has read them.
- Live extraction and GL coding with Claude Haiku 5.5 (`claude-haiku-5-5`), through the key mounted from Secret Manager.

## Spend protection: read this before deploying

There are three layers. Only the last one is a cross-instance stop.

| Layer | What it does | Limit |
|---|---|---|
| Per-instance ledger cap `AP_LLM_SPEND_CAP_USD=2` | Refuses a model call once this instance's ledger would pass $2. The ledger is `/tmp/ap-spend/spend.json` (`AP_SPEND_LEDGER_PATH`) and **resets on every cold start**, because the container filesystem is ephemeral. | Per instance lifetime |
| `--max-instances=1` | Only one instance can exist, so there is one ledger at a time and no parallel copies. | Cloud Run |
| Anthropic workspace spend limit of $10 | **The real stop.** Enforced by Anthropic across every instance and every restart. | Per calendar month |

Facts about the Anthropic limit, from the rate-limits page: spend limits are monthly; a limit you set is set on the Billing page; when a workspace limit is reached, requests return HTTP 400 `invalid_request_error` ("You have reached your specified workspace API usage limits"); and **limits cannot be set on the default Workspace**. So the key in Secret Manager must belong to a non-default workspace that carries the $10 limit. The app treats that 400 as a failed call (empty fields, human review), never as a value.

Per-request guards in the app: one document per request (a second file or extra field is refused with 400), at most `AP_MAX_CONCURRENT_UPLOADS` (default 2) documents processing at once (429 beyond that), and a fixed call path of one extraction call plus one GL call per document with no loop.

## 0. Before you start (Hector, once)

1. In the Claude Console, create a **dedicated workspace** (not the default one) for this demo, set its monthly spend limit to $10, and create an API key inside it. Put that key in `~/.config/ap-intake/anthropic.env` as `ANTHROPIC_API_KEY=...` (mode 600). The file may also hold other lines; only the key is read.
2. Confirm `palamo-demos` belongs to the palamo.ai organization:
   ```bash
   gcloud projects describe palamo-demos --format='value(parent.type,parent.id)'
   ```
   An empty result means the project has no organization, and IAP then needs the extra one-time OAuth client setup described in the IAP doc (see "Projects without an organization" in the sources). Stop and do that setup first.
3. Your gcloud account needs these roles on the project: Cloud Run Admin, IAP Policy Admin, Artifact Registry Reader, Service Account User (IAP doc), plus Cloud Run Source Developer, Service Usage Consumer, Service Account User (source-deploy doc), plus permission to create service accounts and secrets.
4. Authenticate and select the project:
   ```bash
   gcloud auth login
   gcloud config set project palamo-demos
   ```

## 1. Variables

```bash
export PROJECT=palamo-demos
export REGION=us-west1
export SERVICE=ap-copilot-demo
export SECRET=anthropic-api-key
export RUNTIME_SA_NAME=ap-demo-runtime
export RUNTIME_SA="${RUNTIME_SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
export PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
```

## 2. Enable the APIs

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  iap.googleapis.com \
  --project="$PROJECT"
```

## 3. Runtime identity and the secret

A dedicated service account means the app can read only this one secret.

```bash
gcloud iam service-accounts create "$RUNTIME_SA_NAME" \
  --project="$PROJECT" \
  --display-name="AP Copilot demo runtime"
```

Create the secret from the local env file by piping it. The key goes through a shell variable and a pipe; it is never echoed, written to disk or put on a command line. Run this from the repo root with its virtualenv active (`source .venv/bin/activate`), so `python-dotenv` is available.

```bash
KEY="$(python -c 'import sys; from dotenv import dotenv_values; sys.stdout.write((dotenv_values(sys.argv[1]).get("ANTHROPIC_API_KEY") or "").strip())' "$HOME/.config/ap-intake/anthropic.env")"
if [ -n "$KEY" ]; then
  printf '%s' "$KEY" | gcloud secrets create "$SECRET" \
    --project="$PROJECT" \
    --replication-policy=automatic \
    --data-file=-
else
  echo "ANTHROPIC_API_KEY not found in the env file; no secret was created" >&2
fi
unset KEY
```

The `if` means `gcloud` never runs when the key is missing, so an empty secret is never created. Grant the runtime account access to this one secret:

```bash
gcloud secrets add-iam-policy-binding "$SECRET" \
  --project="$PROJECT" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/secretmanager.secretAccessor"
```

Rotating the key later: `printf '%s' "$NEW" | gcloud secrets versions add "$SECRET" --data-file=-`, then redeploy with `${SECRET}:2`. The deploy pins a version (`:1`) because Google recommends pinning for environment-variable secrets, which are resolved at instance start.

## 4. Build and deploy (private, IAP on)

Run from the repo root. `--source .` builds with the repo `Dockerfile` through Cloud Build and creates the `cloud-run-source-deploy` Artifact Registry repository on first use. `.gcloudignore` keeps `.env` files, `eval/out/` and the answer key out of the upload.

```bash
gcloud run deploy "$SERVICE" \
  --project="$PROJECT" \
  --region="$REGION" \
  --source=. \
  --no-allow-unauthenticated \
  --iap \
  --max-instances=1 \
  --min-instances=0 \
  --cpu=1 \
  --memory=2Gi \
  --cpu-boost \
  --no-cpu-throttling \
  --timeout=300 \
  --concurrency=10 \
  --service-account="$RUNTIME_SA" \
  --set-secrets="ANTHROPIC_API_KEY=${SECRET}:1" \
  --set-env-vars="AP_DEMO_MODE=1,AP_LLM_PROVIDER=anthropic,AP_LLM_MODEL=claude-haiku-5-5,AP_LLM_SPEND_CAP_USD=2,AP_MAX_CONCURRENT_UPLOADS=2"
```

Why these flags:

- `--no-allow-unauthenticated --iap`: no public access; callers go through IAP. This is the "new service" form in the IAP doc.
- `--max-instances=1`: one instance, one spend ledger, and one in-memory session store. Sessions live in process memory, so an instance that scales to zero (about 15 minutes idle) forgets a run paused at the Human Gate; the visitor sees "Session not found" and re-runs the invoice.
- `--min-instances=0`: no idle cost.
- `--no-cpu-throttling`: the workflow runs as a background task after the upload request returns. With the default request-based CPU, the instance is throttled between the dashboard's polls and runs slowly.
- `--memory=2Gi`: ADK plus Tesseract on a 150 dpi page; 512 Mi is not enough headroom.
- `AP_LLM_SPEND_CAP_USD=2`: the per-instance cap described above.

If the deploy prints a warning about enabling IAP in a project without an organization, stop: that is the case from step 0.2.

## 5. IAP access

Give the IAP service agent permission to invoke the service. The agent exists once IAP has been enabled on a service; if the binding below fails with "service account does not exist", create the agent first (the doc shows this form without `beta`; gcloud 562.0.0 had no GA `services identity` group, so `beta` is used here, which may prompt you to install the `beta` component):

```bash
gcloud beta services identity create --service=iap.googleapis.com --project="$PROJECT"
```

Then bind the invoker role:

```bash
gcloud run services add-iam-policy-binding "$SERVICE" \
  --project="$PROJECT" \
  --region="$REGION" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-iap.iam.gserviceaccount.com" \
  --role="roles/run.invoker"
```

Allow the two testers through IAP:

```bash
for USER_EMAIL in halamo@palamo.ai rpalma@palamo.ai; do
  gcloud iap web add-iam-policy-binding \
    --project="$PROJECT" \
    --member="user:${USER_EMAIL}" \
    --role="roles/iap.httpsResourceAccessor" \
    --region="$REGION" \
    --resource-type=cloud-run \
    --service="$SERVICE"
done
```

If a sign-in later shows an out-of-organization error, the IAP troubleshooting section says to check the OAuth brand: out-of-org access is not supported while the brand is Internal.

## 6. Verify

Unauthenticated requests must be refused. The only failures are an HTTP 200 or any body that contains the dashboard.

```bash
URL="$(gcloud run services describe "$SERVICE" --project="$PROJECT" --region="$REGION" --format='value(status.url)')"
BODY="$(mktemp)"
for PATH_PART in / /api/config /api/samples; do
  CODE="$(curl -s -o "$BODY" -w '%{http_code}' "${URL}${PATH_PART}")"
  if [ "$CODE" = "200" ] || grep -q "AP Copilot" "$BODY"; then echo "FAIL ${PATH_PART}: HTTP ${CODE}"; else echo "ok   ${PATH_PART}: HTTP ${CODE}"; fi
done
rm -f "$BODY"
```

Expect a redirect (302) to Google sign-in, or 401/403. The exact status IAP returns to a cookie-less curl was not observed while writing this; the pass rule above does not depend on it.

Then, in a browser:

1. Open `$URL` signed in as `halamo@palamo.ai` (and again as `rpalma@palamo.ai`). Expected: the Google sign-in page, then the dashboard with the "Test build" notice and no cost/ROI banner.
2. Open it signed in as any other Google account. Expected: IAP's "You don't have access" page.
3. Upload one synthetic corpus invoice (or pick a sample). Expected: the pipeline runs, the right panel fills with extracted fields, GL codes with reasons, validation flags and the trail; GL-Coder shows "LLM coder + keyword fallback".

Check spend after the first runs in the Claude Console usage page for the demo workspace.

## 7. Updating, and turning it off

Redeploy after a code change with the same step 4 command. To switch the model without a rebuild:

```bash
gcloud run services update "$SERVICE" --project="$PROJECT" --region="$REGION" \
  --update-env-vars="AP_LLM_MODEL=claude-haiku-5-5"
```

To stop the demo: remove access first, then the service, then the key.

```bash
gcloud run services delete "$SERVICE" --project="$PROJECT" --region="$REGION"
gcloud secrets delete "$SECRET" --project="$PROJECT"
```

Also delete the API key (or the whole demo workspace) in the Claude Console; deleting the Google secret does not revoke it.

## Open points to confirm on the first live run

- The exact HTTP status an unauthenticated curl receives from IAP on Cloud Run (step 6 accepts 302, 401 and 403).
- That `us-west1` supports IAP directly on Cloud Run. The IAP doc lists no region restriction; the deploy fails with a clear error if one exists.
- Whether the build needs `roles/run.builder` on the Compute Engine default service account (source-deploy doc). Cloud Build runs as that account for source deploys; grant it if the build is refused.

## Sources (checked 2026-10-07)

- IAP for Cloud Run (enable with `gcloud run deploy --iap`, service-agent invoker binding, `gcloud iap web add-iam-policy-binding ... --resource-type=cloud-run`, projects without an organization, troubleshooting): https://docs.cloud.google.com/run/docs/securing/identity-aware-proxy-cloud-run
- IAP service agent creation (`gcloud services identity create --service=iap.googleapis.com`): https://docs.cloud.google.com/iap/docs/enabling-cloud-run
- Cloud Run secrets (`--set-secrets`, `roles/secretmanager.secretAccessor`, pin versions for env vars): https://docs.cloud.google.com/run/docs/configuring/services/secrets
- Cloud Run source deploys (Dockerfile build, `cloud-run-source-deploy` repository, deployer and builder roles): https://docs.cloud.google.com/run/docs/deploying-source-code
- Anthropic spend limits (monthly, workspace limit returns 400, no limits on the default workspace): https://platform.claude.com/docs/en/api/rate-limits
- Local checks: `gcloud run deploy --help` (flags `--[no-]iap`, `--[no-]allow-unauthenticated`, `--[no-]cpu-throttling`, `--[no-]cpu-boost`, `--set-secrets`), `gcloud secrets create --help` (`--data-file=-` reads stdin), `gcloud iap web add-iam-policy-binding --help` (`--resource-type`, `--region`, `--service`), gcloud 562.0.0.
