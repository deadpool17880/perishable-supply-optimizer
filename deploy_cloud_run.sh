#!/usr/bin/env bash
set -e

SERVICE_NAME="perishable-supply-optimizer"
REGION="us-central1"

echo "=========================================================="
echo "Deploying DPSRO Dashboard to Google Cloud Run"
echo "=========================================================="

if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed or not in PATH."
    echo "Install gcloud: https://cloud.google.com/sdk/docs/install"
    echo ""
    echo "Alternatively, you can deploy using Cloud Shell or Cloud Console:"
    echo "1. Push this repository to GitHub: https://github.com/deadpool17880/perishable-supply-optimizer"
    echo "2. Open Google Cloud Run: https://console.cloud.google.com/run"
    echo "3. Click 'Create Service' -> 'Continuously deploy from a repository' -> Select GitHub repo"
    echo "4. Set Port: 8080, Memory: 2Gi, Allow unauthenticated invocations"
    exit 1
fi

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo "Error: No active GCP project configured."
    echo "Run: gcloud config set project <YOUR_PROJECT_ID>"
    exit 1
fi

echo "Active Project: $PROJECT_ID"
echo "Target Region:  $REGION"
echo "Service Name:   $SERVICE_NAME"
echo ""

echo "Deploying directly from source using Cloud Build..."
gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --set-env-vars STREAMLIT_SERVER_HEADLESS=true,STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo ""
echo "Deployment successful! Service URL:"
gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format="value(status.url)"
