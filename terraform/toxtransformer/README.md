# ToxTransformer Terraform Deployment

## Overview

This Terraform configuration deploys ToxTransformer with:
- **API**: GCE instance with GPU (non-SPOT) running Flask API
- **UI**: Cloud Run service running Streamlit app
- **Load Balancer**: HTTPS with IAP authentication
- **Domain**: toxtransformer.toxindex.com

## Prerequisites

1. **GCP Project**: `toxindex`
2. **Artifact Registry**: `toxindex-backend` repository in `us` region
3. **GCS Bucket**: `toxtransformer-data` with model files
4. **IAP OAuth Credentials**: Create OAuth 2.0 client in GCP Console

### Creating IAP OAuth Credentials

1. Go to GCP Console > APIs & Services > Credentials
2. Create OAuth 2.0 Client ID (Type: Web application)
3. Add authorized redirect URI: `https://iap.googleapis.com/v1/oauth/clientIds/<CLIENT_ID>:handleRedirect`
4. Note the Client ID and Client Secret

## Deployment

### 1. Configure Variables

Create `terraform.tfvars`:

```hcl
git_commit        = "abf5fab"  # Current git commit SHA
image_tag         = "latest"
iap_client_id     = "YOUR_CLIENT_ID.apps.googleusercontent.com"
iap_client_secret = "YOUR_CLIENT_SECRET"
```

Or use environment variables:

```bash
export TF_VAR_git_commit=$(git rev-parse HEAD)
export TF_VAR_iap_client_id="YOUR_CLIENT_ID"
export TF_VAR_iap_client_secret="YOUR_CLIENT_SECRET"
```

### 2. Initialize Terraform

```bash
cd terraform/toxtransformer
terraform init
```

### 3. Plan

```bash
terraform plan
```

### 4. Apply

```bash
terraform apply
```

This will:
- Build and push Docker images (API and Streamlit)
- Create GCE instance with GPU
- Deploy Cloud Run service
- Configure Load Balancer with IAP
- Set up SSL certificate

### 5. Configure DNS

After apply, get the load balancer IP:

```bash
terraform output load_balancer_ip
```

Create an A record for `toxtransformer.toxindex.com` pointing to this IP in your DNS provider.

### 6. Configure IAP Access

Grant IAP access to users:

```bash
gcloud iap web add-iam-policy-binding \
  --resource-type=backend-services \
  --service=toxtransformer-ui-backend \
  --member=user:example@example.com \
  --role=roles/iap.httpsResourceAccessor

gcloud iap web add-iam-policy-binding \
  --resource-type=backend-services \
  --service=toxtransformer-api-backend \
  --member=user:example@example.com \
  --role=roles/iap.httpsResourceAccessor
```

Or use groups:

```bash
gcloud iap web add-iam-policy-binding \
  --resource-type=backend-services \
  --service=toxtransformer-ui-backend \
  --member=group:team@toxindex.com \
  --role=roles/iap.httpsResourceAccessor
```

## Architecture

```
Internet
  ↓
HTTPS Load Balancer (toxtransformer.toxindex.com)
  ├─ IAP Authentication
  ├─ SSL Certificate (Google-managed)
  ├─ / → Cloud Run (Streamlit UI)
  └─ /api/*, /predict_all, /jobs → GCE (Flask API with GPU)
```

## Costs

Estimated monthly costs (non-SPOT):
- GCE n1-standard-4 + T4 GPU: ~$350/month
- Cloud Run (minimal usage): ~$10/month
- Load Balancer: ~$20/month
- **Total**: ~$380/month

## Monitoring

Check API health:

```bash
curl https://toxtransformer.toxindex.com/health
```

View logs:

```bash
# API logs
gcloud compute ssh toxtransformer --zone=us-central1-a
docker logs -f toxtransformer

# Streamlit logs
gcloud run services logs read toxtransformer-ui --region=us-central1
```

## Updating

To update the deployment:

```bash
# Update code
git pull

# Update images and redeploy
export TF_VAR_git_commit=$(git rev-parse HEAD)
terraform apply
```

The GCE startup script will automatically pull and restart with the new image.

## Troubleshooting

### SSL Certificate Provisioning

Google-managed certificates can take up to 15 minutes to provision. Check status:

```bash
gcloud compute ssl-certificates describe toxtransformer-cert --format="get(managed.status)"
```

### IAP Not Working

1. Verify OAuth consent screen is configured
2. Check IAP is enabled: GCP Console > Security > Identity-Aware Proxy
3. Verify IAM bindings for backend services

### API Not Accessible

1. Check firewall rules allow health checks
2. Verify instance is healthy: `terraform output ssh_command`
3. Check Docker container is running: `docker ps`

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

Note: The persistent disk has `prevent_destroy = true` and must be manually deleted if needed.
