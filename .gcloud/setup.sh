REPO_NAME="ai-biobricks-chemprop-transformer"

# Check if the Artifact Registry repo exists, if not, create it
if ! gcloud artifacts repositories describe $REPO_NAME --location=us-central1 >/dev/null 2>&1; then
  gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository"
fi

# Set your variables
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
IMAGE_NAME="toxtransformer"
TAG="$(git rev-parse --abbrev-ref HEAD)-$(git rev-parse --short HEAD)"

# Build
docker build -t $IMAGE_NAME .

# rename the image_name to 'toxtransformer'
docker tag $IMAGE_NAME $IMAGE_NAME:$TAG

# Tag for Artifact Registry
docker tag $IMAGE_NAME $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG

# Authenticate Docker with Artifact Registry
gcloud auth configure-docker $REGION-docker.pkg.dev

# Push
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG

# check that it was pushed
gcloud artifacts docker images list $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME

# Write startup script to a temporary file
STARTUP_TMP_FILE=$(mktemp)
cat > "$STARTUP_TMP_FILE" <<EOF
#!/bin/bash
docker pull us-central1-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG
docker run --gpus all -p 6515:6515 --rm \
  us-central1-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG
EOF

# Create VM with startup script
gcloud compute instances create toxtransformer-test-a2 \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=count=1,type=nvidia-tesla-a100 \
  --image=pytorch-latest-gpu-v20250325-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --boot-disk-size=200GB \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --tags=http-server \
  --metadata-from-file startup-script="$STARTUP_TMP_FILE"

# kill the instance
gcloud compute instances delete toxtransformer-test-a2 --zone=us-central1-a


# Clean up temp file on script exit
# trap 'rm -f "$STARTUP_TMP_FILE"' EXIT

# # test it
# gcloud compute ssh toxtransformer-test-a2 --zone=us-central1-a

# # SSH into instance
# gcloud compute ssh toxtransformer-test-a2 --zone=us-central1-a

# # List instances
# gcloud compute instances list --project=$PROJECT_ID

