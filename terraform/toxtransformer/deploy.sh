#!/bin/bash
# Deploy toxtransformer to GCP
# Usage: ./deploy.sh [apply|plan|destroy]

set -e

cd "$(dirname "$0")"

# Get git commit from repo root
GIT_COMMIT=$(git -C ../.. rev-parse HEAD)
IMAGE_TAG="${IMAGE_TAG:-$(echo $GIT_COMMIT | cut -c1-8)}"

echo "=== Toxtransformer Deployment ==="
echo "Git commit: $GIT_COMMIT"
echo "Image tag:  $IMAGE_TAG"
echo ""

# Initialize terraform if needed
if [ ! -d ".terraform" ]; then
    echo "Initializing Terraform..."
    terraform init
fi

ACTION="${1:-plan}"

case "$ACTION" in
    plan)
        terraform plan \
            -var="git_commit=$GIT_COMMIT" \
            -var="image_tag=$IMAGE_TAG"
        ;;
    apply)
        terraform apply \
            -var="git_commit=$GIT_COMMIT" \
            -var="image_tag=$IMAGE_TAG" \
            -auto-approve
        echo ""
        echo "=== Deployment Complete ==="
        terraform output
        ;;
    destroy)
        terraform destroy \
            -var="git_commit=$GIT_COMMIT" \
            -var="image_tag=$IMAGE_TAG"
        ;;
    *)
        echo "Usage: $0 [plan|apply|destroy]"
        exit 1
        ;;
esac
