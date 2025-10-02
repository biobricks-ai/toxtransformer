#!/bin/bash

# ToxTransformer Flask CVAE App Deployment to Google Cloud Run
# This script builds and deploys the containerized Flask app to GCP

set -e  # Exit on any error

# Source configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../gcp-config.sh"

# Cloud Run specific configuration
SERVICE_NAME="toxtransformer-cvae"
IMAGE_NAME="toxtransformer-cvae"
ARTIFACT_REGISTRY_REPO="toxtransformer-images"
IMAGE_TAG="latest"
PORT=6515
MEMORY="4Gi"
CPU="2"
MAX_INSTANCES=10
MIN_INSTANCES=0
TIMEOUT=480

# Derived variables
FULL_IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if gcloud is authenticated
check_auth() {
    log_info "Checking gcloud authentication..."
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Not authenticated with gcloud. Please run: gcloud auth login"
        exit 1
    fi
    log_success "gcloud authentication verified"
}

# Function to set the correct project
set_project() {
    log_info "Setting project to ${PROJECT_ID}..."
    gcloud config set project ${PROJECT_ID}
    log_success "Project set to ${PROJECT_ID}"
}

# Function to enable required APIs
enable_apis() {
    log_info "Enabling required Google Cloud APIs..."
    
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        artifactregistry.googleapis.com \
        --project=${PROJECT_ID}
    
    log_success "Required APIs enabled"
}

# Function to create Artifact Registry repository
create_artifact_registry() {
    log_info "Creating Artifact Registry repository..."
    
    # Check if repository already exists
    if gcloud artifacts repositories describe ${ARTIFACT_REGISTRY_REPO} \
        --location=${REGION} \
        --project=${PROJECT_ID} &>/dev/null; then
        log_warning "Artifact Registry repository ${ARTIFACT_REGISTRY_REPO} already exists"
    else
        gcloud artifacts repositories create ${ARTIFACT_REGISTRY_REPO} \
            --repository-format=docker \
            --location=${REGION} \
            --description="ToxTransformer Docker images" \
            --project=${PROJECT_ID}
        log_success "Artifact Registry repository created"
    fi
}

# Function to configure Docker for Artifact Registry
configure_docker() {
    log_info "Configuring Docker for Artifact Registry..."
    gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
    log_success "Docker configured for Artifact Registry"
}

# Function to build and push Docker image
build_and_push() {
    log_info "Building Docker image..."
    
    # Change to project root directory
    cd "${SCRIPT_DIR}/../../"
    
    # Build the image
    docker build \
        --platform linux/amd64 \
        -t ${IMAGE_NAME}:${IMAGE_TAG} \
        -t ${FULL_IMAGE_NAME} \
        -f Dockerfile .
    
    log_success "Docker image built successfully"
    
    log_info "Pushing image to Artifact Registry..."
    docker push ${FULL_IMAGE_NAME}
    log_success "Image pushed to Artifact Registry"
}

# Function to deploy to Cloud Run
deploy_cloud_run() {
    log_info "Deploying to Cloud Run..."
    
    gcloud run deploy ${SERVICE_NAME} \
        --image=${FULL_IMAGE_NAME} \
        --platform=managed \
        --region=${REGION} \
        --allow-unauthenticated \
        --port=${PORT} \
        --memory=${MEMORY} \
        --cpu=${CPU} \
        --max-instances=${MAX_INSTANCES} \
        --min-instances=${MIN_INSTANCES} \
        --timeout=${TIMEOUT} \
        --set-env-vars="PORT=${PORT}" \
        --set-env-vars="FLASK_APP=flask_cvae.app" \
        --project=${PROJECT_ID}
    
    log_success "Application deployed to Cloud Run"
}

# Function to get service URL
get_service_url() {
    log_info "Getting service URL..."
    
    SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
        --region=${REGION} \
        --project=${PROJECT_ID} \
        --format='value(status.url)')
    
    if [ ! -z "$SERVICE_URL" ]; then
        log_success "Service deployed successfully!"
        echo ""
        echo "🌐 Service URL: ${SERVICE_URL}"
        echo ""
        echo "📋 Test endpoints:"
        echo "   Health check: ${SERVICE_URL}/health"
        echo "   Predict all:  ${SERVICE_URL}/predict_all?inchi=InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
        echo "   Predict:      ${SERVICE_URL}/predict?property_token=5042&inchi=InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
        echo ""
        echo "🔧 Manage service:"
        echo "   View logs:    gcloud run services logs read ${SERVICE_NAME} --region=${REGION}"
        echo "   Update:       Re-run this script"
        echo "   Delete:       gcloud run services delete ${SERVICE_NAME} --region=${REGION}"
        echo ""
    else
        log_error "Could not retrieve service URL"
        exit 1
    fi
}

# Function to test deployment
test_deployment() {
    log_info "Testing deployment..."
    
    if [ ! -z "$SERVICE_URL" ]; then
        # Test health endpoint
        log_info "Testing health endpoint..."
        if curl -s "${SERVICE_URL}/health" | grep -q "ok"; then
            log_success "Health check passed"
        else
            log_warning "Health check failed - service might still be starting up"
        fi
        
        echo ""
        log_info "You can test the prediction endpoints manually using the URLs shown above"
    fi
}

# Function to show help
show_help() {
    echo "ToxTransformer Cloud Run Deployment Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --build-only    Build and push image only (no deployment)"
    echo "  --deploy-only   Deploy existing image (no build)"
    echo "  --cleanup       Delete the Cloud Run service"
    echo "  --status        Show current deployment status"
    echo "  --logs          Show service logs"
    echo "  --help          Show this help message"
    echo ""
    echo "Default: Full build and deployment"
}

# Function to cleanup deployment
cleanup() {
    log_info "Cleaning up Cloud Run service..."
    
    gcloud run services delete ${SERVICE_NAME} \
        --region=${REGION} \
        --project=${PROJECT_ID} \
        --quiet
    
    log_success "Cloud Run service deleted"
    
    read -p "Do you want to delete the Docker image from Artifact Registry? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gcloud artifacts docker images delete ${FULL_IMAGE_NAME} \
            --project=${PROJECT_ID} \
            --quiet
        log_success "Docker image deleted from Artifact Registry"
    fi
}

# Function to show status
show_status() {
    log_info "Checking deployment status..."
    
    echo "📊 Cloud Run Services:"
    gcloud run services list \
        --filter="metadata.name:${SERVICE_NAME}" \
        --region=${REGION} \
        --project=${PROJECT_ID}
    
    echo ""
    echo "🐳 Docker Images:"
    gcloud artifacts docker images list \
        --repository=${ARTIFACT_REGISTRY_REPO} \
        --location=${REGION} \
        --project=${PROJECT_ID} \
        --filter="name:${IMAGE_NAME}" || log_warning "No images found or repository doesn't exist"
}

# Function to show logs
show_logs() {
    log_info "Showing service logs..."
    gcloud run services logs read ${SERVICE_NAME} \
        --region=${REGION} \
        --project=${PROJECT_ID} \
        --limit=50
}

# Main deployment function
main_deploy() {
    log_info "Starting ToxTransformer deployment to Cloud Run..."
    echo ""
    
    check_auth
    set_project
    enable_apis
    create_artifact_registry
    configure_docker
    
    if [ "$1" != "--deploy-only" ]; then
        build_and_push
    fi
    
    if [ "$1" != "--build-only" ]; then
        deploy_cloud_run
        get_service_url
        test_deployment
    fi
    
    log_success "Deployment completed successfully!"
}

# Parse command line arguments
case "${1:-}" in
    --build-only)
        log_info "Building and pushing image only..."
        check_auth
        set_project
        enable_apis
        create_artifact_registry
        configure_docker
        build_and_push
        ;;
    --deploy-only)
        log_info "Deploying existing image..."
        check_auth
        set_project
        deploy_cloud_run
        get_service_url
        test_deployment
        ;;
    --cleanup)
        cleanup
        ;;
    --status)
        show_status
        ;;
    --logs)
        show_logs
        ;;
    --help)
        show_help
        ;;
    "")
        main_deploy
        ;;
    *)
        log_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
