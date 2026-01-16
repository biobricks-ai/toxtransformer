terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

locals {
  # Common labels for all resources
  common_labels = {
    managed_by  = "terraform"
    github_repo = replace(var.github_repo, "/", "_")
    git_commit  = substr(var.git_commit, 0, 8)
    environment = var.environment
    service     = "toxtransformer"
  }

  # Use existing toxindex-backend registry in us region
  image_uri = "us-docker.pkg.dev/${var.project_id}/toxindex-backend/toxtransformer:${var.image_tag}"
}

# GCS bucket for brick/ data (cvae.sqlite, models, tokenizer) - read-only via GCS FUSE
resource "google_storage_bucket" "data" {
  name          = var.data_bucket
  location      = "US"
  force_destroy = false

  labels = local.common_labels

  uniform_bucket_level_access = true
}

# Persistent disk for prediction cache (read-write)
# This disk persists independently of the instance
resource "google_compute_disk" "data" {
  name = "toxtransformer-data"
  type = "pd-ssd"
  zone = var.zone
  size = var.data_disk_size_gb

  labels = local.common_labels

  # Prevent accidental deletion of data
  lifecycle {
    prevent_destroy = true
  }
}

# Static external IP
resource "google_compute_address" "toxtransformer" {
  name   = "toxtransformer-ip"
  region = var.region

  labels = local.common_labels
}

# Firewall rule for HTTP/HTTPS
resource "google_compute_firewall" "toxtransformer" {
  name    = "toxtransformer-allow-http"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "6515"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["toxtransformer"]
}

# Build and push Docker image
resource "null_resource" "docker_build" {
  triggers = {
    git_commit = var.git_commit
    dockerfile = filesha256("${path.module}/../../Dockerfile")
  }

  provisioner "local-exec" {
    working_dir = "${path.module}/../.."
    command     = <<-EOT
      set -e

      # Configure Docker for Artifact Registry
      gcloud auth configure-docker us-docker.pkg.dev --quiet

      # Build the image
      docker build -t ${local.image_uri} \
        --label "org.opencontainers.image.source=https://github.com/${var.github_repo}" \
        --label "org.opencontainers.image.revision=${var.git_commit}" \
        .

      # Push to Artifact Registry
      docker push ${local.image_uri}
    EOT
  }
}

# GCE instance with GPU
resource "google_compute_instance" "toxtransformer" {
  name         = "toxtransformer"
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["toxtransformer"]

  labels = local.common_labels

  boot_disk {
    initialize_params {
      image  = "deeplearning-platform-release/common-cu128-ubuntu-2204-nvidia-570-v20260113"
      size   = 100
      type   = "pd-ssd"
      labels = local.common_labels
    }
  }

  # Attach the persistent data disk
  attached_disk {
    source      = google_compute_disk.data.self_link
    device_name = "toxtransformer-data"
    mode        = "READ_WRITE"
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.toxtransformer.address
    }
  }

  guest_accelerator {
    type  = var.gpu_type
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
    preemptible         = true
    provisioning_model  = "SPOT"
  }

  metadata = {
    install-nvidia-driver = "True"
    github-repo           = var.github_repo
    git-commit            = var.git_commit
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    set -e
    exec > >(tee /var/log/startup-script.log) 2>&1

    echo "=== Startup script started at $(date) ==="

    # Mount the persistent disk for cache (read-write)
    DATA_DISK="/dev/disk/by-id/google-toxtransformer-data"
    CACHE_MOUNT="/mnt/cache"

    mkdir -p $CACHE_MOUNT

    # Check if disk is formatted
    if ! blkid $DATA_DISK; then
      echo "Formatting data disk..."
      mkfs.ext4 -F $DATA_DISK
    fi

    # Mount if not already mounted
    if ! mountpoint -q $CACHE_MOUNT; then
      echo "Mounting cache disk..."
      mount -o discard,defaults $DATA_DISK $CACHE_MOUNT
      echo "$DATA_DISK $CACHE_MOUNT ext4 discard,defaults,nofail 0 2" >> /etc/fstab
    fi

    echo "Cache disk mounted at $CACHE_MOUNT"
    df -h $CACHE_MOUNT

    # Install GCS FUSE for mounting brick/ data from GCS
    echo "Installing GCS FUSE..."
    export GCSFUSE_REPO=gcsfuse-$(lsb_release -cs)

    # Remove stale repos from old image that cause apt-get update to fail
    rm -f /etc/apt/sources.list.d/kubernetes.list 2>/dev/null || true
    sed -i '/backports/d' /etc/apt/sources.list 2>/dev/null || true

    # Add gcsfuse repo (remove existing key first to prevent dearmor failure)
    rm -f /usr/share/keyrings/gcsfuse.gpg 2>/dev/null || true
    echo "deb [signed-by=/usr/share/keyrings/gcsfuse.gpg] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --batch --dearmor -o /usr/share/keyrings/gcsfuse.gpg

    apt-get update --allow-releaseinfo-change || true
    apt-get install -y gcsfuse

    # Mount GCS bucket for brick/ (read-only)
    BRICK_MOUNT="/mnt/brick"
    mkdir -p $BRICK_MOUNT

    echo "Mounting GCS bucket ${var.data_bucket} at $BRICK_MOUNT..."
    gcsfuse --implicit-dirs -o ro,allow_other ${var.data_bucket} $BRICK_MOUNT
    echo "GCS bucket mounted"
    ls -la $BRICK_MOUNT

    # Wait for NVIDIA driver
    echo "Waiting for NVIDIA driver..."
    until nvidia-smi; do
      sleep 10
    done
    echo "NVIDIA driver ready"

    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
      echo "Installing Docker..."
      curl -fsSL https://get.docker.com | sh
    fi

    # Install NVIDIA Container Toolkit
    if ! dpkg -l | grep -q nvidia-container-toolkit; then
      echo "Installing NVIDIA Container Toolkit..."
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
      apt-get update
      apt-get install -y nvidia-container-toolkit
      nvidia-ctk runtime configure --runtime=docker
      systemctl restart docker
    fi

    # Configure Docker to use Artifact Registry
    echo "Configuring Docker for Artifact Registry..."
    gcloud auth configure-docker us-docker.pkg.dev --quiet

    # Pull and run the container
    CONTAINER_NAME="toxtransformer"
    IMAGE="${local.image_uri}"

    echo "Pulling image: $IMAGE"
    docker pull $IMAGE

    # Stop existing container if running
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true

    # Run the container with GCS FUSE for brick/ and persistent disk for cache/
    echo "Starting container..."
    docker run -d \
      --name $CONTAINER_NAME \
      --restart unless-stopped \
      --gpus all \
      -p 6515:6515 \
      --privileged \
      -v $BRICK_MOUNT:/app/brick:ro \
      -v $CACHE_MOUNT:/app/cache \
      -e GITHUB_REPO="${var.github_repo}" \
      -e GIT_COMMIT="${var.git_commit}" \
      $IMAGE

    echo "=== Startup script completed at $(date) ==="
  EOF

  service_account {
    scopes = ["cloud-platform"]
  }

  depends_on = [null_resource.docker_build]
}
