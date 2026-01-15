variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "toxindex"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "machine_type" {
  description = "GCE machine type"
  type        = string
  default     = "n1-standard-4"
}

variable "gpu_type" {
  description = "GPU type"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "github_repo" {
  description = "GitHub repository (org/repo format)"
  type        = string
  default     = "biobricks-ai/chemprop-transformer"
}

variable "git_commit" {
  description = "Git commit SHA for traceability"
  type        = string
}

variable "environment" {
  description = "Environment name (production, staging, dev)"
  type        = string
  default     = "production"
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}

variable "data_disk_size_gb" {
  description = "Size of the persistent data disk in GB (for prediction cache)"
  type        = number
  default     = 500  # Keep at 500GB since disk already exists and can't be shrunk
}

variable "data_bucket" {
  description = "GCS bucket containing brick/ data (cvae.sqlite, models, tokenizer)"
  type        = string
  default     = "toxtransformer-data"
}
