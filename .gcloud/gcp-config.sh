#!/bin/bash

# GCP Configuration
PROJECT_ID="toxtransformer"
REGION="us-east5"
ZONE="us-east5-a"
INSTANCE_NAME="toxtransformer-8h100"
DISK_NAME="toxtransformer-data-disk"

# Instance Configuration
MACHINE_TYPE="a3-highgpu-8g"
IMAGE="pytorch-latest-gpu-v20250325-ubuntu-2204-py310"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE="200GB"
DATA_DISK_SIZE="10TB"