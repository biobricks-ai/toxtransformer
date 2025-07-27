#!/bin/bash

PROJECT_ID="toxtransformer"
REGION="us-east5"
ZONE="us-east5-a"
INSTANCE_NAME="toxtransformer-8h100"
DISK_NAME="toxtransformer-data-disk"

# list active projects
gcloud projects list --filter="lifecycleState:ACTIVE"

# zip the project files
zip -r toxtransformer.zip .

# list active compute instances
gcloud compute instances list 

# Create a 3TB SSD persistent disk (fastest option)
echo "📀 Creating 10TB SSD persistent disk..."
gcloud compute disks create $DISK_NAME \
  --zone=$ZONE \
  --size=10TB \
  --type=pd-ssd \
  --project=$PROJECT_ID

gcloud compute disks list 
# delete the disk
# gcloud compute disks delete $DISK_NAME --zone=$ZONE --project=$PROJECT_ID

# Launch an 8xH100 instance with the persistent disk attached
echo "🚀 Launching instance with persistent disk..."
gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --machine-type=a3-highgpu-8g \
  --image=pytorch-latest-gpu-v20250325-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --boot-disk-size=200GB \
  --disk=name=$DISK_NAME,device-name=data-disk,mode=rw,boot=no \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --tags=http-server \
  --metadata=startup-script='#!/bin/bash
    # Auto-mount the persistent disk on startup
    DEVICE="/dev/disk/by-id/google-data-disk"
    MOUNT_POINT="/data"
    
    # Check if disk is already formatted
    if ! blkid $DEVICE; then
        echo "Formatting disk..."
        mkfs.ext4 -F $DEVICE
    fi
    
    # Create mount point and mount
    mkdir -p $MOUNT_POINT
    mount $DEVICE $MOUNT_POINT
    
    # Add to fstab for automatic mounting
    echo "$DEVICE $MOUNT_POINT ext4 defaults 0 2" >> /etc/fstab
    
    # Set permissions
    chmod 755 $MOUNT_POINT
    chown $(logname):$(logname) $MOUNT_POINT 2>/dev/null || true

    # RAID 0 setup
    echo "Creating RAID 0 array from NVMe SSDs..."
    apt-get update && apt-get install -y mdadm

    DISKS=$(ls /dev/nvme*n1 | grep -v "nvme0n1" | grep -v "nvme0n2")
    mdadm --create --verbose /dev/md0 --level=0 --raid-devices=$(echo $DISKS | wc -w) $DISKS

    # Wait for RAID to initialize
    sleep 10

    # Format and mount RAID
    mkfs.ext4 -F /dev/md0
    mkdir -p /mnt/raid0
    mount /dev/md0 /mnt/raid0
    echo "/dev/md0 /mnt/raid0 ext4 defaults,nofail,discard 0 0" >> /etc/fstab
    chmod 755 /mnt/raid0
    chown $(logname):$(logname) /mnt/raid0 2>/dev/null || true

    # Optional: Copy contents from /data/toxtransformer to RAID
    if [ -d /data/toxtransformer ]; then
      echo "Copying data to RAID volume..."
      rsync -a --info=progress2 /data/toxtransformer/ /mnt/raid0/toxtransformer/
    fi

    chown -R :google-sudoers /mnt/raid0
    chmod -R 775 /mnt/raid0
    chown -R :google-sudoers /data
    chmod -R 775 /data/toxtransformer

    echo "✅ RAID 0 array ready at /mnt/raid0"
'

# SSH into the instance (NEED TO MANUALLY INSTALL nvidia-driver)
echo "🔗 Connecting to instance..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID

# Upload the zip file to the instance
IP=$(gcloud compute instances describe toxtransformer-8h100 \
  --zone=us-east5-a \
  --project=toxtransformer \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

# install rclone
gcloud compute ssh 
rsync -avz --info=progress2 -e "ssh -i ~/.ssh/google_compute_engine" ./ insilica@$IP:/data



# gcloud compute scp toxtransformer.zip $INSTANCE_NAME:/data --zone=$ZONE --project=$PROJECT_ID
# rsync -av --info=progress2 -e "ssh -i ~/.ssh/google_compute_engine" toxtransformer.zip insilica@$IP:/data/




# Function to delete just the instance (keeping the disk)
delete_instance() {
    echo "💥 Shutting down the instance (keeping persistent disk)..."
    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --keep-disks=all
}

# Function to delete both instance and disk
delete_all() {
    echo "💥 Shutting down instance and deleting persistent disk..."
    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
    gcloud compute disks delete $DISK_NAME --zone=$ZONE --project=$PROJECT_ID
}

# Uncomment the function you want to use:
# delete_instance  # Keeps the persistent disk for reuse
# delete_all      # Deletes everything