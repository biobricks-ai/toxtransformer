#!/bin/bash

# Mount persistent data disk
mount_data_disk() {
    DEVICE="/dev/disk/by-id/google-data-disk"
    MOUNT_POINT="/data"
    
    echo "💾 Mounting data disk..."
    
    # Check if disk is already formatted
    if ! blkid $DEVICE; then
        echo "Formatting disk..."
        mkfs.ext4 -F $DEVICE
    fi
    
    # Create mount point and mount
    mkdir -p $MOUNT_POINT
    mount $DEVICE $MOUNT_POINT
    
    # Add to fstab for automatic mounting (if not already there)
    if ! grep -q "$DEVICE" /etc/fstab; then
        echo "$DEVICE $MOUNT_POINT ext4 defaults 0 2" >> /etc/fstab
    fi
    
    # Set permissions
    chmod 755 $MOUNT_POINT
    chown $(logname):$(logname) $MOUNT_POINT 2>/dev/null || true
    chown -R :google-sudoers $MOUNT_POINT
    chmod -R 775 $MOUNT_POINT
}

# Setup RAID 0 with local NVMe SSDs
setup_raid() {
    echo "⚡ Setting up RAID 0 array..."
    
    apt-get update && apt-get install -y mdadm
    
    # Get NVMe disks (excluding boot disks)
    DISKS=$(ls /dev/nvme*n1 | grep -v "nvme0n1" | grep -v "nvme0n2")
    
    if [ -z "$DISKS" ]; then
        echo "No additional NVMe disks found for RAID"
        return
    fi
    
    # Create RAID array
    mdadm --create --verbose /dev/md0 --level=0 --raid-devices=$(echo $DISKS | wc -w) $DISKS
    
    # Wait for RAID to initialize
    sleep 10
    
    # Format and mount RAID
    mkfs.ext4 -F /dev/md0
    mkdir -p /mnt/raid0
    mount /dev/md0 /mnt/raid0
    
    # Add to fstab (if not already there)
    if ! grep -q "/dev/md0" /etc/fstab; then
        echo "/dev/md0 /mnt/raid0 ext4 defaults,nofail,discard 0 0" >> /etc/fstab
    fi
    
    # Set permissions
    chmod 755 /mnt/raid0
    chown $(logname):$(logname) /mnt/raid0 2>/dev/null || true
    chown -R :google-sudoers /mnt/raid0
    chmod -R 775 /mnt/raid0
    
    echo "✅ RAID 0 array ready at /mnt/raid0"
}

# Install required software
install_software() {
    echo "📦 Installing software..."
    
    apt-get update
    
    # Install Java
    apt-get install -y openjdk-17-jdk
    JAVA_HOME_PATH=$(readlink -f /usr/bin/java | sed "s:bin/java::")
    export JAVA_HOME=$JAVA_HOME_PATH
    export PATH=$PATH:$JAVA_HOME/bin
    
    # Install pipx
    apt-get install -y pipx
    
    echo "✅ Software installation complete"
}

# Main startup sequence
main() {
    echo "🚀 Starting instance setup..."
    mount_data_disk
    setup_raid
    install_software
    echo "🎉 Instance setup complete!"
}

main