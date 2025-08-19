#!/bin/bash

# Source configuration
source gcp-config.sh

# Create persistent disk
create_disk() {
    echo "📀 Creating ${DATA_DISK_SIZE} SSD persistent disk..."
    gcloud compute disks create $DISK_NAME \
      --zone=$ZONE \
      --size=$DATA_DISK_SIZE \
      --type=pd-ssd \
      --project=$PROJECT_ID
}

# Create new spot instance
create_instance() {
    echo "🚀 Creating new spot instance..."
    
    gcloud compute instances create $INSTANCE_NAME \
      --zone=$ZONE \
      --machine-type=$MACHINE_TYPE \
      --image=$IMAGE \
      --image-project=$IMAGE_PROJECT \
      --maintenance-policy=TERMINATE \
      --provisioning-model=SPOT \
      --boot-disk-size=$BOOT_DISK_SIZE \
      --disk=name=$DISK_NAME,device-name=data-disk,mode=rw,boot=no \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --tags=http-server \
      --metadata-from-file=startup-script=startup-script.sh
}

# Restart existing spot instance (most common use case)
restart_instance() {
    echo "🔄 Restarting spot instance..."
    
    # Delete instance if it exists (keeping disks)
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID &>/dev/null; then
        echo "⚠️  Deleting existing instance..."
        delete_instance
        sleep 5
    fi
    
    # Create new instance
    create_instance
    
    echo "⏳ Waiting for instance to be ready..."
    
    # Wait for instance to be running
    while true; do
        STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --format='get(status)' 2>/dev/null)
        if [ "$STATUS" = "RUNNING" ]; then
            break
        fi
        echo "   Status: $STATUS, waiting..."
        sleep 10
    done
    
    # Show connection info
    show_connection_info
    
    # Update SSH config if it exists
    update_ssh_config
}

# Connect to instance
connect() {
    echo "🔗 Connecting to instance..."
    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
}

# Show connection information
show_connection_info() {
    IP=$(gcloud compute instances describe $INSTANCE_NAME \
      --zone=$ZONE \
      --project=$PROJECT_ID \
      --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null)
    
    if [ ! -z "$IP" ]; then
        echo "✅ Instance ready!"
        echo "🌐 External IP: $IP"
        echo "🔗 Connect: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
        echo "📁 Rsync: rsync -avz --info=progress2 -e \"ssh -i ~/.ssh/google_compute_engine\" ./ insilica@$IP:/data/"
    else
        echo "❌ Instance not found or not running"
    fi
}

# Upload files to instance
upload_files() {
    IP=$(gcloud compute instances describe $INSTANCE_NAME \
      --zone=$ZONE \
      --project=$PROJECT_ID \
      --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null)
    
    if [ -z "$IP" ]; then
        echo "❌ Instance not running"
        return 1
    fi
    
    echo "📤 Uploading files to instance..."
    rsync -avz --info=progress2 -e "ssh -i ~/.ssh/google_compute_engine" ./ insilica@$IP:/data/
}

# Delete instance only (keep disk)
delete_instance() {
    echo "💥 Deleting instance (keeping persistent disk)..."
    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --keep-disks=all
    gcloud compute config-ssh --remove
}

# Delete everything (instance + disk)
delete_all() {
    echo "💥 Deleting instance and persistent disk..."
    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --quiet 2>/dev/null || true
    gcloud compute disks delete $DISK_NAME --zone=$ZONE --project=$PROJECT_ID --quiet
    gcloud compute config-ssh --remove
}

# Show status
status() {
    echo "📊 Current Status:"
    echo ""
    echo "🖥️  Instances:"
    gcloud compute instances list --filter="name:$INSTANCE_NAME"
    echo ""
    echo "💾 Disks:"
    gcloud compute disks list --filter="name:$DISK_NAME OR name:$INSTANCE_NAME"
}

# Configure local SSH for easier access
configure_ssh() {
    echo "🔧 Configuring local SSH using gcloud..."
    
    # Use gcloud to automatically configure SSH
    gcloud compute config-ssh --project=$PROJECT_ID
    
    # Also add a VS Code friendly entry
    configure_vscode_ssh
    
    echo "✅ SSH configured!"
    echo ""
    echo "Now you can use:"
    echo "  ssh $INSTANCE_NAME.$ZONE.$PROJECT_ID    # Connect to instance"
    echo "  ssh toxtransformer-vscode               # VS Code friendly alias"
    echo "  scp file.txt $INSTANCE_NAME.$ZONE.$PROJECT_ID:/data/"
    echo "  rsync -avz ./ $INSTANCE_NAME.$ZONE.$PROJECT_ID:/data/"
    echo ""
    echo "💡 gcloud automatically manages SSH keys and known hosts"
}

# Configure VS Code specific SSH entry
configure_vscode_ssh() {
    echo "🎨 Adding VS Code friendly SSH config..."
    
    # Get current instance IP
    IP=$(gcloud compute instances describe $INSTANCE_NAME \
      --zone=$ZONE \
      --project=$PROJECT_ID \
      --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null)
    
    if [ -z "$IP" ]; then
        echo "❌ Instance not running - start instance first"
        return 1
    fi
    
    # SSH config file
    SSH_CONFIG="$HOME/.ssh/config"
    HOST_ENTRY="toxtransformer-vscode"
    
    # Remove existing VS Code entry if it exists
    if grep -q "Host $HOST_ENTRY" "$SSH_CONFIG" 2>/dev/null; then
        # Remove the existing entry
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "/^Host $HOST_ENTRY$/,/^$/d" "$SSH_CONFIG"
        else
            # Linux
            sed -i "/^Host $HOST_ENTRY$/,/^$/d" "$SSH_CONFIG"
        fi
    fi
    
    # Get the exact paths that gcloud uses
    IDENTITY_FILE="$HOME/.ssh/google_compute_engine"
    KNOWN_HOSTS="$HOME/.ssh/google_compute_known_hosts"
    
    # Add VS Code optimized SSH config entry (matching gcloud format)
    cat >> "$SSH_CONFIG" << EOF

# ToxTransformer VS Code Remote (auto-generated)
Host $HOST_ENTRY
    HostName $IP
    User insilica
    IdentityFile $IDENTITY_FILE
    UserKnownHostsFile=$KNOWN_HOSTS
    IdentitiesOnly=yes
    CheckHostIP=no
    StrictHostKeyChecking=no
    ServerAliveInterval 60
    ServerAliveCountMax 3

EOF
}

# Update SSH config when instance IP changes (called by restart_instance)
update_ssh_config() {
    echo "🔄 Updating SSH config..."
    gcloud compute config-ssh --project=$PROJECT_ID
    configure_vscode_ssh
}

# Fix PATH issues on existing instance
fix_path() {
    echo "🔧 Fixing PATH issues on remote instance..."
    
    IP=$(gcloud compute instances describe $INSTANCE_NAME \
      --zone=$ZONE \
      --project=$PROJECT_ID \
      --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null)
    
    if [ -z "$IP" ]; then
        echo "❌ Instance not running"
        return 1
    fi
    
    # Copy the startup script to the instance and run the fix
    echo "📤 Uploading fix script..."
    scp -i ~/.ssh/google_compute_engine startup-script.sh tom@$IP:/tmp/
    
    echo "🔧 Running PATH fix on remote instance..."
    ssh -i ~/.ssh/google_compute_engine tom@$IP "sudo bash /tmp/startup-script.sh fix"
    
    echo "✅ PATH fix complete!"
    echo "💡 You may need to logout and login again on the remote instance"
}

# Help function
help() {
    echo "🚀 GCP Instance Management"
    echo ""
    echo "Available commands:"
    echo "  restart_instance  - Restart the spot instance (most common)"
    echo "  configure_ssh    - Set up SSH config for easy access"
    echo "  fix_ssh          - Fix SSH permissions and regenerate config"
    echo "  fix_path         - Fix PATH issues on existing instance"
    echo "  test_vscode_ssh  - Test and troubleshoot VS Code SSH connection"
    echo "  create_disk      - Create the persistent disk"
    echo "  create_instance  - Create new instance"
    echo "  connect          - SSH to instance"
    echo "  upload_files     - Upload current directory to instance"
    echo "  status          - Show current status"
    echo "  delete_instance - Delete instance (keep disk)"
    echo "  delete_all      - Delete instance and disk"
    echo "  help            - Show this help"
}

# If script is run directly, show help
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    help
fi