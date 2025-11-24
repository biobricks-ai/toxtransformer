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

# Create new ON-DEMAND instance (auto-handles GPU vs CPU-only)
create_instance_on_demand() {
    echo "🚀 Creating new ON-DEMAND instance..."

    # Fail fast if data disk isn't in this $ZONE
    if ! gcloud compute disks describe "$DISK_NAME" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
        echo "❌ Data disk '$DISK_NAME' not found in zone '$ZONE'."
        echo "   Run: copy_disk_to_zone $ZONE  &&  switch_to_zone $ZONE"
        return 1
    fi

    # Choose maintenance policy based on whether a GPU is requested
    local MAINTENANCE_POLICY="MIGRATE"
    local EXTRA_ARGS=()
    MAINTENANCE_POLICY="TERMINATE"
    EXTRA_ARGS+=(--accelerator="type=nvidia-h100-80gb,count=8")

    gcloud compute instances create "$INSTANCE_NAME" \
      --zone="$ZONE" \
      --machine-type="$MACHINE_TYPE" \
      --image="$IMAGE" \
      --image-project="$IMAGE_PROJECT" \
      --provisioning-model=STANDARD \
      --maintenance-policy="$MAINTENANCE_POLICY" \
      --boot-disk-size="$BOOT_DISK_SIZE" \
      --disk="name=$DISK_NAME,device-name=data-disk,mode=rw,boot=no" \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --tags=http-server \
      --metadata-from-file=startup-script=startup-script.sh \
      --project="$PROJECT_ID" \
      "${EXTRA_ARGS[@]}"
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

# --- helpers ---------------------------------------------------------------

# Lowercase + safe chars; only prefix if first char isn't a letter; max 63.
_sanitize_name() {
  local s="$1"
  s="$(echo "$s" | tr '[:upper:]' '[:lower:]')"     # lowercase
  s="$(echo "$s" | sed 's/[^a-z0-9-]/-/g')"         # non [a-z0-9-] -> -
  s="$(echo "$s" | sed 's/--*/-/g')"                # collapse --
  s="${s%-}"                                        # drop trailing -
  [[ "$s" =~ ^[a-z] ]] || s="a$s"                   # must start with letter
  if [ ${#s} -gt 63 ]; then
    s="${s:0:63}"
    s="${s%-}"
    [[ "$s" =~ [a-z0-9]$ ]] || s="${s:0:${#s}-1}0"
  fi
  echo "$s"
}

# Compute target disk name by stripping the current $ZONE suffix if present,
# then appending "-$target_zone"
_target_disk_name() {
  local src_disk="$1"
  local src_zone="$2"
  local target_zone="$3"
  local base="$src_disk"
  if [[ "$src_disk" == *"-${src_zone}" ]]; then
    base="${src_disk%-${src_zone}}"
  fi
  _sanitize_name "${base}-${target_zone}"
}

# Quick capability check for a zone. Returns 0 if allowed & exists; 1 otherwise.
_check_zone_writable() {
  local target_zone="$1"
  local probe_name
  probe_name="$(_sanitize_name "probe-disk-${target_zone}-$(date +%Y%m%d%H%M%S)")"

  # Zone must exist
  if ! gcloud compute zones describe "$target_zone" --project="$PROJECT_ID" &>/dev/null; then
    echo "❌ Zone '$target_zone' not found (typo or not available in project?)"
    return 1
  fi

  # Try creating a tiny scratch disk, then delete it.
  # If Org Policy blocks the location, this will fail with the same "Permission denied on 'locations/...'"
  if ! gcloud compute disks create "$probe_name" \
        --zone="$target_zone" \
        --size=1GB \
        --type=pd-standard \
        --project="$PROJECT_ID" \
        --quiet &>/dev/null; then
    echo "❌ Cannot create resources in '$target_zone'. Likely blocked by org policy (e.g., gcp.resourceLocations)."
    echo "   Ask your org admin to allow region of '$target_zone' or choose another zone."
    return 1
  fi
  # Cleanup
  gcloud compute disks delete "$probe_name" --zone="$target_zone" --project="$PROJECT_ID" --quiet &>/dev/null || true
  return 0
}

# --- fixed copy function ---------------------------------------------------

copy_disk_to_zone() {
  local target_zone="${1:-us-central1-a}"

  echo "💾 Copying persistent disk to zone: $target_zone"
  echo "   Source: $DISK_NAME in $ZONE"

  # Validate source disk
  if ! gcloud compute disks describe "$DISK_NAME" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
    echo "❌ Source disk '$DISK_NAME' not found in zone '$ZONE'"
    return 1
  fi

  # Preflight: can we create anything in the target zone?
  if ! _check_zone_writable "$target_zone"; then
    return 1
  fi

  # Build safe target disk name
  local target_disk_name
  target_disk_name="$(_target_disk_name "$DISK_NAME" "$ZONE" "$target_zone")"
  echo "   Target: $target_disk_name in $target_zone"

  # Check if target disk already exists
  if gcloud compute disks describe "$target_disk_name" --zone="$target_zone" --project="$PROJECT_ID" &>/dev/null; then
    echo "⚠️  Target disk already exists."
    read -p "Delete and recreate '$target_disk_name'? (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
      echo "🗑️  Deleting existing target disk..."
      gcloud compute disks delete "$target_disk_name" --zone="$target_zone" --project="$PROJECT_ID" --quiet || return 1
    else
      echo "❌ Aborted"
      return 1
    fi
  fi

  # Create a valid, short snapshot name
  local ts hash6 base_for_snap snapshot_name
  ts="$(date +%Y%m%d%H%M%S)"
  base_for_snap="${DISK_NAME%-${ZONE}}"
  [[ "$base_for_snap" == "$DISK_NAME" ]] && base_for_snap="$DISK_NAME"
  hash6=$(printf "%s" "${PROJECT_ID}/${ZONE}/${DISK_NAME}/${ts}" | md5sum | awk '{print $1}' | cut -c1-6 2>/dev/null)
  snapshot_name="$(_sanitize_name "${base_for_snap}-snap-${ts}-${hash6}")"

  echo "📸 Creating snapshot: $snapshot_name"
  if ! gcloud compute disks snapshot "$DISK_NAME" \
        --zone="$ZONE" \
        --snapshot-names="$snapshot_name" \
        --project="$PROJECT_ID"; then
    echo "❌ Failed to create snapshot"
    return 1
  fi

  echo "💾 Creating disk in target zone from snapshot..."
  if ! gcloud compute disks create "$target_disk_name" \
        --zone="$target_zone" \
        --source-snapshot="$snapshot_name" \
        --type=pd-ssd \
        --project="$PROJECT_ID"; then
    echo "❌ Failed to create disk in target zone"
    echo "🧹 Cleaning up snapshot..."
    gcloud compute snapshots delete "$snapshot_name" --project="$PROJECT_ID" --quiet || true
    return 1
  fi

  echo "✅ Disk copied successfully!"
  echo ""
  echo "📋 Summary:"
  echo "   Original disk: $DISK_NAME in $ZONE"
  echo "   New disk:      $target_disk_name in $target_zone"
  echo "   Snapshot:      $snapshot_name (can be deleted later)"
  echo ""
  echo "💡 Next steps:"
  echo "   switch_to_zone $target_zone"
}


# Switch configuration to use a different zone/disk
switch_to_zone() {
    local target_zone=${1:-"us-central1-a"}
    local target_disk_name="${DISK_NAME}-${target_zone}"
    
    echo "🔄 Switching configuration to zone: $target_zone"
    
    # Check if target disk exists
    if ! gcloud compute disks describe $target_disk_name --zone=$target_zone --project=$PROJECT_ID &>/dev/null; then
        echo "❌ Disk $target_disk_name not found in zone $target_zone"
        echo "💡 Run: copy_disk_to_zone $target_zone"
        return 1
    fi
    
    # Update environment variables for this session
    export ZONE=$target_zone
    export DISK_NAME=$target_disk_name
    
    echo "✅ Configuration updated for this session:"
    echo "   Zone: $ZONE"
    echo "   Disk: $DISK_NAME"
    echo ""
    echo "💡 To make this permanent, update gcp-config.sh:"
    echo "   ZONE=\"$target_zone\""
    echo "   DISK_NAME=\"$target_disk_name\""
}

# Create instance in us-central1 (convenience function)
restart_instance_central1() {
    echo "🚀 Starting instance in us-central1-a..."
    
    # Check if we need to copy the disk first
    local target_zone="us-central1-a"
    local target_disk_name="${DISK_NAME}-${target_zone}"
    
    if ! gcloud compute disks describe $target_disk_name --zone=$target_zone --project=$PROJECT_ID &>/dev/null; then
        echo "💾 Disk not found in $target_zone, copying..."
        copy_disk_to_zone $target_zone
        if [ $? -ne 0 ]; then
            return 1
        fi
    fi
    
    # Switch to the target zone
    switch_to_zone $target_zone
    
    # Restart instance with new configuration
    restart_instance
}

# Clean up old resources after successful migration
cleanup_old_zone() {
    local old_zone=${1:-"us-east5-a"}
    local old_disk_name=${2:-"toxtransformer-data-disk"}
    
    echo "🧹 Cleaning up resources in old zone: $old_zone"
    
    # List what will be deleted
    echo "📋 Resources to be deleted:"
    echo "   Disk: $old_disk_name in $old_zone"
    echo "   Instance: $INSTANCE_NAME in $old_zone (if exists)"
    echo ""
    
    read -p "Are you sure you want to delete these resources? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "❌ Cleanup cancelled"
        return 1
    fi
    
    # Delete instance if it exists
    if gcloud compute instances describe $INSTANCE_NAME --zone=$old_zone --project=$PROJECT_ID &>/dev/null; then
        echo "🗑️  Deleting instance in old zone..."
        gcloud compute instances delete $INSTANCE_NAME --zone=$old_zone --project=$PROJECT_ID --quiet
    fi
    
    # Delete old disk
    if gcloud compute disks describe $old_disk_name --zone=$old_zone --project=$PROJECT_ID &>/dev/null; then
        echo "🗑️  Deleting disk in old zone..."
        gcloud compute disks delete $old_disk_name --zone=$old_zone --project=$PROJECT_ID --quiet
    fi
    
    echo "✅ Cleanup completed!"
}

# Show status
status() {
    echo "📊 Current Status:"
    echo ""
    echo "🖥️  Instances:"
    gcloud compute instances list --project=$PROJECT_ID --filter="name~$INSTANCE_NAME"
    echo ""
    echo "💾 Disks:"
    gcloud compute disks list --project=$PROJECT_ID --filter="name~$DISK_NAME OR name~$INSTANCE_NAME"
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
    echo "  restart_instance         - Restart the spot instance (most common)"
    echo "  restart_instance_central1 - Start instance in us-central1-a (auto-copies disk)"
    echo "  copy_disk_to_zone [zone] - Copy persistent disk to another zone"
    echo "  switch_to_zone [zone]    - Switch config to use different zone"
    echo "  cleanup_old_zone [zone]  - Delete resources in old zone after migration"
    echo "  configure_ssh           - Set up SSH config for easy access"
    echo "  create_disk             - Create the persistent disk"
    echo "  create_instance         - Create new instance"
    echo "  connect                 - SSH to instance"
    echo "  upload_files            - Upload current directory to instance"
    echo "  status                  - Show current status"
    echo "  delete_instance         - Delete instance (keep disk)"
    echo "  delete_all              - Delete instance and disk"
    echo "  help                    - Show this help"
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