# GCP Instance Management

Clean, modular scripts for managing your GCP spot instances.

## Files

- `gcp-config.sh` - Configuration variables
- `startup-script.sh` - Instance startup script (disk mounting, RAID, software)
- `gcp-functions.sh` - Main management functions
- `README.md` - This file

## Quick Start

```bash
# Source the functions
source gcp-functions.sh

# Restart your terminated spot instance (most common)
restart_instance

# Configure SSH for easy access
configure_ssh

# Now you can use simple commands:
ssh toxtransformer              # Instead of long gcloud ssh command
scp file.txt toxtransformer:/data/
rsync -avz ./ toxtransformer:/data/
```

## Common Workflows

### First Time Setup
```bash
source gcp-functions.sh
create_disk          # Only needed once
create_instance      # Creates instance with disk attached
```

### Daily Usage (Spot Instance Management)
```bash
source gcp-functions.sh
restart_instance     # Restart terminated spot instance
configure_ssh       # Set up easy SSH access
ssh toxtransformer  # Connect using simple hostname
```

### Cleanup
```bash
source gcp-functions.sh
delete_instance     # Keep disk, delete instance
delete_all         # Delete everything
```

## Functions

| Function | Description |
|----------|-------------|
| `restart_instance` | **Most common** - Restart terminated spot instance |
| `configure_ssh` | Set up SSH config for easy access (`ssh toxtransformer`) |
| `create_disk` | Create the persistent disk (one-time setup) |
| `create_instance` | Create new instance |
| `connect` | SSH to the instance |
| `upload_files` | Rsync current directory to /data |
| `status` | Show instance and disk status |
| `delete_instance` | Delete instance, keep disk |
| `delete_all` | Delete instance and disk |

## Key Improvements

- ✅ **Modular**: Separate config, startup script, and functions
- ✅ **No Redundancy**: Startup script is external file, not embedded
- ✅ **Reusable**: Source once, use multiple functions
- ✅ **Clean**: Each function has a single responsibility
- ✅ **Safe**: Proper error checking and status reporting