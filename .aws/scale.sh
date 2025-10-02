# LOGIN ======================================================================
export AWS_REGION="us-east-1"
export AWS_PROFILE=chemprop-transformer
aws sso login --profile $AWS_PROFILE

# Load secrets
echo "Reading chemprop-transformer secret..." >&2
SECRET_NAME="github.com/biobricks-ai/chemprop-transformer"
SECRET=$(aws secretsmanager get-secret-value --secret-id $SECRET_NAME --query 'SecretString' --output text)

# Load variables from secret
export CLUSTER_NAME=$(echo $SECRET | jq -r '.CLUSTER_NAME')
export SERVICE_NAME=$(echo $SECRET | jq -r '.SERVICE_NAME')
export AUTO_SCALING_GROUP_NAME=$(echo $SECRET | jq -r '.AUTO_SCALING_GROUP_NAME')

echo "SCALING AWS resources..."
# Function to scale the ECS service and Auto Scaling Group to n instances
scale_service() {
    local n=$1
    echo "Scaling service to $n instances..."

    # Update Auto Scaling Group
    aws autoscaling update-auto-scaling-group \
        --auto-scaling-group-name "$AUTO_SCALING_GROUP_NAME" \
        --min-size $n \
        --max-size $n \
        --desired-capacity $n

    # Scale ECS service
    echo "Scaling ECS service to $n tasks..."
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --desired-count $n

    # Wait for service to stabilize
    echo "Waiting for service to stabilize..."
    aws ecs wait services-stable \
        --cluster $CLUSTER_NAME \
        --services $SERVICE_NAME

    # If scaling down to 0, terminate any remaining instances
    if [ "$n" -eq 0 ]; then
        echo "Terminating any remaining EC2 instances..."
        INSTANCE_IDS=$(aws autoscaling describe-auto-scaling-groups \
            --auto-scaling-group-names "$AUTO_SCALING_GROUP_NAME" \
            --query "AutoScalingGroups[0].Instances[*].InstanceId" \
            --output text)

        if [ ! -z "$INSTANCE_IDS" ]; then
            for INSTANCE_ID in $INSTANCE_IDS; do
                echo "Terminating instance $INSTANCE_ID..."
                aws ec2 terminate-instances --instance-ids $INSTANCE_ID
            done
        fi
        echo "Shutdown complete. Expensive resources have been terminated."
    else
        echo "Service scaled to $n instances."
    fi
}

# Call the function with the provided argument
scale_service 10