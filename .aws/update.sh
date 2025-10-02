# Rebuild the docker image and push it to ECR
# Destroy the service 
# Destroy the instances in the capacity provider
# Create a new service with the updated image
export AWS_REGION="us-east-1"
export AWS_PROFILE=chemprop-transformer
aws sso login --profile $AWS_PROFILE
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)

# Load secrets =================================================================
echo "Reading chemprop-transformer secret..." >&2
SECRET_NAME="github.com/biobricks-ai/chemprop-transformer"
SECRET=$(aws secretsmanager get-secret-value --secret-id $SECRET_NAME --query 'SecretString' --output text)

# Variables (update these for your AWS setup)
export ECR_REPO_NAME=$(echo $SECRET | jq -r '.ECR_REPO_NAME')
export CLUSTER_NAME=$(echo $SECRET | jq -r '.CLUSTER_NAME')
export SERVICE_NAME=$(echo $SECRET | jq -r '.SERVICE_NAME')
export SUBNET_IDS=$(echo $SECRET | jq -r '.SUBNET_IDS')
export SECURITY_GROUP_ID=$(echo $SECRET | jq -r '.SECURITY_GROUP_ID')
export TASK_DEFINITION_FILE=$(echo $SECRET | jq -r '.TASK_DEFINITION_FILE')
export DOCKER_IMAGE_TAG=$(echo $SECRET | jq -r '.DOCKER_IMAGE_TAG')
export ALB_NAME=$(echo $SECRET | jq -r '.ALB_NAME')
export TARGET_GROUP_NAME=$(echo $SECRET | jq -r '.TARGET_GROUP_NAME')
export VPC_ID=$(echo $SECRET | jq -r '.VPC_ID')
export SSO_START_URL=$(echo $SECRET | jq -r '.SSO_START_URL')  # Your SSO Start URL
export SSO_REGION=$(echo $SECRET | jq -r '.SSO_REGION')        # Your SSO region
export TEMP_PROFILE=$(echo $SECRET | jq -r '.TEMP_PROFILE')
export CAPACITY_PROVIDER_NAME=$(echo $SECRET | jq -r '.CAPACITY_PROVIDER_NAME')
export INSTANCE_TYPE=$(echo $SECRET | jq -r '.INSTANCE_TYPE')  # GPU instance type for ECS
export AMI_ID=$(echo $SECRET | jq -r '.AMI_ID')  # Amazon ECS-optimized AMI with GPU support
export KEY_NAME=$(echo $SECRET | jq -r '.KEY_NAME')  # SSH key pair name
export IAM_INSTANCE_PROFILE=$(echo $SECRET | jq -r '.IAM_INSTANCE_PROFILE')  # IAM role for ECS instances
export AUTO_SCALING_GROUP_NAME=$(echo $SECRET | jq -r '.AUTO_SCALING_GROUP_NAME')
export LAUNCH_TEMPLATE_NAME=$(echo $SECRET | jq -r '.LAUNCH_TEMPLATE_NAME')

# Build and push Docker image to ECR ===========================================
# depends on: [ ]
echo "Building Docker image..."
docker build -t $ECR_REPO_NAME .

echo "Tagging Docker image for ECR..."
docker tag "$ECR_REPO_NAME:$DOCKER_IMAGE_TAG" "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$DOCKER_IMAGE_TAG"

echo "Pushing Docker image to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
aws ecr describe-repositories --repository-names $ECR_REPO_NAME >/dev/null 2>&1 || aws ecr create-repository --repository-name $ECR_REPO_NAME
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$DOCKER_IMAGE_TAG

# Destroy the service ==========================================================
# it seems like this probably is not needed.
aws ecs delete-service --cluster $CLUSTER_NAME --service chemprop-transformer-service-4 --force

# Perform an autoscaling instances refresh to replace instances =============================
# Start an instance refresh to replace all instances in the ASG
echo "Starting Auto Scaling group instance refresh..."
# aws autoscaling start-instance-refresh \
#     --auto-scaling-group-name "$AUTO_SCALING_GROUP_NAME" \
#     --preferences "MinHealthyPercentage=0,InstanceWarmup=60" \
#     --strategy "Rolling"

# # Wait for the refresh to complete
# echo "Waiting for instance refresh to complete..."
# while true; do
#     REFRESH_STATUS=$(aws autoscaling describe-instance-refreshes \
#         --auto-scaling-group-name "$AUTO_SCALING_GROUP_NAME" \
#         --query 'InstanceRefreshes[0].Status' \
#         --output text)
    
#     echo "Instance refresh status: $REFRESH_STATUS"
    
#     if [ "$REFRESH_STATUS" = "Successful" ]; then
#         echo "Instance refresh completed successfully"
#         break
#     elif [ "$REFRESH_STATUS" = "Failed" ] || [ "$REFRESH_STATUS" = "Cancelled" ]; then
#         echo "Instance refresh failed or was cancelled"
#         exit 1
#     fi
    
#     sleep 30
# done
INSTANCE_IDS=($(aws autoscaling describe-auto-scaling-groups \
  --auto-scaling-group-names "$AUTO_SCALING_GROUP_NAME" \
  --query "AutoScalingGroups[0].Instances[*].InstanceId" \
  --output text))

# get capacity provider instances
for INSTANCE_ID in "${INSTANCE_IDS[@]}"; do
  echo "Terminating instance $INSTANCE_ID..."
  aws ec2 terminate-instances --instance-ids $INSTANCE_ID
done

# new instances will now be created with the updated image

# Create a new service with the updated image ================================
TG_ARN=$(aws elbv2 describe-target-groups --names $TARGET_GROUP_NAME --query 'TargetGroups[0].TargetGroupArn' --output text)
TASK=$(jq -r '.family' .aws/secrets/ecs_task_definition.json)
TASK_ARN=$(aws ecs describe-task-definition --task-definition $TASK --query 'taskDefinition.taskDefinitionArn' --output text)
echo $TASK_ARN

aws ecs create-service \
    --cluster $CLUSTER_NAME \
    --service-name chemprop-transformer-service-5 \
    --task-definition $TASK_ARN \
    --desired-count 2 \
    --health-check-grace-period-seconds 360 \
    --capacity-provider-strategy "capacityProvider=$CAPACITY_PROVIDER_NAME,weight=1" \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SECURITY_GROUP_ID]}" \
    --load-balancers "targetGroupArn=$TG_ARN,containerName=chemprop-transformer-container,containerPort=6515" >&2
