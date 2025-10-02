# LOGIN =======================================================================
export AWS_REGION="us-east-1"
export AWS_PROFILE=chemprop-transformer

# Check if profile exists and prompt for setup if it doesn't
if ! aws configure list-profiles 2>/dev/null | grep -q "^$AWS_PROFILE$"; then
    echo -e "\e[35mPlease create an AWS profile named 'chemprop-transformer'\e[0m"
    echo -e "Required permissions: ECS, ECR, S3, CloudWatch Logs"
    echo -e "Run: aws configure sso --profile chemprop-transformer\n"
    exit 1
fi

# Login and get account ID
aws sso login --profile $AWS_PROFILE
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
echo $AWS_ACCOUNT_ID

# PULL SECRETS ================================================================

# Define secrets mapping (secret_id -> local_path)
declare -A SECRETS
SECRETS[github.com/biobricks-ai/chemprop-transformer]=".aws/secrets/secrets.json"
SECRETS[chemprop-transformer/ecs-task-definition]=".aws/secrets/ecs_task_definition.json"
SECRETS[chemprop-transformer/deploy_aws.sh]=".aws/secrets/deploy_aws.sh"

aws secretsmanager get-secret-value --secret-id "github.com/biobricks-ai/chemprop-transformer"
# Pull all secrets
for secret_id in "${(@k)SECRETS}"; do
    echo "Pulling $secret_id..." >&2
    aws secretsmanager get-secret-value \
        --secret-id "$secret_id" \
        --query 'SecretString' \
        --output text > "${SECRETS[$secret_id]}"
done


# UPDATE SECRETS ==============================================================
create_or_update_secrets() {
    for secret_id in "${(@k)SECRETS}"; do
        echo "Updating $secret_id..." >&2
        if ! aws secretsmanager create-secret \
            --name "$secret_id" \
            --secret-string "$(cat "${SECRETS[$secret_id]}")" 2>/dev/null; then
            aws secretsmanager update-secret \
                --secret-id "$secret_id" \
                --secret-string "$(cat "${SECRETS[$secret_id]}")"
        fi
    done
}

# RUN THIS TO UPDATE ALL SECRETS IN AWS SECRETS MANAGER
create_or_update_secrets

