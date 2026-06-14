#!/usr/bin/env bash
# Launch ONE cost-optimized GPU spot instance to run the experiment, then self-terminate.
# Defaults reflect the verified working configuration (Seoul, g6e.xlarge L40S, the
# neuroxt-batch instance profile). Requires: code+data already staged to S3 (infra/stage.sh).
#
# Cost: g6e.xlarge spot ≈ $0.54/hr in ap-northeast-2; expected runtime ≤ ~2.5h ⇒ < ~$2.
# Set DRY_RUN=1 to validate permissions/params without launching.
set -euo pipefail

REGION="${REGION:-ap-northeast-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6e.xlarge}"
INSTANCE_PROFILE="${INSTANCE_PROFILE:-neuroxt-batch-instance-profile}"
RUN_TAG="${RUN_TAG:-ec2}"
S3_BUCKET="${S3_BUCKET:-neuroxt-personal}"
S3_PREFIX="${S3_PREFIX:-yhjeon/finance-ai-leakage}"
VOLUME_GB="${VOLUME_GB:-120}"
DRY_RUN="${DRY_RUN:-0}"

# Deep Learning Base GPU AMI (Ubuntu 22.04) — NVIDIA drivers preinstalled.
AMI_ID="${AMI_ID:-$(aws ec2 describe-images --region "$REGION" --owners amazon \
  --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*' \
            'Name=state,Values=available' \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)}"

# Default-VPC subnet + default security group (outbound only; SSM needs no inbound).
VPC=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text)
SUBNET_ID="${SUBNET_ID:-$(aws ec2 describe-subnets --region "$REGION" \
  --filters Name=vpc-id,Values="$VPC" --query 'Subnets[0].SubnetId' --output text)}"
SEC_GROUP="${SEC_GROUP:-$(aws ec2 describe-security-groups --region "$REGION" \
  --filters Name=vpc-id,Values="$VPC" Name=group-name,Values=default \
  --query 'SecurityGroups[0].GroupId' --output text)}"

echo "AMI=$AMI_ID type=$INSTANCE_TYPE region=$REGION subnet=$SUBNET_ID sg=$SEC_GROUP profile=$INSTANCE_PROFILE"

# Inject run env ahead of the bootstrap so it knows where to read/write.
USERDATA=$(mktemp)
{
  echo '#!/usr/bin/env bash'
  echo "export S3_BUCKET=$S3_BUCKET S3_PREFIX=$S3_PREFIX RUN_TAG=$RUN_TAG"
  cat "$(dirname "$0")/bootstrap.sh"
} > "$USERDATA"

DRY=()
[ "$DRY_RUN" = "1" ] && DRY=(--dry-run)

aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --instance-market-options 'MarketType=spot' \
  --iam-instance-profile "Name=$INSTANCE_PROFILE" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$VOLUME_GB,VolumeType=gp3,DeleteOnTermination=true}" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SEC_GROUP" \
  --user-data "file://$USERDATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=leakage-$RUN_TAG},{Key=project,Value=finance-ai-leakage}]" \
  "${DRY[@]}" \
  --query 'Instances[0].InstanceId' --output text

rm -f "$USERDATA"
echo "Watch: aws s3 cp s3://$S3_BUCKET/$S3_PREFIX/$RUN_TAG/bootstrap.log - | tail"
