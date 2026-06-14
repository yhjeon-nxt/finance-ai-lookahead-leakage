#!/usr/bin/env bash
# Launch ONE cost-optimized GPU spot instance to run the experiment, then self-terminate.
# Reads bootstrap.sh as user-data. Requires: an IAM instance profile that can read/write the
# S3 prefix, and that code+data have been staged to S3 (see stage.sh).
#
# Cost note: g5.xlarge spot ≈ $0.40-0.60/hr; expected runtime ≤ ~1h ⇒ < ~$2 total.
# Override INSTANCE_TYPE=g4dn.xlarge for the cheapest option (slower T4 GPU).
set -euo pipefail

REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
INSTANCE_PROFILE="${INSTANCE_PROFILE:?set INSTANCE_PROFILE to an IAM role with S3 access}"
KEY_NAME="${KEY_NAME:-}"            # optional, for SSH debugging
SUBNET_ID="${SUBNET_ID:-}"         # optional
SEC_GROUP="${SEC_GROUP:-}"         # optional
RUN_TAG="${RUN_TAG:-ec2}"
S3_BUCKET="${S3_BUCKET:-neuroxt-personal}"
S3_PREFIX="${S3_PREFIX:-yhjeon/finance-ai-leakage}"

# Deep Learning Base AMI (Ubuntu 22.04) — has NVIDIA drivers preinstalled. Resolve latest.
AMI_ID="${AMI_ID:-$(aws ec2 describe-images --region "$REGION" --owners amazon \
  --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*' \
            'Name=state,Values=available' \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)}"

echo "AMI=$AMI_ID type=$INSTANCE_TYPE region=$REGION tag=$RUN_TAG"

# Inject env into user-data so bootstrap knows where to read/write.
USERDATA=$(mktemp)
{
  echo '#!/usr/bin/env bash'
  echo "export S3_BUCKET=$S3_BUCKET S3_PREFIX=$S3_PREFIX RUN_TAG=$RUN_TAG"
  cat "$(dirname "$0")/bootstrap.sh"
} > "$USERDATA"

EXTRA=()
[ -n "$KEY_NAME" ] && EXTRA+=(--key-name "$KEY_NAME")
[ -n "$SUBNET_ID" ] && EXTRA+=(--subnet-id "$SUBNET_ID")
[ -n "$SEC_GROUP" ] && EXTRA+=(--security-group-ids "$SEC_GROUP")

aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --instance-market-options 'MarketType=spot' \
  --iam-instance-profile "Name=$INSTANCE_PROFILE" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=60,VolumeType=gp3,DeleteOnTermination=true}' \
  --user-data "file://$USERDATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=leakage-$RUN_TAG},{Key=project,Value=finance-ai-leakage}]" \
  "${EXTRA[@]}" \
  --query 'Instances[0].InstanceId' --output text

rm -f "$USERDATA"
echo "Launched. Watch: aws s3 cp s3://$S3_BUCKET/$S3_PREFIX/$RUN_TAG/bootstrap.log - | tail"
