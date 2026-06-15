#!/usr/bin/env bash
# Relaunch ONE fleet instance by tag after a spot interruption. Thanks to the bootstrap's
# S3 cache-restore, it RESUMES from where the interrupted instance left off (only un-cached
# decision-days are recomputed). Usage: bash infra/relaunch.sh <tag>
set -euo pipefail
tag="${1:?usage: relaunch.sh <tag>}"
REGION="${REGION:-ap-northeast-2}"; AMI="${AMI:-ami-029e8704d52728410}"
SUBNET="${SUBNET:-subnet-02db1a0228e65c480}"; SG="${SG:-sg-09e9566dd34e98a17}"

case "$tag" in
  ec2_32b)   ENV='export LEAKAGE_FORCE_TREATMENT=qwen3:32b LEAKAGE_CONTROL_MODEL=qwen2.5:32b LEAKAGE_SEEDS=0,1,2';;
  gemma_32b) ENV='export LEAKAGE_FORCE_TREATMENT=gemma3:27b LEAKAGE_CONTROL_MODEL=qwen2.5:32b LEAKAGE_SEEDS=0,1,2';;
  permodel_small) ENV='export LEAKAGE_RUN_MODULE=leakage.run.per_model_windows LEAKAGE_PERMODEL_ONLY=llama3.1:8b,qwen2.5:7b,qwen3:8b,gemma3:12b LEAKAGE_PULL_MODELS="llama3.1:8b qwen2.5:7b qwen3:8b gemma3:12b" LEAKAGE_SEEDS=0,1,2';;
  permodel_qwen3-32b)  ENV='export LEAKAGE_RUN_MODULE=leakage.run.per_model_windows LEAKAGE_PERMODEL_ONLY=qwen3:32b LEAKAGE_PULL_MODELS=qwen3:32b LEAKAGE_SEEDS=0,1,2';;
  permodel_qwen25-32b) ENV='export LEAKAGE_RUN_MODULE=leakage.run.per_model_windows LEAKAGE_PERMODEL_ONLY=qwen2.5:32b LEAKAGE_PULL_MODELS=qwen2.5:32b LEAKAGE_SEEDS=0,1,2';;
  permodel_gemma3-27b) ENV='export LEAKAGE_RUN_MODULE=leakage.run.per_model_windows LEAKAGE_PERMODEL_ONLY=gemma3:27b LEAKAGE_PULL_MODELS=gemma3:27b LEAKAGE_SEEDS=0,1,2';;
  permodel_mistral)    ENV='export LEAKAGE_RUN_MODULE=leakage.run.per_model_windows LEAKAGE_PERMODEL_ONLY=mistral-small3.2 LEAKAGE_PULL_MODELS=mistral-small3.2 LEAKAGE_SEEDS=0,1,2';;
  *) echo "unknown tag: $tag"; exit 1;;
esac

UD=$(mktemp)
{ echo '#!/usr/bin/env bash'
  echo "export S3_BUCKET=neuroxt-personal S3_PREFIX=yhjeon/finance-ai-leakage RUN_TAG=$tag"
  echo "$ENV"
  cat "$(dirname "$0")/bootstrap.sh"; } > "$UD"
aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type g6e.xlarge \
  --instance-market-options 'MarketType=spot' --iam-instance-profile Name=neuroxt-batch-instance-profile \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3,DeleteOnTermination=true}' \
  --subnet-id "$SUBNET" --security-group-ids "$SG" --user-data "file://$UD" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=leak-$tag},{Key=project,Value=finance-ai-leakage}]" \
  --query 'Instances[0].InstanceId' --output text
rm -f "$UD"
