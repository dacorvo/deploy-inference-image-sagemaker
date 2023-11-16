#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <local_image> <aws_account_id> <aws_region>"
fi

# Pass arguments
BASE_IMAGE="$1"
AWS_ACCOUNT_ID="$2"
AWS_REGION="$3"

# Verify the base image exists locally
if [[ "$(docker images -q $BASE_IMAGE 2> /dev/null)" == "" ]]; then
  echo "ERROR: ${BASE_IMAGE} not found locally."
  exit -1
fi

# Set the name and tag for the ECR repository and tag
ECR_REPOSITORY=$(echo $BASE_IMAGE | cut -d ":" -f 1)
ECR_IMAGE_TAG=$(echo $BASE_IMAGE | cut -d ":" -f 2)

# Log in to ECR
echo "Logging in to Amazon ECR..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# Create the ECR repository if it doesn't already exist
echo "Creating ECR repository $ECR_REPOSITORY..."
aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" || aws ecr create-repository --repository-name "$ECR_REPOSITORY"

# Tag the Docker Hub image with the ECR repository and image tag
echo "Tagging image $BASE_IMAGE as $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$ECR_IMAGE_TAG..."
docker tag "$BASE_IMAGE" "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$ECR_IMAGE_TAG"

# Push the Docker image to ECR
echo "Pushing image to ECR..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$ECR_IMAGE_TAG"

# Clean up local images
echo "Cleaning up ECR tag ..."
docker rmi "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$ECR_IMAGE_TAG"
