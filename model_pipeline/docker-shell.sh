#!/bin/bash

set -e

export IMAGE_NAME="215_model_pipeline"
export BASE_DIR="$(pwd)"

echo "Building Docker image..."
docker build -t $IMAGE_NAME .

echo "Running Docker container..."
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
$IMAGE_NAME