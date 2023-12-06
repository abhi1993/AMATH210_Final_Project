#!/bin/bash

set -e

export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export GCS_BUCKET_NAME="team-engai"
export GCP_PROJECT="AC215Project"
export GCP_ZONE="northamerica-northeast2"
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# A hack to fix a "cannot allocate memory in static TLS block" error from torch
export LD_PRELOAD="/root/.local/share/virtualenvs/app-4PlAip0Q/lib/python3.8/site-packages/torch/lib/../../torch.libs/libgomp-6e1a1d1b.so.1.0.0"

# Create the network if we don't have it yet
docker network inspect data-versioning-network >/dev/null 2>&1 || docker network create data-versioning-network

# Build the image based on the Dockerfile
docker build -t perform_lda --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm --name perform_lda -ti \
--mount type=bind,source="$BASE_DIR",target=/app \
-v ~/.gitconfig:/etc/gitconfig \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCP_ZONE=$GCP_ZONE \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e LD_PRELOAD=$LD_PRELOAD \
--network data-versioning-network process_synthetic_data

# Below replaced by `mount` above.
# -v "$BASE_DIR":/app \
# -v "$SECRETS_DIR":/secrets \