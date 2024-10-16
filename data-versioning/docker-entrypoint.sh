#!/bin/bash

echo "Container is running!!!"

gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
mkdir -p /mnt/gcs_data
gcsfuse --key-file=$GOOGLE_APPLICATION_CREDENTIALS --implicit-dirs $GCS_BUCKET_NAME /mnt/gcs_data

# Check if the mounting was actually successful
if ls /mnt/gcs_data/car_preprocessed_folder; then
    echo "GCS bucket mounted successfully at /mnt/gcs_data"
else
    echo "Error: Failed to mount GCS bucket. Exiting."
    exit 1
fi

mkdir -p /app/car_preprocessed_dataset
mount --bind /mnt/gcs_data/car_preprocessed_folder/all_images /app/car_preprocessed_dataset

pipenv shell