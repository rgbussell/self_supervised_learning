#!/bin/bash
# download PET MRI data from openneuro
# using AWS CLI

set -e

DOWNLOAD_DIR=~/data/openneuro_pet_mri

echo "Downloading PET MRI data to $DOWNLOAD_DIR"
mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR

aws s3 sync --no-sign-request s3://openneuro.org/ds002898 ds002898-download/ --exclude "*/*" --include "*/sub-*/pet/*.nii.gz" --include "*/sub-*/anat/*.nii.gz"
#aws s3 sync --no-sign-request s3://openneuro.org/ds003382 ds003382-download/
#aws s3 sync --no-sign-request s3://openneuro.org/ds003397 ds003397-download/
#aws s3 sync --no-sign-request s3://openneuro.org/ds004733 ds004733-download/