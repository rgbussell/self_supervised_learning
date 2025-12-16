#!/bin/bash
# install aws cli

cd /tmp
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
sudo ./aws/install --install-dir /opt/aws-cli --bin-dir /usr/local/bin
rm -rf /tmp/awscliv2.zip /tmp/aws

echo "AWS CLI installed successfully."
echo "You can verify the installation by running 'aws --version'"
echo aws --version