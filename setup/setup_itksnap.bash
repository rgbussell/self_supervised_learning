#!/bin/bash

# ITK-SNAP setup script for Ubuntu
# This installs ITK-SNAP from a local binary package
# Download this binary from the itk snap site
# NOTE: I had an issue installing the latest version 4.4.0, so using 4.0.1 instead
# I don't need anyhing specific from 4.4 so I am avoiding the issue for now

set -e

echo "This script will install ITK-SNAP, remove existing install dir and modify your PATH in ~/.bashrc"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

ITK_BIN_PATH="${HOME}/Downloads/itksnap-4.2.0-20240422-Linux-gcc64.tar.gz"

echo "Installing ITK-SNAP..."
echo "from local binary package at $ITK_BIN_PATH"

if [ ! -f "$ITK_BIN_PATH" ]; then
    echo "Error: ITK-SNAP binary not found at $ITK_BIN_PATH"
    exit 1
fi

# Update package manager
sudo apt-get update

# Create directory for ITK-SNAP
ITKSNAP_DIR="${HOME}/opt/itksnap"
rm -rf "$ITKSNAP_DIR"
mkdir -p "$ITKSNAP_DIR"
cd "$ITKSNAP_DIR"

# Extract
cp "$ITK_BIN_PATH" itksnap.tar.gz
tar -xzf itksnap.tar.gz
rm itksnap.tar.gz

# Add to PATH
ITKSNAP_BIN="$ITKSNAP_DIR/itksnap-4.2.0-20240422-Linux-gcc64/bin"
if ! grep -q "$ITKSNAP_BIN" ~/.bashrc; then
    echo "export PATH=\"\$PATH:$ITKSNAP_BIN\"" >> ~/.bashrc
fi

echo "ITK-SNAP installed successfully at $ITKSNAP_DIR"
echo "Run 'source ~/.bashrc' or restart your terminal"