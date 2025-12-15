#!/bin/bash

# slicer setup script for Ubuntu
# This installs slicer from a local binary package
# Download this binary from the slicer site: https://download.slicer.org/

set -e

echo "This script will install slicer, remove existing install dir and modify your PATH in ~/.bashrc"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

SLICER_BIN_PATH="${HOME}/Downloads/Slicer-5.10.0-linux-amd64.tar.gz"

echo "Installing slicer..."
echo "from local binary package at $SLICER_BIN_PATH"

if [ ! -f "$SLICER_BIN_PATH" ]; then
    echo "Error: Slicer binary not found at $SLICER_BIN_PATH"
    exit 1
fi

# Create directory for Slicer
SLICER_DIR="${HOME}/opt/slicer"
rm -rf "$SLICER_DIR"
mkdir -p "$SLICER_DIR"
cd "$SLICER_DIR"

# Extract
cp "$SLICER_BIN_PATH" slicer.tar.gz
tar -xzf slicer.tar.gz
rm slicer.tar.gz

# Add to PATH
SLICER_BIN="$SLICER_DIR/Slicer-5.10.0-linux-amd64"
if ! grep -q "$SLICER_BIN" ~/.bashrc; then
    echo "export PATH=\"\$PATH:$SLICER_BIN\"" >> ~/.bashrc
fi

echo "Slicer installed successfully at $SLICER_DIR"
echo "Run 'source ~/.bashrc' or restart your terminal"