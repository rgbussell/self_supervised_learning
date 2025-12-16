#!/bin/bash
# Deno setup script
# install deno to download openneuro data

if command -v deno &> /dev/null; then
    echo "Deno is already installed, skipping installation."
else
    echo "Installing Deno..."
    curl -fsSL https://deno.land/install.sh | sh
fi

# install the openneuro cli
deno install -A --global jsr:@openneuro/cli -n openneuro