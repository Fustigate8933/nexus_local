#!/bin/bash
# Install Tauri dependencies for Arch Linux

echo "Installing Tauri dependencies for Arch Linux..."

# Check if running on Arch
if ! grep -q "Arch Linux" /etc/os-release 2>/dev/null; then
    echo "Warning: This script is for Arch Linux. You may need different packages."
    exit 1
fi

# Install required packages
sudo pacman -S --needed \
  webkit2gtk-4.1 \
  base-devel \
  curl \
  wget \
  openssl \
  alsa-lib \
  libxkbcommon \
  gtk3 \
  appmenu-gtk-module \
  libappindicator-gtk3

echo ""
echo "Dependencies installed!"
echo ""
echo "Verify installation:"
pkg-config --exists webkit2gtk-4.1 && echo "✓ webkit2gtk-4.1 found" || echo "✗ webkit2gtk-4.1 NOT found"
pkg-config --exists javascriptcoregtk-4.1 && echo "✓ javascriptcoregtk-4.1 found" || echo "✗ javascriptcoregtk-4.1 NOT found"
