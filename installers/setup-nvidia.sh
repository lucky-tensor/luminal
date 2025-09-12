#!/bin/bash

# NVIDIA CUDA Setup Script
# This script installs NVIDIA drivers and CUDA toolkit for CI/development environments

set -e

# Configuration
NVIDIA_DRIVER_VERSION="535"
CUDA_VERSION_MAJOR="12"
CUDA_VERSION_MINOR="2" 
CUDA_VERSION="${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}"
CUDA_VERSION_FULL="${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}"
UBUNTU_VERSION="2204"

echo "🚀 Starting NVIDIA CUDA ${CUDA_VERSION_FULL} setup..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_NAME="$NAME"
        OS_VERSION="$VERSION_ID"
        echo "📋 Detected OS: $OS_NAME $OS_VERSION"
    else
        echo "❌ Cannot detect Linux distribution"
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "❌ macOS is not supported for NVIDIA CUDA installation"
    exit 1
else
    echo "❌ Unsupported operating system: $OSTYPE"
    exit 1
fi

# Check if already installed
if command -v nvcc &> /dev/null && nvidia-smi &> /dev/null; then
    INSTALLED_CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "ℹ️  CUDA $INSTALLED_CUDA_VERSION is already installed"
    if [ "$INSTALLED_CUDA_VERSION" == "$CUDA_VERSION_FULL" ]; then
        echo "✅ CUDA version matches expected version $CUDA_VERSION_FULL"
        exit 0
    else
        echo "⚠️  CUDA version mismatch. Expected: $CUDA_VERSION_FULL, Found: $INSTALLED_CUDA_VERSION"
        echo "🔄 Proceeding with installation..."
    fi
fi

# Function to cleanup previous installations
cleanup_nvidia() {
    echo "🧹 Cleaning up previous NVIDIA installations..."
    sudo apt-get remove --purge -y "nvidia-*" "cuda-*" "*nvidia*" || true
    sudo apt-get autoremove -y || true
    sudo apt-get autoclean || true
}

# Function to install NVIDIA drivers
install_nvidia_drivers() {
    echo "🔧 Installing NVIDIA drivers ${NVIDIA_DRIVER_VERSION}..."
    
    # Update package lists
    sudo apt-get update
    
    # Install drivers
    sudo apt-get install -y \
        nvidia-driver-${NVIDIA_DRIVER_VERSION} \
        nvidia-dkms-${NVIDIA_DRIVER_VERSION} \
        nvidia-utils-${NVIDIA_DRIVER_VERSION}
    
    echo "✅ NVIDIA drivers installed"
}

# Function to install CUDA toolkit
install_cuda_toolkit() {
    echo "🔧 Installing CUDA toolkit ${CUDA_VERSION_FULL}..."
    
    # Download and install CUDA keyring
    local keyring_file="cuda-keyring_1.1-1_all.deb"
    local keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/${keyring_file}"
    
    if [ ! -f "$keyring_file" ]; then
        echo "📥 Downloading CUDA keyring..."
        wget -q "$keyring_url" || {
            echo "❌ Failed to download CUDA keyring"
            exit 1
        }
    fi
    
    sudo dpkg -i "$keyring_file" || {
        echo "❌ Failed to install CUDA keyring"
        exit 1
    }
    
    # Update package lists
    sudo apt-get update
    
    # Install CUDA toolkit components
    echo "📦 Installing CUDA toolkit components..."
    sudo apt-get install -y \
        cuda-compiler-${CUDA_VERSION} \
        cuda-libraries-dev-${CUDA_VERSION} \
        cuda-driver-dev-${CUDA_VERSION} \
        cuda-cudart-dev-${CUDA_VERSION} \
        cuda-runtime-${CUDA_VERSION} \
        cuda-nvrtc-dev-${CUDA_VERSION} \
        cuda-nvml-dev-${CUDA_VERSION}
    
    echo "✅ CUDA toolkit installed"
}

# Function to setup library paths
setup_library_paths() {
    echo "🔗 Setting up library paths..."
    
    local cuda_lib_path="/usr/local/cuda-${CUDA_VERSION_FULL}/lib64"
    local system_lib_path="/usr/lib/x86_64-linux-gnu"
    
    # Add to ldconfig
    echo "$cuda_lib_path" | sudo tee -a /etc/ld.so.conf > /dev/null
    echo "$system_lib_path" | sudo tee -a /etc/ld.so.conf > /dev/null
    sudo ldconfig
    
    # Create symlink for /usr/local/cuda if it doesn't exist
    if [ ! -L /usr/local/cuda ]; then
        sudo ln -sf "/usr/local/cuda-${CUDA_VERSION_FULL}" /usr/local/cuda
    fi
    
    echo "✅ Library paths configured"
}

# Function to verify installation
verify_installation() {
    echo "🔍 Verifying CUDA installation..."
    
    # Check nvidia-smi
    if nvidia-smi &> /dev/null; then
        echo "✅ nvidia-smi working"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        echo "⚠️  nvidia-smi not available (this is expected in CI environments without GPU)"
    fi
    
    # Check nvcc
    if command -v nvcc &> /dev/null; then
        echo "✅ nvcc available"
        nvcc --version
    else
        echo "❌ nvcc not found"
        return 1
    fi
    
    # Check CUDA libraries
    local cuda_lib_path="/usr/local/cuda-${CUDA_VERSION_FULL}/lib64"
    if [ -d "$cuda_lib_path" ]; then
        echo "✅ CUDA libraries found at $cuda_lib_path"
        ls -la "${cuda_lib_path}"/libcuda* 2>/dev/null || echo "⚠️  libcuda not found in CUDA lib path"
    else
        echo "⚠️  CUDA lib directory not found"
    fi
    
    # Check system libraries
    ls -la /usr/lib/x86_64-linux-gnu/libcuda* 2>/dev/null || echo "⚠️  System libcuda not found"
    
    # Check CUDA headers
    local cuda_include_path="/usr/local/cuda-${CUDA_VERSION_FULL}/include"
    if [ -f "${cuda_include_path}/cuda_fp16.h" ]; then
        echo "✅ CUDA headers found (cuda_fp16.h exists)"
    else
        echo "❌ CUDA headers not found"
        return 1
    fi
    
    echo "✅ CUDA installation verification completed"
}

# Main installation process
main() {
    echo "🎯 Target configuration:"
    echo "   - NVIDIA Driver: ${NVIDIA_DRIVER_VERSION}"
    echo "   - CUDA Version: ${CUDA_VERSION_FULL}"
    echo "   - Ubuntu Version: ${UBUNTU_VERSION}"
    echo ""
    
    # Skip cleanup if SKIP_CLEANUP is set
    if [ "${SKIP_CLEANUP:-}" != "true" ]; then
        cleanup_nvidia
    else
        echo "⏭️  Skipping cleanup (SKIP_CLEANUP=true)"
    fi
    
    install_nvidia_drivers
    install_cuda_toolkit
    setup_library_paths
    verify_installation
    
    echo ""
    echo "🎉 NVIDIA CUDA ${CUDA_VERSION_FULL} setup completed successfully!"
    echo ""
    echo "📋 Environment setup:"
    echo "   export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION_FULL}"
    echo "   export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "   export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"
}

# Run main function
main "$@"