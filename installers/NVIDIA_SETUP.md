# NVIDIA CUDA Setup

This document describes how to set up NVIDIA CUDA for the Luminal project.

## Automated Setup Script

The project includes a `installers/setup-nvidia.sh` script that automatically installs the required NVIDIA drivers and CUDA toolkit.

### Usage

```bash
# Make executable and run
chmod +x installers/setup-nvidia.sh
./installers/setup-nvidia.sh
```

### What it installs

- **NVIDIA Drivers**: Version 535 (compatible with CUDA 12.2)
- **CUDA Toolkit**: Version 12.2
- **Components**:
  - CUDA compiler (`nvcc`)
  - CUDA runtime libraries
  - CUDA development headers
  - cuRAND, cuBLAS, and other essential libraries

### Configuration

The script configures the following:

- **CUDA Path**: `/usr/local/cuda-12.2/`
- **Libraries**: Adds CUDA libs to `LD_LIBRARY_PATH`
- **Symlink**: Creates `/usr/local/cuda` → `/usr/local/cuda-12.2`
- **Headers**: Installs to `/usr/local/cuda-12.2/include/`

### Environment Variables

After installation, you may want to set:

```bash
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

## Manual Setup

If you prefer manual installation:

### 1. Install NVIDIA Drivers

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-535 nvidia-dkms-535
```

### 2. Install CUDA Toolkit

```bash
# Download CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Install CUDA components
sudo apt-get update
sudo apt-get install -y cuda-compiler-12-2 cuda-libraries-dev-12-2 cuda-driver-dev-12-2 cuda-cudart-dev-12-2 cuda-runtime-12-2
```

### 3. Configure Libraries

```bash
echo "/usr/local/cuda-12.2/lib64" | sudo tee -a /etc/ld.so.conf
echo "/usr/lib/x86_64-linux-gnu" | sudo tee -a /etc/ld.so.conf
sudo ldconfig
```

## Verification

Check that everything is working:

```bash
# Check CUDA compiler
nvcc --version

# Check NVIDIA driver (may not work in CI)
nvidia-smi

# Check CUDA headers
ls -la /usr/local/cuda-12.2/include/cuda_fp16.h
```

## Running Luminal Examples with CUDA

Once CUDA is installed, you can run examples with GPU acceleration:

```bash
# Simple linear algebra example
cargo run --manifest-path examples/simple/Cargo.toml --features cuda

# Neural network training
cargo run --manifest-path examples/train_math_net/Cargo.toml --features cuda

# Language model inference
cargo run --manifest-path examples/llama/Cargo.toml --features cuda
```

## Troubleshooting

### Common Issues

1. **"cuda_fp16.h not found"**
   - The automated script fixes CUDA include path detection
   - Ensure CUDA headers are installed: `sudo apt-get install cuda-cudart-dev-12-2`

2. **"nvidia-smi: command not found"**
   - Normal in CI environments without physical GPUs
   - CUDA compilation should still work

3. **Library linking errors**
   - Run `sudo ldconfig` to refresh library cache
   - Check `LD_LIBRARY_PATH` includes CUDA lib directories

### Version Compatibility

- **CUDA 12.2** - Recommended, tested version
- **Driver 535** - Compatible with CUDA 12.2
- **Ubuntu 22.04** - Tested platform

### CI/CD Usage

The setup script is designed to work in GitHub Actions and other CI environments:

```yaml
- name: Install CUDA and NVIDIA drivers
  run: |
    chmod +x ./installers/setup-nvidia.sh
    ./installers/setup-nvidia.sh
```

For environments where cleanup might be problematic:

```bash
SKIP_CLEANUP=true ./installers/setup-nvidia.sh
```