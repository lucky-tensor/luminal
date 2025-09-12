# GitHub Actions GPU Runners Setup

This document explains how to configure GitHub Actions to use GPU-powered runners for CUDA testing.

## Overview

The Luminal project uses GitHub Actions workflows to test CUDA functionality. These tests can run in two modes:

1. **Compilation Mode**: Tests CUDA code compilation without physical GPU
2. **Hardware Mode**: Tests actual GPU acceleration with physical hardware

## GitHub GPU Runners

GitHub Actions provides GPU-powered runners with NVIDIA T4 GPUs for GitHub Team and Enterprise customers.

### Specifications
- **Hardware**: 4 vCPUs, 28GB RAM, NVIDIA T4 GPU (16GB VRAM)
- **Cost**: $0.07 per minute
- **Availability**: GitHub Team or Enterprise Cloud plans only
- **Image**: NVIDIA GPU-Optimized Image for AI and HPC

## Setup Instructions

### Step 1: Configure GPU Runners

1. Go to your organization settings
2. Navigate to Actions > Runners
3. Click "New runner" → "New GitHub-hosted runner"
4. Select "Linux" platform
5. Choose "NVIDIA GPU-Optimized Image"
6. Select GPU-powered size (4-core, T4 GPU)
7. Create a runner group named `gpu-runners`

### Step 2: Update Repository Settings

In your repository's workflow files, update the `runs-on` field:

```yaml
# Replace 'your-org' with your GitHub organization name
runs-on: ${{ github.repository_owner == 'lucky-tensor' && 'gpu-runners' || 'ubuntu-latest' }}
```

### Step 3: Cost Control

GPU tests only run when:
- **Manual trigger**: Add `test-gpu` label to PRs
- **Workflow dispatch**: Manual workflow runs
- **Physical GPU detection**: Automatic detection of available hardware

## Current Workflow Configuration

### Simple Example Test
- **CPU Tests**: Always run on ubuntu-latest and macos-latest
- **CUDA Tests**: Only run with `test-gpu` label or workflow dispatch

### Phi Runtime Test  
- **Metal Test**: Always run on macOS (no additional cost)
- **CUDA Test**: Only run with GPU hardware or explicit request

## Usage

### Running GPU Tests

1. **For Pull Requests**: Add the `test-gpu` label
```bash
# Using GitHub CLI
gh pr edit <PR_NUMBER> --add-label "test-gpu"
```

2. **Manual Runs**: Use workflow dispatch in Actions tab

### Example Workflow Output

**With GPU Hardware:**
```
✅ GPU detected:
Tesla T4, 15360 MiB, 545.23.08
✅ Using physical GPU acceleration
```

**Without GPU (Compilation Only):**
```
ℹ️  No GPU detected, will install CUDA for compilation only
⚠️  Running in CPU simulation mode (no physical GPU)
```

## GPU Detection Logic

The workflows automatically detect GPU availability:

```bash
if command -v nvidia-smi &> /dev/null; then
  if nvidia-smi &> /dev/null; then
    echo "gpu_available=true"
  else  
    echo "gpu_available=false"
  fi
else
  echo "gpu_available=false"
fi
```

## Cost Management

### Estimated Costs
- **Simple CUDA test**: ~2-3 minutes = $0.14-0.21
- **Phi model test**: ~5 minutes = $0.35
- **Both tests**: ~$0.50 per run

### Best Practices
1. Use labels to control when GPU tests run
2. Keep test duration short (use `--gen_tokens 10`)
3. Monitor usage in billing dashboard
4. Consider self-hosted runners for frequent testing

## Alternative: Self-Hosted GPU Runners

For cost savings with frequent testing:

1. **AWS/Azure GPU instances**: ~$0.50-1.00/hour
2. **RunsOn.com**: Third-party service with lower GPU costs
3. **Local hardware**: For development teams with GPU machines

### Self-Hosted Setup
```yaml
# Use self-hosted runner with 'gpu' label
runs-on: [self-hosted, linux, gpu]
```

## Troubleshooting

### Common Issues

1. **"No GPU runners available"**
   - Verify runner group configuration
   - Check organization billing/plan status
   - Confirm runner group permissions

2. **"CUDA tests skipped"**
   - Tests only run with `test-gpu` label
   - Check workflow conditions and triggers

3. **"GPU detection failed"**
   - Normal on standard ubuntu-latest runners
   - Install CUDA for compilation testing

### Debug Commands
```bash
# Check GPU availability
nvidia-smi

# Check CUDA installation
nvcc --version

# Check driver version
cat /proc/driver/nvidia/version
```

## Repository Configuration

To enable GPU runners for your repository:

1. Replace `your-org` in workflow files with your organization name (already configured for `lucky-tensor`)
2. Create `gpu-runners` group in organization settings  
3. Configure billing and spending limits
4. Test with workflow dispatch first

## Security Considerations

- GPU runners use partner-managed images (not GitHub-managed)
- Review third-party image security policies
- Limit GPU runner access to trusted contributors
- Monitor costs and usage regularly

For more information, see [GitHub's official documentation](https://docs.github.com/en/actions/using-github-hosted-runners/using-larger-runners).