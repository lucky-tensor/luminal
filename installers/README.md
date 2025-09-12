# Installers

This directory contains installation scripts and utilities for setting up dependencies and development environments for the Luminal project.

## Available Installers

### `setup-nvidia.sh`

Automated NVIDIA CUDA installation script for Ubuntu/Linux environments.

**Usage:**
```bash
chmod +x installers/setup-nvidia.sh
./installers/setup-nvidia.sh
```

**What it installs:**
- NVIDIA drivers (version 535)
- CUDA toolkit (version 12.2)
- Development headers and libraries
- Configures library paths

**Features:**
- ✅ Automatic version detection
- ✅ Clean installation (removes conflicts)
- ✅ CI/CD friendly
- ✅ Installation verification
- ✅ Detailed logging

**Environment Variables:**
- `SKIP_CLEANUP=true` - Skip removal of existing NVIDIA packages
- `FORCE_CI_MODE=true` - Force CI mode (skip DKMS installation)
- `CUDA_HOME` - Override CUDA installation path detection

**CI/Headless Environment Support:**
- Automatically detects CI environments (GitHub Actions, etc.)
- Skips DKMS installation when EFI is not available
- Provides appropriate error messages for headless systems

For detailed documentation, see [NVIDIA_SETUP.md](../NVIDIA_SETUP.md).

## Adding New Installers

When adding new installation scripts:

1. **Make them executable**: `chmod +x installers/your-script.sh`
2. **Follow naming convention**: `setup-<technology>.sh`
3. **Include verification**: Test that installation succeeded
4. **Add error handling**: Use `set -e` and proper error messages
5. **Document usage**: Update this README
6. **CI integration**: Update relevant GitHub workflows

## CI/CD Integration

These scripts are designed to work in automated environments:

```yaml
# GitHub Actions example
- name: Install dependencies
  run: |
    chmod +x ./installers/setup-nvidia.sh
    ./installers/setup-nvidia.sh
```

## Supported Platforms

| Script | Ubuntu | macOS | Windows |
|--------|--------|--------|---------|
| `setup-nvidia.sh` | ✅ | ❌ | ❌ |

## Contributing

When contributing new installers:
- Test on target platforms
- Include comprehensive error handling
- Follow the existing script structure
- Update documentation
- Test in CI environments