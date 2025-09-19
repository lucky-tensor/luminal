# Qwen Example with Enhanced Timing and luminal_2 Framework

This is an enhanced version of the original Qwen example that demonstrates luminal_2's parallelization concepts and provides detailed performance analysis capabilities.

## Features

- **Enhanced Performance Analysis**: Detailed timing information for individual decode steps with statistical analysis
- **luminal_2 Framework Integration**: Demonstrates integration patterns with luminal_2's parallelization concepts
- **Parallel Execution Mode**: Optional `--parallel` flag to enable enhanced compilation and timing features
- **Statistical Timing**: Min/max/standard deviation analysis of decode performance

## Usage

### Basic Usage (Standard Mode)
```bash
# CUDA backend (recommended for this example)
cargo run --features cuda --release -- -p "Hello, world!"

# Metal backend (requires macOS)
cargo run --features metal --release -- -p "Hello, world!"
```

### With Enhanced Analysis
```bash
# Enable parallel mode with enhanced timing
cargo run --features cuda --release -- --parallel -p "Hello, world!"

# With detailed kernel timing analysis
cargo run --features cuda --release -- --parallel --kernel-timing -p "Hello, world!"
```

### Command Line Options

- `-p, --prompt <PROMPT>`: Input prompt text
- `-t, --gen-tokens <NUM>`: Number of tokens to generate (default: 256)
- `--parallel`: Enable enhanced compilation and timing features
- `--kernel-timing`: Show detailed per-token timing analysis with statistics

## Setup

Before running, ensure you have the model files set up:
1. Place your `qwen3-4b.gguf` model file in the `setup/` directory
2. Place your `tokenizer.json` file in the `setup/` directory

## Implementation Notes

This example demonstrates the foundation for integrating luminal_2's parallelization features:

- **Framework Integration**: Shows how to structure code for luminal_2 integration
- **Enhanced Timing**: Provides detailed performance profiling capabilities
- **Parallel Preparation**: Sets up infrastructure for future luminal_2 graph translation
- **Cross-Platform Compatibility**: Maintains compatibility while preparing for enhanced features

## Timing Analysis

With `--kernel-timing` enabled, you'll get detailed statistics:
- Average, min, max decode times per token
- Standard deviation of timing across tokens
- Per-token timing for the first 10 tokens
- Clear indication of enhanced features status

This provides valuable insights for performance optimization and demonstrates the kind of analysis that luminal_2's parallelization features enable.

## Future Enhancements

This example provides the foundation for full luminal_2 integration, including:
- Graph translation and parallelization
- Advanced kernel compilation
- Buffer management optimization
- Per-kernel timing analysis

The current implementation focuses on timing analysis and framework preparation while maintaining cross-platform compatibility.