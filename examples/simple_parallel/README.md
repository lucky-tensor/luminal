# Simple Parallel Neural Network Example

This example demonstrates parallel execution of neural network computations using Luminal with Rayon for CPU parallelization.

## Features

- **Parallel Neural Network Operations**: Runs multiple independent neural network computations in parallel
- **Performance Comparison**: Compare standard vs parallel execution with speedup metrics
- **Thread Utilization Reporting**: Shows exactly which threads are being used
- **Configurable Workloads**: Adjust operation count and tensor sizes
- **Detailed Timing**: Optional per-operation timing information

## Usage

### Basic Usage

```bash
# Run in standard mode
cargo run --features cuda --release

# Run in parallel mode
cargo run --features cuda --release -- --parallel

# Run with more operations to see better parallelization
cargo run --features cuda --release -- --parallel --operations 200
```

### Performance Comparison

```bash
# Compare standard vs parallel with timing details
cargo run --features cuda --release -- --timing

# Test with larger tensors
cargo run --features cuda --release -- --timing --input-size 2000 --output-size 1000

# Test with more operations
cargo run --features cuda --release -- --timing --operations 500
```

### Command Line Options

- `--parallel`: Enable parallel execution mode
- `--operations <N>`: Number of neural network operations to run (default: 100)
- `--input-size <N>`: Size of input tensors (default: 1000)
- `--output-size <N>`: Size of output tensors (default: 500)
- `--timing`: Show detailed timing information and run comparison

## What it demonstrates

1. **Parallel Graph Execution**: Each operation creates and executes its own computation graph in parallel
2. **Thread Pool Management**: Uses Rayon's thread pool to manage CPU cores efficiently
3. **Real Performance Gains**: Shows actual speedup from parallelization
4. **Thread Reporting**: Displays which threads handle which operations
5. **Scalability Testing**: Easily adjust workload size to test parallel performance

## Sample Output

```
🔬 Simple Parallel Neural Network Example
CPU cores available: 8

=== PERFORMANCE COMPARISON ===
🔧 Running in STANDARD mode
Operations: 100, Input size: 1000, Output size: 500
⏱️  Standard execution: 1234.56ms
📊 Average per operation: 12.35ms

🚀 Running in PARALLEL mode
Operations: 100, Input size: 1000, Output size: 500
Using 8 CPU threads
⚡ Parallel execution: 187.23ms
📊 Average per operation: 1.87ms
🧵 Thread utilization: 8 threads used

=== RESULTS ===
Standard time: 1234.56ms
Parallel time: 187.23ms
Speedup: 6.59x
Parallel efficiency: 82.4%
🎉 Parallel execution was 559.0% faster!
```

## Implementation Notes

- Creates independent computation graphs for true parallelization
- Uses Rayon for work-stealing parallel execution
- Each thread gets its own Luminal graph compilation and execution
- Real neural network computations (Linear layers with matrix multiplication)
- Measures actual wall-clock time for realistic performance metrics

This example provides a practical demonstration of how to parallelize Luminal computations across multiple CPU cores.