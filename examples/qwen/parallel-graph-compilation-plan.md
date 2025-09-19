# Parallel Graph Compilation Plan

## Investigation Summary

After investigating the current state of parallelization in the Qwen example, I found that:

1. **Parallel kernel compilation IS implemented** in `luminal_2/src/run.rs` (lines 95-159)
2. **The Qwen example uses the main `luminal` crate**, which does NOT have parallel compilation
3. **Thread pool configuration exists but is unused** - `main.rs:290-298` sets up rayon threads but main luminal doesn't use them
4. **Two separate parallelization efforts exist:**
   - Graph definition parallelization (`src/parallel_graph/` module)
   - Kernel compilation parallelization (`luminal_2` crate)

## Problem Analysis

### Why Compilation is Single-Threaded

The Qwen example's compilation bottleneck occurs because:

```rust
// In main.rs:447-469, the compilation uses main luminal
cx.compile((
    GenericCompiler::default(),
    // GPU-specific compilers that compile kernels sequentially
    luminal_metal::MetalCompilerPreBuffer::<f32>::default(),
    // ...
))
```

The main `luminal::Graph::compile()` method (line 133-138 in `src/graph.rs`) simply delegates to individual compiler implementations, which process kernels sequentially.

### Existing Parallel Implementation

`luminal_2/src/run.rs` contains a fully functional parallel kernel compilation implementation:

**For CUDA** (lines 119-146):
```rust
let compiled_results: Vec<_> = unique_kernels
    .par_iter()  // Parallel execution using rayon
    .map(|kernel| {
        // CPU-intensive PTX compilation in parallel
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(&kernel.code, options)
    })
    .collect();

// Sequential module loading (CUDA context is not thread-safe)
for (code, ptx) in compiled_results {
    let module = ctx.load_module(ptx).unwrap();
    // ...
}
```

**For Metal** (lines 175-190):
```rust
let compiled_results: Vec<_> = unique_kernels
    .par_iter()  // Parallel execution
    .map(|kernel| {
        let thread_device = MTLCreateSystemDefaultDevice().unwrap();
        let lib = thread_device.newLibraryWithSource_options_error(/*...*/);
        // ...
    })
    .collect();
```

## Design Plan

### Phase 1: Integration with Existing luminal_2 Implementation

**Objective**: Integrate the existing parallel compilation from `luminal_2` into the Qwen example.

#### 1.1 Modify Qwen Example Dependencies
```toml
# Add to Cargo.toml
[dependencies]
luminal_2 = { path = "../../crates/luminal_2" }
```

#### 1.2 Create Parallel Compilation Bridge
Create a new module `src/parallel_compilation.rs`:

```rust
use luminal_2::{compile_kernels, Kernel};
use luminal::prelude::*;

pub struct ParallelGraphCompiler {
    thread_count: Option<usize>,
}

impl ParallelGraphCompiler {
    pub fn new(thread_count: Option<usize>) -> Self {
        Self { thread_count }
    }

    pub fn compile_with_parallel_kernels<T: ToIdsMut>(
        &self,
        graph: &mut Graph,
        base_compiler: impl Compiler,
        remap: T,
    ) -> CompilerOutput {
        // Extract kernels from graph
        // Use luminal_2::compile_kernels for parallel compilation
        // Integrate results back into the main compilation pipeline
    }
}
```

#### 1.3 Integration Points
- **Extract kernel graph**: Convert `luminal::Graph` operations to `luminal_2::Kernel` format
- **Parallel compilation**: Use `luminal_2::compile_kernels()`
- **Result integration**: Map compiled kernels back to main compilation pipeline

### Phase 2: Enhanced Parallel Architecture

**Objective**: Design a comprehensive parallel compilation system.

#### 2.1 Multi-Stage Parallel Pipeline

```rust
pub struct ParallelCompilationPipeline {
    stages: Vec<CompilationStage>,
    thread_pool: rayon::ThreadPool,
}

enum CompilationStage {
    GraphAnalysis,     // Analyze dependencies, extract kernel subgraphs
    KernelExtraction,  // Convert subgraphs to kernel code
    ParallelCompile,   // Compile kernels in parallel
    ModuleLoading,     // Sequential GPU module loading
    Optimization,      // Post-compilation optimizations
}
```

#### 2.2 Kernel Batching Strategy

**Small Kernels**: Batch multiple small kernels together to reduce overhead
**Large Kernels**: Compile independently in parallel
**Dependencies**: Respect compilation order for interdependent kernels

```rust
struct KernelBatch {
    kernels: Vec<Kernel>,
    dependencies: Vec<usize>, // Indices of required previous batches
    estimated_compile_time: Duration,
}
```

#### 2.3 Progress Tracking Integration

Integrate with existing progress tracking system:

```rust
pub struct CompilationProgress {
    total_kernels: usize,
    compiled_kernels: AtomicUsize,
    current_stage: AtomicUsize,
    stage_names: Vec<&'static str>,
}
```

### Phase 3: Testing and Validation

#### 3.1 Performance Benchmarks

**Compilation Time Measurement**:
- Sequential vs parallel compilation times
- Thread scaling analysis (1, 2, 4, 8, 16 threads)
- Memory usage comparison

**Test Setup**:
```rust
#[cfg(test)]
mod benchmarks {
    use criterion::{criterion_group, criterion_main, Criterion};

    fn benchmark_compilation_methods(c: &mut Criterion) {
        let mut group = c.benchmark_group("kernel_compilation");

        group.bench_function("sequential", |b| {
            b.iter(|| compile_qwen_sequential())
        });

        group.bench_function("parallel_2_threads", |b| {
            b.iter(|| compile_qwen_parallel(2))
        });

        group.bench_function("parallel_4_threads", |b| {
            b.iter(|| compile_qwen_parallel(4))
        });
        // ...
    }
}
```

#### 3.2 Correctness Validation

**Functional Tests**:
- Verify parallel-compiled graphs produce identical results
- Test with different thread counts
- Validate on both CUDA and Metal backends

**Integration Tests**:
```rust
#[test]
fn test_parallel_compilation_correctness() {
    let sequential_results = run_qwen_with_sequential_compilation();
    let parallel_results = run_qwen_with_parallel_compilation();

    assert_eq!(sequential_results.output_tokens, parallel_results.output_tokens);
    assert_eq!(sequential_results.execution_time.as_millis() / 10,
               parallel_results.execution_time.as_millis() / 10); // Allow 10% variance
}
```

#### 3.3 Stress Testing

**Large Model Testing**:
- Test with different model sizes
- Memory pressure scenarios
- Thread exhaustion scenarios

**Error Handling**:
- Compilation failure recovery
- Thread panic handling
- GPU resource exhaustion

## Implementation Timeline

### Week 1: Foundation
- [ ] Integrate `luminal_2` dependency
- [ ] Create basic parallel compilation bridge
- [ ] Implement kernel extraction from main `luminal::Graph`

### Week 2: Core Implementation
- [ ] Complete parallel compilation pipeline
- [ ] Add progress tracking integration
- [ ] Implement both CUDA and Metal support

### Week 3: Testing & Optimization
- [ ] Performance benchmarking suite
- [ ] Correctness validation tests
- [ ] Memory usage optimization

### Week 4: Integration & Documentation
- [ ] CLI integration with existing `--parallel_threads` flag
- [ ] Documentation and examples
- [ ] Final performance validation

## Success Metrics

1. **Performance**: 2-4x speedup in compilation time with 4+ CPU cores
2. **Scalability**: Linear speedup up to CPU core count
3. **Compatibility**: 100% functional compatibility with sequential compilation
4. **Reliability**: Pass all existing tests and new parallel-specific tests
5. **Memory**: No more than 50% increase in peak memory usage

## Risk Mitigation

### Technical Risks

**GPU Context Thread Safety**:
- Solution: Keep module loading sequential, only parallelize CPU-intensive compilation

**Memory Pressure**:
- Solution: Implement compilation batching to limit concurrent memory usage

**Dependency Management**:
- Solution: Topological sorting of kernel dependencies before parallel execution

### Integration Risks

**Backward Compatibility**:
- Solution: Make parallel compilation opt-in via CLI flags

**Debugging Complexity**:
- Solution: Comprehensive logging and ability to fall back to sequential compilation

## Future Enhancements

1. **Dynamic Load Balancing**: Adjust thread allocation based on kernel complexity
2. **Caching**: Cache compiled kernels across runs
3. **Distributed Compilation**: Compile across multiple machines for very large models
4. **GPU-Assisted Compilation**: Use GPU for certain compilation stages

## Conclusion

The parallel kernel compilation infrastructure already exists in `luminal_2`. The main task is integrating this proven implementation into the Qwen example and ensuring robust testing. This approach minimizes risk while providing significant performance improvements for large model compilation.

The estimated 2-4x compilation speedup will substantially improve the development and experimentation workflow for large models like Qwen.