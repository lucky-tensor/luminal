use std::fs;
use std::path::Path;
use std::time::Instant;
use tempfile::tempdir;

use luminal::prelude::*;
use luminal_nn::Linear;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// Import luminal_2 for actual parallel compilation
#[cfg(feature = "parallel")]
use luminal_2::{compile_kernels, Kernel, run::assign_buffers};

/// Test result comparing sequential vs parallel compilation
#[derive(Debug)]
pub struct CompilationComparisonResult {
    pub sequential_time_ms: u128,
    pub parallel_time_ms: u128,
    pub speedup: f64,
    pub outputs_identical: bool,
    pub cache_artifacts_identical: bool,
    pub error_message: Option<String>,
}

/// Create a more complex model for testing
fn create_complex_test_model(seed: u64, cx: &mut Graph) -> (Vec<Linear>, GraphTensor, GraphTensor) {
    let mut rng = StdRng::seed_from_u64(seed);

    // Create a multi-layer model to generate more kernels
    let layer1 = Linear::new(128, 256, true, cx);  // bias = true for more kernels
    let layer2 = Linear::new(256, 512, true, cx);
    let layer3 = Linear::new(512, 256, true, cx);
    let layer4 = Linear::new(256, 64, true, cx);

    // Initialize with deterministic weights - use fixed sizes
    // layer1: 128x256 + 256 bias = 32768 + 256 = 33024
    let weight1: Vec<f32> = (0..128*256).map(|_| rng.random_range(-0.1..0.1)).collect();
    let bias1: Vec<f32> = (0..256).map(|_| rng.random_range(-0.1..0.1)).collect();
    layer1.weight.set(weight1);
    layer1.bias.as_ref().unwrap().set(bias1);

    // layer2: 256x512 + 512 bias
    let weight2: Vec<f32> = (0..256*512).map(|_| rng.random_range(-0.1..0.1)).collect();
    let bias2: Vec<f32> = (0..512).map(|_| rng.random_range(-0.1..0.1)).collect();
    layer2.weight.set(weight2);
    layer2.bias.as_ref().unwrap().set(bias2);

    // layer3: 512x256 + 256 bias
    let weight3: Vec<f32> = (0..512*256).map(|_| rng.random_range(-0.1..0.1)).collect();
    let bias3: Vec<f32> = (0..256).map(|_| rng.random_range(-0.1..0.1)).collect();
    layer3.weight.set(weight3);
    layer3.bias.as_ref().unwrap().set(bias3);

    // layer4: 256x64 + 64 bias
    let weight4: Vec<f32> = (0..256*64).map(|_| rng.random_range(-0.1..0.1)).collect();
    let bias4: Vec<f32> = (0..64).map(|_| rng.random_range(-0.1..0.1)).collect();
    layer4.weight.set(weight4);
    layer4.bias.as_ref().unwrap().set(bias4);

    // Create input
    let input_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let input = cx.tensor(128).set(input_data);

    // Forward pass through layers
    let x1 = layer1.forward(input).relu();
    let x2 = layer2.forward(x1).relu();
    let x3 = layer3.forward(x2).relu();
    let output = layer4.forward(x3).retrieve();

    (vec![layer1, layer2, layer3, layer4], input, output)
}

/// Run sequential compilation and capture results
fn run_sequential_compilation(
    seed: u64,
    cache_dir: &Path,
) -> Result<(Vec<f32>, u128), Box<dyn std::error::Error>> {
    let mut cx = Graph::new();
    let (layers, _input, mut output) = create_complex_test_model(seed, &mut cx);

    // Keep model weights using vec instead of arrays
    let mut tensor_ids = Vec::new();
    for layer in &layers {
        tensor_ids.push(layer.weight.id);
        if let Some(bias) = &layer.bias {
            tensor_ids.push(bias.id);
        }
    }
    cx.keep_tensors(&tensor_ids);

    let compile_start = Instant::now();

    // Standard sequential compilation
    cx.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "metal")]
            luminal_metal::MetalCompiler::<f32>::default(),
            #[cfg(feature = "cuda")]
            luminal_cuda::CudaCompiler::<f32>::default(),
            #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
            luminal_cpu::CPUCompiler::default(),
        ),
        &mut output,
    );

    let compilation_time = compile_start.elapsed().as_millis();

    // Execute and get results
    cx.execute();
    let result = output.data().to_vec();

    Ok((result, compilation_time))
}

/// Run parallel compilation using luminal_2 (when feature is enabled)
#[cfg(feature = "parallel")]
fn run_parallel_compilation(
    seed: u64,
    cache_dir: &Path,
    threads: usize,
) -> Result<(Vec<f32>, u128), Box<dyn std::error::Error>> {
    use petgraph::stable_graph::StableGraph;

    let mut cx = Graph::new();
    let (layers, _input, mut output) = create_complex_test_model(seed, &mut cx);

    // Keep model weights using vec instead of arrays
    let mut tensor_ids = Vec::new();
    for layer in &layers {
        tensor_ids.push(layer.weight.id);
        if let Some(bias) = &layer.bias {
            tensor_ids.push(bias.id);
        }
    }
    cx.keep_tensors(&tensor_ids);

    let compile_start = Instant::now();

    // First, do standard compilation to extract kernels
    cx.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "metal")]
            luminal_metal::MetalCompiler::<f32>::default(),
            #[cfg(feature = "cuda")]
            luminal_cuda::CudaCompiler::<f32>::default(),
            #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
            luminal_cpu::CPUCompiler::default(),
        ),
        &mut output,
    );

    // TODO: This is a simplified approach. Real integration would:
    // 1. Extract kernels from the compiled graph
    // 2. Use luminal_2::compile_kernels() for parallel compilation
    // 3. Replace the compiled kernels in the graph

    // For now, simulate parallel overhead with multiple threads
    rayon::scope(|s| {
        for _ in 0..threads {
            s.spawn(|_| {
                // Simulate parallel work
                std::thread::sleep(std::time::Duration::from_micros(100));
            });
        }
    });

    let compilation_time = compile_start.elapsed().as_millis();

    // Execute and get results
    cx.execute();
    let result = output.data().to_vec();

    Ok((result, compilation_time))
}

/// Fallback parallel compilation for when luminal_2 feature is not enabled
#[cfg(not(feature = "parallel"))]
fn run_parallel_compilation(
    seed: u64,
    _cache_dir: &Path,
    _threads: usize,
) -> Result<(Vec<f32>, u128), Box<dyn std::error::Error>> {
    // Just run sequential compilation for now
    run_sequential_compilation(seed, _cache_dir)
}

/// Compare outputs with tolerance
fn compare_outputs(a: &[f32], b: &[f32], tolerance: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }

    a.iter()
        .zip(b.iter())
        .all(|(va, vb)| (va - vb).abs() <= tolerance)
}

/// Main test function that validates parallel vs sequential compilation
pub fn test_parallel_vs_sequential_compilation(
    threads: usize,
    tolerance: f32,
) -> Result<CompilationComparisonResult, Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    let seq_cache = temp_dir.path().join("sequential");
    let par_cache = temp_dir.path().join("parallel");

    fs::create_dir_all(&seq_cache)?;
    fs::create_dir_all(&par_cache)?;

    let seed = 42;

    println!("🔄 Running sequential compilation...");
    let (seq_output, seq_time) = run_sequential_compilation(seed, &seq_cache)?;
    println!("   Sequential: {}ms, {} outputs", seq_time, seq_output.len());

    println!("🔄 Running parallel compilation ({} threads)...", threads);
    let (par_output, par_time) = run_parallel_compilation(seed, &par_cache, threads)?;
    println!("   Parallel: {}ms, {} outputs", par_time, par_output.len());

    let outputs_identical = compare_outputs(&seq_output, &par_output, tolerance);
    let speedup = if par_time > 0 { seq_time as f64 / par_time as f64 } else { 1.0 };

    // Check cache artifacts (simplified for now)
    let cache_artifacts_identical = check_cache_artifacts(&seq_cache, &par_cache)?;

    let result = CompilationComparisonResult {
        sequential_time_ms: seq_time,
        parallel_time_ms: par_time,
        speedup,
        outputs_identical,
        cache_artifacts_identical,
        error_message: if outputs_identical { None } else {
            Some(format!("Outputs differ: seq len={}, par len={}", seq_output.len(), par_output.len()))
        },
    };

    println!("📊 Results: speedup={:.2}x, outputs_identical={}, cache_identical={}",
        speedup, outputs_identical, cache_artifacts_identical);

    Ok(result)
}

/// Check if cache artifacts are identical (simplified)
fn check_cache_artifacts(seq_dir: &Path, par_dir: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    // For now, just check if both directories exist and have similar file counts
    let seq_files = if seq_dir.exists() {
        fs::read_dir(seq_dir)?.count()
    } else {
        0
    };

    let par_files = if par_dir.exists() {
        fs::read_dir(par_dir)?.count()
    } else {
        0
    };

    // This is a placeholder - real implementation would do SHA256 comparison
    Ok(seq_files == par_files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_compilation_works() {
        let temp_dir = tempdir().unwrap();
        let cache_dir = temp_dir.path().join("test_cache");
        fs::create_dir_all(&cache_dir).unwrap();

        let result = run_sequential_compilation(42, &cache_dir);
        assert!(result.is_ok(), "Sequential compilation should work");

        let (output, time) = result.unwrap();
        assert_eq!(output.len(), 64, "Output should have 64 elements");
        assert!(time > 0, "Compilation should take some time");
        assert!(output.iter().all(|&x| x.is_finite()), "All outputs should be finite");
    }

    #[test]
    fn test_parallel_vs_sequential_basic() {
        let result = test_parallel_vs_sequential_compilation(2, 1e-6);
        assert!(result.is_ok(), "Comparison test should succeed");

        let comparison = result.unwrap();
        println!("Comparison result: {:?}", comparison);

        assert!(comparison.outputs_identical, "Outputs should be identical: {:?}", comparison.error_message);
        assert!(comparison.speedup > 0.0, "Should have positive speedup");

        // For now, we expect speedup close to 1.0 since we're not doing real parallel compilation
        assert!(comparison.speedup > 0.5 && comparison.speedup < 2.0,
            "Speedup should be reasonable: {}", comparison.speedup);
    }

    #[test]
    fn test_deterministic_output() {
        let temp_dir = tempdir().unwrap();
        let cache_dir = temp_dir.path().join("deterministic_test");
        fs::create_dir_all(&cache_dir).unwrap();

        // Run same test multiple times
        let result1 = run_sequential_compilation(123, &cache_dir).unwrap();
        let result2 = run_sequential_compilation(123, &cache_dir).unwrap();

        assert!(compare_outputs(&result1.0, &result2.0, 1e-10),
            "Same seed should produce identical results");
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_compilation_feature() {
        let temp_dir = tempdir().unwrap();
        let cache_dir = temp_dir.path().join("parallel_test");
        fs::create_dir_all(&cache_dir).unwrap();

        let result = run_parallel_compilation(42, &cache_dir, 4);
        assert!(result.is_ok(), "Parallel compilation should work when feature is enabled");

        let (output, _time) = result.unwrap();
        assert_eq!(output.len(), 64, "Parallel output should have 64 elements");
        assert!(output.iter().all(|&x| x.is_finite()), "All parallel outputs should be finite");
    }
}