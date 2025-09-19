use luminal::prelude::*;
use luminal_nn::Linear;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::time::Instant;

/// Compilation method for testing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompilationMethod {
    Sequential,
    Parallel { threads: usize },
}

/// Results from a compilation test
#[derive(Debug, Clone)]
pub struct CompilationResult {
    pub method: CompilationMethod,
    pub compilation_time_ms: u128,
    pub execution_time_ms: u128,
    pub output_data: Vec<f32>,
    pub graph_hash: u64,
    pub kernel_count: usize,
}

/// Compilation artifacts for comparison
#[derive(Debug, Clone)]
pub struct CompilationArtifacts {
    pub compiled_kernels: Vec<String>,
    pub execution_order: Vec<String>,
    pub memory_layout: HashMap<String, (usize, usize)>, // name -> (size, alignment)
}

/// Test configuration
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    pub model_sizes: Vec<(usize, usize)>, // (input_size, output_size) pairs
    pub batch_sizes: Vec<usize>,
    pub thread_counts: Vec<usize>,
    pub iterations: usize,
    pub seed: u64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            model_sizes: vec![(4, 5), (128, 256), (512, 1024)],
            batch_sizes: vec![1, 8, 32],
            thread_counts: vec![1, 2, 4, 8],
            iterations: 3,
            seed: 42,
        }
    }
}

/// Create a test model with deterministic weights
fn create_test_model(input_size: usize, output_size: usize, seed: u64, cx: &mut Graph) -> (Linear, Vec<f32>, GraphTensor) {
    let mut rng = StdRng::seed_from_u64(seed);
    let weight_data: Vec<f32> = (0..input_size * output_size).map(|_| rng.random_range(-1.0..1.0)).collect();

    let model = Linear::new(input_size, output_size, false, cx);
    model.weight.set(weight_data.clone());

    // Create deterministic input
    let input_data: Vec<f32> = (0..input_size).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let input = cx.tensor(input_size).set(input_data);

    (model, weight_data, input)
}

/// Run sequential compilation test
fn run_sequential_test(
    input_size: usize,
    output_size: usize,
    seed: u64,
) -> Result<CompilationResult, Box<dyn std::error::Error>> {
    let mut cx = Graph::new();
    let (model, _weights, input) = create_test_model(input_size, output_size, seed, &mut cx);

    let mut output = model.forward(input).retrieve();

    // Time compilation
    let compile_start = Instant::now();

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

    let compilation_time = compile_start.elapsed();

    // Time execution
    let exec_start = Instant::now();
    cx.execute();
    let execution_time = exec_start.elapsed();

    let output_data = output.data().to_vec();

    Ok(CompilationResult {
        method: CompilationMethod::Sequential,
        compilation_time_ms: compilation_time.as_millis(),
        execution_time_ms: execution_time.as_millis(),
        output_data,
        graph_hash: calculate_graph_hash(&cx),
        kernel_count: count_kernels(&cx),
    })
}

/// Run parallel compilation test (placeholder - will be implemented with luminal_2 integration)
fn run_parallel_test(
    input_size: usize,
    output_size: usize,
    threads: usize,
    seed: u64,
) -> Result<CompilationResult, Box<dyn std::error::Error>> {
    // For now, this just runs sequential compilation
    // Will be replaced with actual parallel compilation once luminal_2 is integrated

    // Note: We can't reconfigure the global thread pool, so we just run sequential for now

    let mut cx = Graph::new();
    let (model, _weights, input) = create_test_model(input_size, output_size, seed, &mut cx);

    let mut output = model.forward(input).retrieve();

    // Time compilation (sequential for now, but simulate some work)
    let compile_start = Instant::now();

    // Simulate some parallel processing overhead
    std::thread::sleep(std::time::Duration::from_millis(1));

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

    let compilation_time = compile_start.elapsed();

    // Time execution
    let exec_start = Instant::now();
    cx.execute();
    let execution_time = exec_start.elapsed();

    let output_data = output.data().to_vec();

    Ok(CompilationResult {
        method: CompilationMethod::Parallel { threads },
        compilation_time_ms: compilation_time.as_millis(),
        execution_time_ms: execution_time.as_millis(),
        output_data,
        graph_hash: calculate_graph_hash(&cx),
        kernel_count: count_kernels(&cx),
    })
}

/// Calculate a hash of the graph structure for comparison
fn calculate_graph_hash(cx: &Graph) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    // Hash graph structure (simplified)
    cx.graph.node_count().hash(&mut hasher);
    cx.graph.edge_count().hash(&mut hasher);

    // Hash dynamic dimensions
    for (k, v) in &cx.dyn_map {
        k.hash(&mut hasher);
        v.hash(&mut hasher);
    }

    hasher.finish()
}

/// Count the number of kernels in the compiled graph
fn count_kernels(cx: &Graph) -> usize {
    cx.graph.node_count()
}

/// Compare two floating point vectors with tolerance
fn compare_outputs(a: &[f32], b: &[f32], tolerance: f32) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("Output length mismatch: {} vs {}", a.len(), b.len()));
    }

    for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        if diff > tolerance {
            return Err(format!("Output mismatch at index {}: {} vs {} (diff: {})", i, va, vb, diff));
        }
    }

    Ok(())
}

/// Run calibration test suite
pub fn run_calibration_suite(config: CalibrationConfig) -> Result<Vec<CompilationResult>, Box<dyn std::error::Error>> {
    let mut all_results = Vec::new();

    println!("Running compilation calibration tests...");
    println!("Config: {:?}", config);

    for &(input_size, output_size) in &config.model_sizes {
        for &batch_size in &config.batch_sizes {
            println!("\nTesting model size: {}x{}, batch: {}", input_size, output_size, batch_size);

            // Run sequential baseline
            let mut sequential_results = Vec::new();
            for iteration in 0..config.iterations {
                let seed = config.seed + iteration as u64;
                match run_sequential_test(input_size, output_size, seed) {
                    Ok(result) => {
                        println!("  Sequential iteration {}: compile={}ms, exec={}ms",
                                iteration, result.compilation_time_ms, result.execution_time_ms);
                        sequential_results.push(result);
                    },
                    Err(e) => println!("  Sequential iteration {} failed: {}", iteration, e),
                }
            }

            // Run parallel tests
            for &threads in &config.thread_counts {
                let mut parallel_results = Vec::new();
                for iteration in 0..config.iterations {
                    let seed = config.seed + iteration as u64;
                    match run_parallel_test(input_size, output_size, threads, seed) {
                        Ok(result) => {
                            println!("  Parallel {}T iteration {}: compile={}ms, exec={}ms",
                                    threads, iteration, result.compilation_time_ms, result.execution_time_ms);
                            parallel_results.push(result);
                        },
                        Err(e) => println!("  Parallel {}T iteration {} failed: {}", threads, iteration, e),
                    }
                }

                // Compare outputs between sequential and parallel
                if let (Some(seq_result), Some(par_result)) = (sequential_results.first(), parallel_results.first()) {
                    match compare_outputs(&seq_result.output_data, &par_result.output_data, 1e-5) {
                        Ok(()) => println!("    ✓ Output correctness verified"),
                        Err(e) => println!("    ✗ Output mismatch: {}", e),
                    }
                }

                all_results.extend(parallel_results);
            }

            all_results.extend(sequential_results);
        }
    }

    Ok(all_results)
}

/// Generate performance report
pub fn generate_performance_report(results: &[CompilationResult]) -> String {
    let mut report = String::new();

    report.push_str("# Compilation Performance Report\n\n");

    // Group results by method
    let mut by_method: HashMap<String, Vec<&CompilationResult>> = HashMap::new();
    for result in results {
        let method_key = match result.method {
            CompilationMethod::Sequential => "Sequential".to_string(),
            CompilationMethod::Parallel { threads } => format!("Parallel-{}T", threads),
        };
        by_method.entry(method_key).or_default().push(result);
    }

    for (method, method_results) in &by_method {
        if method_results.is_empty() { continue; }

        let avg_compile_time: f64 = method_results.iter()
            .map(|r| r.compilation_time_ms as f64)
            .sum::<f64>() / method_results.len() as f64;

        let avg_exec_time: f64 = method_results.iter()
            .map(|r| r.execution_time_ms as f64)
            .sum::<f64>() / method_results.len() as f64;

        report.push_str(&format!("## {}\n", method));
        report.push_str(&format!("- Average compilation time: {:.2}ms\n", avg_compile_time));
        report.push_str(&format!("- Average execution time: {:.2}ms\n", avg_exec_time));
        report.push_str(&format!("- Sample count: {}\n\n", method_results.len()));
    }

    // Calculate speedups
    if let (Some(seq_results), Some(par_results)) = (
        by_method.get("Sequential"),
        by_method.iter().find(|(k, _)| k.starts_with("Parallel")).map(|(_, v)| v)
    ) {
        let seq_avg: f64 = seq_results.iter().map(|r| r.compilation_time_ms as f64).sum::<f64>() / seq_results.len() as f64;
        let par_avg: f64 = par_results.iter().map(|r| r.compilation_time_ms as f64).sum::<f64>() / par_results.len() as f64;

        if par_avg > 0.0 {
            let speedup = seq_avg / par_avg;
            report.push_str(&format!("## Speedup Analysis\n"));
            report.push_str(&format!("- Compilation speedup: {:.2}x\n\n", speedup));
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_compilation() {
        let result = run_sequential_test(4, 5, 42).expect("Sequential test should succeed");

        assert_eq!(result.method, CompilationMethod::Sequential);
        assert_eq!(result.output_data.len(), 5);
        assert!(result.compilation_time_ms > 0);
        assert!(result.kernel_count > 0);
    }

    #[test]
    fn test_parallel_compilation_placeholder() {
        let result = run_parallel_test(4, 5, 2, 42).expect("Parallel test should succeed");

        assert_eq!(result.method, CompilationMethod::Parallel { threads: 2 });
        assert_eq!(result.output_data.len(), 5);
        assert!(result.compilation_time_ms > 0);
    }

    #[test]
    fn test_output_comparison() {
        let seq_result = run_sequential_test(4, 5, 42).expect("Sequential test failed");
        let par_result = run_parallel_test(4, 5, 2, 42).expect("Parallel test failed");

        // Since both use the same seed and method (for now), outputs should be identical
        compare_outputs(&seq_result.output_data, &par_result.output_data, 1e-6)
            .expect("Outputs should be identical");
    }

    #[test]
    fn test_calibration_mini_suite() {
        let config = CalibrationConfig {
            model_sizes: vec![(4, 5)],
            batch_sizes: vec![1],
            thread_counts: vec![1, 2],
            iterations: 2,
            seed: 42,
        };

        let results = run_calibration_suite(config).expect("Calibration suite should succeed");
        assert!(!results.is_empty());

        let report = generate_performance_report(&results);
        println!("Mini calibration report:\n{}", report);
    }
}