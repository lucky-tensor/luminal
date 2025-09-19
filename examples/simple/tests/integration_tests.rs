use std::time::Instant;
use tempfile::tempdir;

mod compilation_calibration;
mod binary_cache_comparison;
mod parallel_validation;

use compilation_calibration::{CalibrationConfig, CompilationMethod, run_calibration_suite, generate_performance_report};
use binary_cache_comparison::CacheManager;

/// Main integration test that validates compilation correctness and performance
#[test]
fn test_compilation_methods_integration() {
    println!("🧪 Running compilation methods integration test...");

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let cache_manager = CacheManager::new(temp_dir.path());

    // Set up separate cache directories
    let (seq_cache_dir, par_cache_dir) = cache_manager
        .setup_test_directories()
        .expect("Failed to setup cache directories");

    println!("📁 Sequential cache: {:?}", seq_cache_dir);
    println!("📁 Parallel cache: {:?}", par_cache_dir);

    // Run calibration tests
    let config = CalibrationConfig {
        model_sizes: vec![(4, 5), (32, 64)],
        batch_sizes: vec![1],
        thread_counts: vec![1, 2, 4],
        iterations: 2,
        seed: 12345,
    };

    println!("⏳ Running calibration suite...");
    let results = run_calibration_suite(config)
        .expect("Calibration suite should succeed");

    println!("✅ Calibration complete with {} results", results.len());

    // Generate performance report
    let performance_report = generate_performance_report(&results);
    println!("📊 Performance Report:\n{}", performance_report);

    // Validate that we have both sequential and parallel results
    let has_sequential = results.iter().any(|r| r.method == CompilationMethod::Sequential);
    let has_parallel = results.iter().any(|r| matches!(r.method, CompilationMethod::Parallel { .. }));

    assert!(has_sequential, "Should have sequential results");
    assert!(has_parallel, "Should have parallel results");

    // Test output correctness - compare outputs between methods with same seed
    validate_output_correctness(&results);

    println!("✅ Integration test completed successfully");
}

/// Validate that different compilation methods produce identical outputs
fn validate_output_correctness(results: &[compilation_calibration::CompilationResult]) {
    use std::collections::HashMap;

    println!("🔍 Validating output correctness...");

    // Group results by test parameters (model size, batch size, seed)
    let mut grouped_results: HashMap<String, Vec<&compilation_calibration::CompilationResult>> = HashMap::new();

    for result in results {
        // Create a key that identifies the test case (excluding method and threads)
        let key = format!("out_len_{}_hash_{}", result.output_data.len(), result.graph_hash);
        grouped_results.entry(key).or_default().push(result);
    }

    let mut correctness_failures = 0;
    let mut total_comparisons = 0;

    for (test_case, case_results) in grouped_results {
        if case_results.len() < 2 {
            continue; // Need at least 2 results to compare
        }

        // Compare all pairs of results for this test case
        for i in 0..case_results.len() {
            for j in (i + 1)..case_results.len() {
                total_comparisons += 1;

                let result_a = case_results[i];
                let result_b = case_results[j];

                // Compare outputs with tolerance
                if let Err(e) = compare_outputs(&result_a.output_data, &result_b.output_data, 1e-6) {
                    correctness_failures += 1;
                    eprintln!("❌ Output mismatch in test case '{}' between {:?} and {:?}: {}",
                        test_case, result_a.method, result_b.method, e);
                }
            }
        }
    }

    println!("🎯 Correctness validation: {}/{} comparisons passed",
        total_comparisons - correctness_failures, total_comparisons);

    assert_eq!(correctness_failures, 0,
        "Found {} output correctness failures out of {} comparisons",
        correctness_failures, total_comparisons);
}

/// Compare two floating point vectors with tolerance
fn compare_outputs(a: &[f32], b: &[f32], tolerance: f32) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("Output length mismatch: {} vs {}", a.len(), b.len()));
    }

    let mut max_diff = 0.0f32;
    let mut diff_count = 0;

    for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        max_diff = max_diff.max(diff);

        if diff > tolerance {
            diff_count += 1;
            if diff_count <= 5 { // Only show first 5 mismatches
                eprintln!("    Index {}: {} vs {} (diff: {})", i, va, vb, diff);
            }
        }
    }

    if diff_count > 0 {
        Err(format!("{} values exceed tolerance (max diff: {:.2e})", diff_count, max_diff))
    } else {
        Ok(())
    }
}

/// Performance benchmark test
#[test]
fn test_compilation_performance() {
    println!("⚡ Running compilation performance benchmark...");

    let config = CalibrationConfig {
        model_sizes: vec![(128, 256), (256, 512)],
        batch_sizes: vec![1],
        thread_counts: vec![1, 2, 4, 8],
        iterations: 3,
        seed: 98765,
    };

    let start_time = Instant::now();
    let results = run_calibration_suite(config)
        .expect("Performance benchmark should succeed");
    let total_time = start_time.elapsed();

    println!("⏱️  Total benchmark time: {:.2}s", total_time.as_secs_f64());

    // Analyze performance scaling
    analyze_performance_scaling(&results);

    // Ensure we have reasonable performance (not too slow)
    assert!(total_time.as_secs() < 300, "Benchmark should complete within 5 minutes");
}

/// Analyze how performance scales with thread count
fn analyze_performance_scaling(results: &[compilation_calibration::CompilationResult]) {
    use std::collections::HashMap;

    println!("📈 Analyzing performance scaling...");

    let mut by_threads: HashMap<usize, Vec<u128>> = HashMap::new();

    for result in results {
        match result.method {
            CompilationMethod::Sequential => {
                by_threads.entry(1).or_default().push(result.compilation_time_ms);
            }
            CompilationMethod::Parallel { threads } => {
                by_threads.entry(threads).or_default().push(result.compilation_time_ms);
            }
        }
    }

    let mut thread_counts: Vec<usize> = by_threads.keys().copied().collect();
    thread_counts.sort();

    println!("Thread scaling analysis:");
    let baseline_avg = by_threads.get(&1)
        .map(|times| times.iter().sum::<u128>() as f64 / times.len() as f64)
        .unwrap_or(0.0);

    for thread_count in thread_counts {
        if let Some(times) = by_threads.get(&thread_count) {
            let avg_time = times.iter().sum::<u128>() as f64 / times.len() as f64;
            let speedup = if avg_time > 0.0 && baseline_avg > 0.0 {
                baseline_avg / avg_time
            } else {
                1.0
            };

            println!("  {} threads: {:.1}ms avg ({}x speedup)",
                thread_count, avg_time, speedup);
        }
    }
}

/// Test cache artifact comparison (placeholder for when parallel compilation generates actual cache)
#[test]
#[ignore] // Ignore until parallel compilation actually creates cache files
fn test_cache_artifact_comparison() {
    println!("🗂️  Testing cache artifact comparison...");

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let cache_manager = CacheManager::new(temp_dir.path());
    let (seq_dir, par_dir) = cache_manager.setup_test_directories()
        .expect("Failed to setup directories");

    // TODO: This test will be enabled once we have actual parallel compilation
    // that generates cache files. For now, we test the comparison infrastructure.

    // Simulate some cache artifacts for testing the comparison logic
    std::fs::write(seq_dir.join("kernel1.ptx"), b"mock ptx content").unwrap();
    std::fs::write(par_dir.join("kernel1.ptx"), b"mock ptx content").unwrap(); // Same content

    let seq_artifacts = cache_manager.extract_artifacts(&seq_dir).unwrap();
    let par_artifacts = cache_manager.extract_artifacts(&par_dir).unwrap();

    let comparison = cache_manager.compare_caches(&seq_artifacts, &par_artifacts);

    println!("Cache comparison result: {:?}", comparison.identical);

    // For identical mock files, should be identical
    assert!(comparison.identical, "Identical cache files should pass comparison");

    let report = cache_manager.generate_comparison_report(&comparison);
    println!("Comparison report:\n{}", report);
}

/// Minimal smoke test to ensure basic functionality works
#[test]
fn test_basic_compilation() {
    use luminal::prelude::*;
    use luminal_nn::Linear;
    use rand::{SeedableRng, Rng, rngs::StdRng};

    println!("🔥 Running basic compilation smoke test...");

    let mut rng = StdRng::seed_from_u64(42);
    let mut cx = Graph::new();

    // Create a simple model
    let model = Linear::new(8, 4, false, &mut cx);
    let weight_data: Vec<f32> = (0..8*4).map(|_| rng.random_range(-1.0..1.0)).collect();
    model.weight.set(weight_data);

    let input = cx.tensor(8).set(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let mut output = model.forward(input).retrieve();

    // Compile and execute
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
    let compile_time = compile_start.elapsed();

    let exec_start = Instant::now();
    cx.execute();
    let exec_time = exec_start.elapsed();

    let result = output.data();

    println!("✅ Compilation: {:.2}ms, Execution: {:.2}ms",
        compile_time.as_millis(), exec_time.as_millis());
    println!("📊 Output shape: {}, first few values: {:?}",
        result.len(), &result[..result.len().min(4)]);

    // Basic sanity checks
    assert_eq!(result.len(), 4, "Output should have 4 elements");
    assert!(compile_time.as_millis() < 10000, "Compilation should complete within 10s");
    assert!(exec_time.as_millis() < 1000, "Execution should complete within 1s");
    assert!(result.iter().all(|&x| x.is_finite()), "All outputs should be finite");
}

/// **MAIN TEST**: Validates that parallel kernel compilation matches sequential compilation
#[test]
fn test_parallel_compilation_correctness() {
    use parallel_validation::test_parallel_vs_sequential_compilation;

    println!("🔬 Testing parallel vs sequential compilation correctness...");

    // Test with different thread counts
    let thread_counts = vec![1, 2, 4];
    let tolerance = 1e-6; // Very strict tolerance for exact match

    for threads in thread_counts {
        println!("\n📊 Testing with {} threads:", threads);

        let result = test_parallel_vs_sequential_compilation(threads, tolerance)
            .expect(&format!("Parallel vs sequential test should succeed with {} threads", threads));

        // Validate outputs are identical
        assert!(result.outputs_identical,
            "Thread count {}: Outputs must be identical between parallel and sequential compilation. Error: {:?}",
            threads, result.error_message);

        // Log performance results
        println!("   ⏱️  Sequential: {}ms", result.sequential_time_ms);
        println!("   ⏱️  Parallel: {}ms", result.parallel_time_ms);
        println!("   🚀 Speedup: {:.2}x", result.speedup);
        println!("   ✅ Outputs identical: {}", result.outputs_identical);
        println!("   📦 Cache identical: {}", result.cache_artifacts_identical);

        // Basic sanity checks
        assert!(result.sequential_time_ms > 0, "Sequential compilation should take some time");
        assert!(result.parallel_time_ms > 0, "Parallel compilation should take some time");
        assert!(result.speedup > 0.0, "Speedup should be positive");
    }

    println!("\n🎉 All parallel compilation correctness tests passed!");
}