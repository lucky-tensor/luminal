use itertools::Itertools;
use luminal::prelude::*;
use luminal_nn::Linear;
use rand::{rng, Rng};
use rayon::prelude::*;
use clap::Parser;
use colored::Colorize;
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Use parallel processing
    #[clap(long = "parallel", default_value = "false")]
    parallel: bool,

    /// Number of operations to run in parallel
    #[clap(long = "operations", default_value = "100")]
    operations: usize,

    /// Size of input tensors
    #[clap(long = "input-size", default_value = "1000")]
    input_size: usize,

    /// Size of output tensors
    #[clap(long = "output-size", default_value = "500")]
    output_size: usize,

    /// Show detailed timing information
    #[clap(long = "timing", default_value = "false")]
    timing: bool,
}

fn create_model_and_data(input_size: usize, output_size: usize) -> (Graph, Linear, Vec<f32>, GraphTensor) {
    let mut rng = rng();
    let weight = (0..input_size * output_size).map(|_| rng.random()).collect_vec();
    let input_data = (0..input_size).map(|i| i as f32 * 0.1).collect_vec();

    // Create a new graph
    let mut cx = Graph::new();

    // Create linear layer
    let model = Linear::new(input_size, output_size, false, &mut cx);
    model.weight.set(weight);

    // Make an input tensor
    let input_tensor = cx.tensor(input_size).set(input_data.clone());

    (cx, model, input_data, input_tensor)
}

fn run_standard_mode(args: &Args) -> f64 {
    println!("🔧 Running in STANDARD mode");
    println!("Operations: {}, Input size: {}, Output size: {}", args.operations, args.input_size, args.output_size);

    let start_time = Instant::now();

    for i in 0..args.operations {
        let (mut cx, model, _input_data, input_tensor) = create_model_and_data(args.input_size, args.output_size);

        // Feed tensor through model
        let mut output = model.forward(input_tensor).retrieve();

        // Compile the graph for optimal execution
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

        // Execute the graph
        cx.execute();

        if args.timing && i < 5 {
            println!("Operation {}: Output sample: [{:.3}, {:.3}, {:.3}, ...]",
                     i + 1,
                     output.data()[0],
                     output.data()[1],
                     output.data().get(2).unwrap_or(&0.0));
        }
    }

    let elapsed = start_time.elapsed();
    let elapsed_ms = elapsed.as_millis() as f64;

    println!("⏱️  Standard execution: {:.2}ms", elapsed_ms);
    println!("📊 Average per operation: {:.2}ms", elapsed_ms / args.operations as f64);

    elapsed_ms
}

fn run_parallel_mode(args: &Args) -> f64 {
    let num_threads = num_cpus::get();
    println!("🚀 Running in PARALLEL mode");
    println!("Operations: {}, Input size: {}, Output size: {}", args.operations, args.input_size, args.output_size);
    println!("Using {} CPU threads", num_threads);

    // Configure rayon thread pool
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let start_time = Instant::now();

    let results: Vec<Vec<f32>> = thread_pool.install(|| {
        (0..args.operations).into_par_iter().map(|i| {
            let thread_id = rayon::current_thread_index().unwrap_or(0);

            if args.timing && i < 10 {
                println!("Thread {} processing operation {}", thread_id, i + 1);
            }

            let (mut cx, model, _input_data, input_tensor) = create_model_and_data(args.input_size, args.output_size);

            // Feed tensor through model
            let mut output = model.forward(input_tensor).retrieve();

            // Compile the graph for optimal execution
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

            // Execute the graph
            cx.execute();

            output.data()
        }).collect()
    });

    let elapsed = start_time.elapsed();
    let elapsed_ms = elapsed.as_millis() as f64;

    println!("⚡ Parallel execution: {:.2}ms", elapsed_ms);
    println!("📊 Average per operation: {:.2}ms", elapsed_ms / args.operations as f64);
    println!("🧵 Thread utilization: {} threads used", num_threads);

    if args.timing {
        println!("First few results:");
        for (i, result) in results.iter().take(3).enumerate() {
            println!("  Result {}: [{:.3}, {:.3}, {:.3}, ...]",
                     i + 1, result[0], result[1], result.get(2).unwrap_or(&0.0));
        }
    }

    elapsed_ms
}

fn run_comparison(args: &Args) {
    println!("\n{}", "=== PERFORMANCE COMPARISON ===".bright_cyan().bold());

    // Run standard mode
    let standard_time = run_standard_mode(args);

    println!(); // Add spacing

    // Run parallel mode
    let parallel_time = run_parallel_mode(args);

    // Calculate speedup
    let speedup = standard_time / parallel_time;
    let efficiency = speedup / num_cpus::get() as f64 * 100.0;

    println!("\n{}", "=== RESULTS ===".bright_green().bold());
    println!("Standard time: {:.2}ms", standard_time);
    println!("Parallel time: {:.2}ms", parallel_time);
    println!("Speedup: {:.2}x", speedup);
    println!("Parallel efficiency: {:.1}%", efficiency);

    if speedup > 1.0 {
        println!("{}", format!("🎉 Parallel execution was {:.1}% faster!", (speedup - 1.0) * 100.0).bright_green());
    } else {
        println!("{}", format!("⚠️  Parallel execution was {:.1}% slower (overhead)", (1.0 - speedup) * 100.0).yellow());
    }

    println!("{}", "================================".bright_cyan());
}

fn main() {
    let args = Args::parse();

    println!("{}", "🔬 Simple Parallel Neural Network Example".bright_cyan().bold());
    println!("CPU cores available: {}", num_cpus::get());

    if args.parallel && !args.timing {
        // Just run parallel mode
        run_parallel_mode(&args);
    } else if !args.parallel && !args.timing {
        // Just run standard mode
        run_standard_mode(&args);
    } else {
        // Run comparison or detailed timing
        run_comparison(&args);
    }

    println!("\n✅ Done!");
}