use std::{
    io::{self, Write},
    time::Instant,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};

use clap::Parser;
use colored::Colorize;
use rayon::prelude::*;

// Thread counter for detailed reporting
static ACTIVE_THREADS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct CLIArgs {
    /// Number of parallel operations per step
    #[clap(long = "ops", default_value = "50")]
    operations: usize,
}

fn report_thread_activity(step_name: &str, phase: &str) {
    let active = ACTIVE_THREADS.load(Ordering::Relaxed);
    let total_threads = num_cpus::get();
    let current_thread = rayon::current_thread_index().unwrap_or(999);

    println!("🧵 {} | {} | Thread {} | Active: {}/{} threads",
             phase, step_name, current_thread, active, total_threads);
}

fn parallel_kernel_compilation_step(step_name: &str, operations: usize) -> Vec<String> {
    println!("\n🔧 {} - Starting parallel compilation", step_name.bright_cyan().bold());
    report_thread_activity(step_name, "START");

    let start_time = Instant::now();

    let results: Vec<String> = (0..operations).into_par_iter().map(|i| {
        // Track active threads
        ACTIVE_THREADS.fetch_add(1, Ordering::Relaxed);

        let thread_id = rayon::current_thread_index().unwrap_or(0);
        let kernel_name = format!("{}_kernel_{}", step_name.to_lowercase().replace(" ", "_"), i);

        // Simulate kernel compilation work
        let work_duration = 5 + (i % 15); // Variable work time 5-20ms
        thread::sleep(std::time::Duration::from_millis(work_duration as u64));

        if i < 5 || i % 10 == 0 {
            println!("  🔨 Thread {} compiling {}", thread_id, kernel_name);
        }

        ACTIVE_THREADS.fetch_sub(1, Ordering::Relaxed);
        kernel_name
    }).collect();

    let elapsed = start_time.elapsed();
    report_thread_activity(step_name, "END  ");
    println!("✅ {} completed: {:.2}ms ({} kernels, {:.2} kernels/ms)",
             step_name, elapsed.as_millis(), operations, operations as f64 / elapsed.as_millis() as f64);

    results
}

fn parallel_neural_network_simulation(operations: usize) {
    println!("\n🧠 {} - Parallel Neural Network Processing", "NEURAL".bright_green().bold());
    report_thread_activity("Neural Processing", "START");

    let start_time = Instant::now();

    // Simulate neural network layers processing in parallel
    let layer_results: Vec<Vec<f32>> = (0..operations).into_par_iter().map(|layer_id| {
        ACTIVE_THREADS.fetch_add(1, Ordering::Relaxed);

        let thread_id = rayon::current_thread_index().unwrap_or(0);

        // Simulate neural computation
        let computation_time = 10 + (layer_id % 20);
        thread::sleep(std::time::Duration::from_millis(computation_time as u64));

        // Generate some fake neural network output
        let output: Vec<f32> = (0..5).map(|i| (layer_id + i) as f32 * 0.1).collect();

        if layer_id < 5 {
            println!("  🧠 Thread {} processed layer {} -> [{:.1}, {:.1}, {:.1}, ...]",
                     thread_id, layer_id, output[0], output[1], output[2]);
        }

        ACTIVE_THREADS.fetch_sub(1, Ordering::Relaxed);
        output
    }).collect();

    let elapsed = start_time.elapsed();
    report_thread_activity("Neural Processing", "END  ");
    println!("✅ Neural processing completed: {:.2}ms ({} layers)",
             elapsed.as_millis(), layer_results.len());
}

fn parallel_memory_operations(operations: usize) {
    println!("\n💾 {} - Parallel Memory Operations", "MEMORY".bright_magenta().bold());
    report_thread_activity("Memory Operations", "START");

    let start_time = Instant::now();

    let _results: Vec<()> = (0..operations).into_par_iter().map(|op_id| {
        ACTIVE_THREADS.fetch_add(1, Ordering::Relaxed);

        let thread_id = rayon::current_thread_index().unwrap_or(0);

        // Simulate memory allocation/copying
        let mem_time = 3 + (op_id % 8);
        thread::sleep(std::time::Duration::from_millis(mem_time as u64));

        if op_id < 5 {
            println!("  💾 Thread {} memory operation {}", thread_id, op_id);
        }

        ACTIVE_THREADS.fetch_sub(1, Ordering::Relaxed);
    }).collect();

    let elapsed = start_time.elapsed();
    report_thread_activity("Memory Operations", "END  ");
    println!("✅ Memory operations completed: {:.2}ms", elapsed.as_millis());
}

fn main() {
    let cli_args = CLIArgs::parse();

    println!("{}", "🚀 QWEN PARALLEL-ONLY DEMONSTRATION".bright_cyan().bold());
    println!("CPU cores available: {}", num_cpus::get());
    println!("Rayon threads configured: {}", rayon::current_num_threads());
    println!("Operations per step: {}", cli_args.operations);
    println!("{}", "=".repeat(70));

    let total_start = Instant::now();

    // Step 1: Kernel Analysis and Optimization
    let _analysis_kernels = parallel_kernel_compilation_step("Kernel Analysis", cli_args.operations / 2);

    // Step 2: Memory Layout Optimization
    let _memory_kernels = parallel_kernel_compilation_step("Memory Optimization", cli_args.operations / 3);

    // Step 3: Compute Kernel Generation
    let _compute_kernels = parallel_kernel_compilation_step("Compute Generation", cli_args.operations);

    // Step 4: Neural Network Processing
    parallel_neural_network_simulation(cli_args.operations / 4);

    // Step 5: Memory Operations
    parallel_memory_operations(cli_args.operations / 2);

    // Step 6: Final Compilation and Linking
    let _final_kernels = parallel_kernel_compilation_step("Final Linking", cli_args.operations / 5);

    let total_elapsed = total_start.elapsed();

    println!("\n{}", "=".repeat(70));
    println!("📊 {} - Performance Summary", "RESULTS".bright_green().bold());
    println!("Total parallel execution time: {:.2}ms", total_elapsed.as_millis());
    println!("CPU cores used: {}/{}", rayon::current_num_threads(), num_cpus::get());
    println!("Parallelization efficiency: {:.1}%",
             (rayon::current_num_threads() as f64 / num_cpus::get() as f64) * 100.0);
    println!("✅ All operations completed using PARALLEL-ONLY computation!");

    // Show thread utilization summary
    println!("\n🧵 Thread Usage Summary:");
    println!("  - Each step shows START/END thread activity");
    println!("  - Active thread count tracked throughout execution");
    println!("  - Multiple threads working simultaneously on each operation");
    println!("  - {} total CPU threads available", num_cpus::get());
    println!("  - {} threads actively used by Rayon", rayon::current_num_threads());
}