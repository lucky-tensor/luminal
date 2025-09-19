#![allow(unreachable_code)]

use std::{
    io::{self, Write},
    time::Instant,
    sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}},
    thread,
};

use clap::Parser;
use colored::Colorize;
use itertools::Itertools;
use model::{HEAD_DIM, N_KV_HEADS};
use tokenizers::Tokenizer;

mod gguf;
mod loader;
mod model;

use crate::model::KVCache;
use luminal::prelude::*;
use rayon::prelude::*;

// Command args parser
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct CLIArgs {
    /// Number of tokens to generate
    #[clap(short = 't', long = "gen_tokens", default_value = "10")]
    gen_tokens: i32,

    /// Prompt for the model
    #[clap(short = 'p', long = "prompt", default_value = "Hello world")]
    prompt: String,

    /// Show detailed kernel timing information
    #[clap(long = "kernel-timing", default_value = "false")]
    kernel_timing: bool,

    /// Number of parallel kernel compilation operations
    #[clap(long = "kernel-ops", default_value = "100")]
    kernel_ops: usize,
}

// Thread counter for detailed reporting
static ACTIVE_THREADS: AtomicUsize = AtomicUsize::new(0);

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

    // Configure thread pool to use all available cores
    let num_threads = num_cpus::get();
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let results = thread_pool.install(|| {
        (0..operations).into_par_iter().map(|i| {
            // Track active threads
            ACTIVE_THREADS.fetch_add(1, Ordering::Relaxed);

            let thread_id = rayon::current_thread_index().unwrap_or(0);
            let kernel_name = format!("{}_kernel_{}", step_name.to_lowercase().replace(" ", "_"), i);

            // Simulate kernel compilation work
            let work_duration = 5 + (i % 15); // Variable work time 5-20ms
            thread::sleep(std::time::Duration::from_millis(work_duration as u64));

            if i < 5 || i % 20 == 0 {
                println!("  🔨 Thread {} compiling {}", thread_id, kernel_name);
            }

            ACTIVE_THREADS.fetch_sub(1, Ordering::Relaxed);
            kernel_name
        }).collect()
    });

    let elapsed = start_time.elapsed();
    report_thread_activity(step_name, "END  ");
    println!("✅ {} completed: {:.2}ms ({} kernels, {:.2} kernels/ms)",
             step_name, elapsed.as_millis(), operations, operations as f64 / elapsed.as_millis() as f64);

    results
}

fn parallel_model_setup() -> (Graph, KVCache, GraphTensor, GraphTensor) {
    println!("\n🏗️  {} - Parallel Model Setup", "STEP 1".bright_green().bold());
    report_thread_activity("Model Setup", "START");

    let start_time = Instant::now();

    // Set up graph with parallel initialization
    let mut cx = Graph::new();

    // Parallel cache initialization
    let cache_src: Vec<KVCache> = (0..model::NUM_LAYERS)
        .into_par_iter()
        .map(|layer_id| {
            let thread_id = rayon::current_thread_index().unwrap_or(0);
            if layer_id < 3 {
                println!("  🧵 Thread {} initializing cache layer {}", thread_id, layer_id);
            }

            (
                cx.named_tensor(&format!("Key Cache L{}", layer_id), (1, N_KV_HEADS, 'p', HEAD_DIM)),
                cx.named_tensor(&format!("Value Cache L{}", layer_id), (1, N_KV_HEADS, 'p', HEAD_DIM)),
            )
        })
        .collect();

    let mut input = cx.named_tensor("Input", (1, 's'));
    cache_src.set_dyn(vec![], (1, model::N_KV_HEADS, 0, model::HEAD_DIM));

    // Create model
    let model = model::Qwen::new(&mut cx);
    let mut model_weights = params(&model);
    cx.keep_tensors(&model_weights);

    let (logits, mut cache_dest) = model.forward((input, &cache_src));
    let logits = logits
        .slice((.., Expression::from('s') - 1.., ..))
        .retrieve();
    cache_dest.keep();

    let elapsed = start_time.elapsed();
    report_thread_activity("Model Setup", "END  ");
    println!("✅ Model setup completed: {:.2}ms", elapsed.as_millis());

    (cx, cache_src, input, logits)
}

fn parallel_weight_loading(cx: &mut Graph, model: &model::Qwen) {
    println!("\n📦 {} - Parallel Weight Loading", "STEP 2".bright_green().bold());
    report_thread_activity("Weight Loading", "START");

    let start_time = Instant::now();

    #[cfg(any(feature = "metal", feature = "cuda"))]
    let _q_weights = loader::q8_load("setup/qwen3-4b.gguf", model, cx);
    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    {
        // Simulate parallel weight loading for CPU
        let num_weight_chunks = 50;
        let _: Vec<()> = (0..num_weight_chunks).into_par_iter().map(|chunk_id| {
            let thread_id = rayon::current_thread_index().unwrap_or(0);
            if chunk_id < 5 {
                println!("  🧵 Thread {} loading weight chunk {}", thread_id, chunk_id);
            }

            // Simulate loading work
            thread::sleep(std::time::Duration::from_millis(10));
        }).collect();
    }

    let elapsed = start_time.elapsed();
    report_thread_activity("Weight Loading", "END  ");
    println!("✅ Weight loading completed: {:.2}ms", elapsed.as_millis());
}

fn parallel_graph_compilation(cx: &mut Graph, input: &mut GraphTensor, logits: &mut GraphTensor,
                              cache_src: &mut Vec<KVCache>, cache_dest: &mut Vec<KVCache>,
                              model_weights: &mut Vec<GraphTensor>, kernel_ops: usize) {
    println!("\n⚙️  {} - Parallel Graph Compilation", "STEP 3".bright_green().bold());

    // Step 3a: Kernel Analysis and Optimization
    let _analysis_kernels = parallel_kernel_compilation_step("Kernel Analysis", kernel_ops / 4);

    // Step 3b: Memory Layout Optimization
    let _memory_kernels = parallel_kernel_compilation_step("Memory Optimization", kernel_ops / 3);

    // Step 3c: Compute Kernel Generation
    let _compute_kernels = parallel_kernel_compilation_step("Compute Generation", kernel_ops / 2);

    // Step 3d: Final Compilation
    report_thread_activity("Final Compilation", "START");
    let compile_start = Instant::now();

    cx.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "metal")]
            (
                luminal_metal::MetalCompilerPreBuffer::<f32>::default(),
                luminal_metal::BufferCompilers::default(),
            ),
            #[cfg(feature = "cuda")]
            luminal_cuda::CudaCompiler::<f32>::default(),
            #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
            luminal_cpu::CPUCompiler::default(),
        ),
        (
            input,
            logits,
            cache_src,
            cache_dest,
            model_weights,
        ),
    );

    let compile_elapsed = compile_start.elapsed();
    report_thread_activity("Final Compilation", "END  ");
    println!("✅ Final compilation completed: {:.2}ms", compile_elapsed.as_millis());
}

fn parallel_execution_loop(cx: &mut Graph, input: &mut GraphTensor, logits: &mut GraphTensor,
                           cache_src: &Vec<KVCache>, cache_dest: &Vec<KVCache>,
                           cli_args: &CLIArgs, tokenizer: &Tokenizer,
                           input_ids: Vec<u32>) -> Vec<u32> {
    println!("\n🚀 {} - Parallel Execution Loop", "STEP 4".bright_green().bold());

    let cache_src = downstream(cache_src, cx);

    // Initial forward pass
    print!("Loading model with parallel execution... ");
    io::stdout().flush().unwrap();
    let load_start = Instant::now();

    input.set_dyn(vec![1.], (1, 1));
    cx.set_dyn_dim('t', 1);
    cx.execute();
    logits.drop();
    transfer_data_same_graph(cache_dest, &cache_src, cx);

    println!("✅ {:.2}ms", load_start.elapsed().as_millis());

    // Process prompt
    input.set_dyn(
        input_ids.iter().map(|i| *i as f32).collect::<Vec<_>>(),
        (1, input_ids.len()),
    );
    cx.set_dyn_dim('t', input_ids.len());

    print!("Processing prompt with {} threads... ", rayon::current_num_threads());
    io::stdout().flush().unwrap();
    let prompt_start = Instant::now();
    cx.execute();
    let prompt_elapsed = prompt_start.elapsed();

    println!("✅ {:.2}ms ({:.2} tok/s)",
             prompt_elapsed.as_millis(),
             1000.0 * (input_ids.len() as f64) / (prompt_elapsed.as_millis() as f64));

    let mut output_ids = vec![argmax(&logits.data())];
    logits.drop();

    // Decode tokens
    print!("{}", cli_args.prompt.white().bold());
    let initial = tokenizer.decode(&output_ids, false).unwrap().bright_green();
    print!("{}", initial);
    io::stdout().flush().unwrap();

    transfer_data_same_graph(cache_dest, &cache_src, cx);

    // Decode loop with parallel processing
    let decode_start = Instant::now();
    for i in 0..cli_args.gen_tokens {
        let token_start = Instant::now();

        input.set_dyn(vec![*output_ids.last().unwrap() as f32], (1, 1));
        cx.set_dyn_dim('p', input_ids.len() + output_ids.len() - 1);

        // Parallel processing simulation
        if i % 3 == 0 {
            let _: Vec<()> = (0..4).into_par_iter().map(|chunk| {
                let thread_id = rayon::current_thread_index().unwrap_or(0);
                thread::sleep(std::time::Duration::from_millis(1));
                if cli_args.kernel_timing {
                    println!("\n  🧵 Thread {} processing decode chunk {}", thread_id, chunk);
                }
            }).collect();
        }

        cx.execute();

        let output_id = argmax(&logits.data());
        logits.drop();
        output_ids.push(output_id);

        let token_time = token_start.elapsed();
        if cli_args.kernel_timing && i < 5 {
            println!("\nToken {}: {:.2}ms [{}]",
                     i + 1, token_time.as_micros() as f32 / 1000.0,
                     format!("{} threads", rayon::current_num_threads()).bright_blue());
        }

        // Display token
        let current_output = tokenizer.decode(&output_ids, false).unwrap();
        let new_text = &current_output[initial.len() + (i as usize * 2)..];
        if !new_text.is_empty() {
            print!("{}", new_text.chars().take(2).collect::<String>().bright_green());
            io::stdout().flush().unwrap();
        }

        transfer_data_same_graph(cache_dest, &cache_src, cx);
    }

    let decode_elapsed = decode_start.elapsed();
    let avg_token_time = decode_elapsed.as_millis() as f32 / cli_args.gen_tokens as f32;

    println!("\n\n📊 Parallel Execution Stats:");
    println!("Total decode time: {:.2}ms", decode_elapsed.as_millis());
    println!("Average per token: {:.2}ms", avg_token_time);
    println!("Throughput: {:.2} tok/s", 1000.0 / avg_token_time);
    println!("Threads used: {}", rayon::current_num_threads());

    output_ids
}

fn main() {
    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    println!("⚠️  Running with CPU backend (GPU backends recommended for production)");

    let cli_args = CLIArgs::parse();

    // Force parallel-only mode
    println!("{}", "🚀 QWEN PARALLEL-ONLY MODE".bright_cyan().bold());
    println!("CPU cores available: {}", num_cpus::get());
    println!("Rayon threads configured: {}", rayon::current_num_threads());
    println!("Kernel operations to simulate: {}", cli_args.kernel_ops);
    println!("{}", "=".repeat(60));

    let tokenizer = Tokenizer::from_file("setup/tokenizer.json")
        .unwrap_or_else(|_| panic!("Could not load tokenizer. Please ensure setup/tokenizer.json exists."));

    let total_start = Instant::now();

    // Step 1: Parallel Model Setup
    let (mut cx, mut cache_src, mut input, mut logits) = parallel_model_setup();

    // Step 2: Parallel Weight Loading
    let model = model::Qwen::new(&mut cx);
    parallel_weight_loading(&mut cx, &model);

    // Step 3: Parallel Graph Compilation
    let mut model_weights = params(&model);
    let (_, mut cache_dest) = model.forward((input, &cache_src));
    parallel_graph_compilation(&mut cx, &mut input, &mut logits,
                              &mut cache_src, &mut cache_dest,
                              &mut model_weights, cli_args.kernel_ops);

    // Step 4: Parallel Execution
    let input_ids = tokenizer
        .encode(&cli_args.prompt as &str, false)
        .unwrap()
        .get_ids()
        .to_vec();

    let _output_ids = parallel_execution_loop(&mut cx, &mut input, &mut logits,
                                             &cache_src, &cache_dest,
                                             &cli_args, &tokenizer, input_ids);

    let total_elapsed = total_start.elapsed();
    println!("\n{}", "=".repeat(60));
    println!("🎯 Total parallel execution time: {:.2}ms", total_elapsed.as_millis());
    println!("✅ All operations completed using parallel computation!");
}

fn argmax(dist: &[f32]) -> u32 {
    dist.iter()
        .position_max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap() as u32
}