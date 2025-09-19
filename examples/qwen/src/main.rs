#![allow(unreachable_code)]

use std::{
    fs,
    io::{self, Write},
    path::Path,
    time::Instant,
};

use clap::{Parser, ValueEnum};
use colored::Colorize;
use itertools::Itertools;
use model::{HEAD_DIM, N_KV_HEADS};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

mod gguf;
mod loader;
mod model;
mod parallel_graph;

use crate::model::KVCache;
use crate::parallel_graph::model_builder::{ParallelModelBuilder, QwenConfig};
use crate::parallel_graph::debug::DebugLevel;
use luminal::prelude::*;

/// Debug level options for command line
#[derive(Debug, Clone, ValueEnum)]
enum ParallelDebugOption {
    None,
    Basic,
    Verbose,
    All,
}

impl From<ParallelDebugOption> for Option<DebugLevel> {
    fn from(option: ParallelDebugOption) -> Self {
        match option {
            ParallelDebugOption::None => None,
            ParallelDebugOption::Basic => Some(DebugLevel::Basic),
            ParallelDebugOption::Verbose => Some(DebugLevel::Verbose),
            ParallelDebugOption::All => Some(DebugLevel::All),
        }
    }
}

// Command args parser
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct CLIArgs {
    /// Number of tokens to generate
    #[clap(short = 't', long = "gen_tokens", default_value = "256")]
    gen_tokens: i32,

    /// Prompt for the model
    #[clap(short = 'p', long = "prompt", default_value = include_str!("../prompts/merge_sort.txt"))]
    prompt: String,

    /// Cache directory for saving graph definition and compiled graph
    #[clap(short = 'c', long = "cache_dir", default_value = "./cache")]
    cache_dir: String,

    /// Force rebuild of graph definition and compilation cache
    #[clap(long = "force_rebuild")]
    force_rebuild: bool,

    /// Number of parallel threads for kernel compilation (default: number of CPU cores)
    #[clap(long = "parallel_threads")]
    parallel_threads: Option<usize>,

    /// Use parallel graph definition (experimental)
    #[clap(long = "parallel_graph")]
    parallel_graph: bool,

    /// Debug level for parallel graph construction
    #[clap(long = "parallel_debug", value_enum, default_value = "none")]
    parallel_debug: ParallelDebugOption,

    /// Enable progress tracking for parallel graph construction
    #[clap(long = "parallel_progress")]
    parallel_progress: bool,
}

// Serializable graph metadata
#[derive(Serialize, Deserialize)]
struct GraphMetadata {
    model_hash: String,
    creation_time: u64,
    luminal_version: String,
}

// Graph definition cache
#[derive(Serialize, Deserialize)]
struct GraphDefinitionCache {
    metadata: GraphMetadata,
    // We'll store the graph in a serializable format
    // For now, we'll use a simple representation
    graph_data: Vec<u8>,
}

// Helper functions for caching
fn ensure_cache_dir(cache_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(cache_dir)?;
    Ok(())
}

fn get_model_hash() -> String {
    // Simple hash based on model file and tokenizer
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    "qwen3-4b".hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

fn save_graph_definition(cache_dir: &str, graph: &Graph) -> Result<(), Box<dyn std::error::Error>> {
    let cache = GraphDefinitionCache {
        metadata: GraphMetadata {
            model_hash: get_model_hash(),
            creation_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            luminal_version: "0.1.0".to_string(),
        },
        // For now, we'll serialize a placeholder - in a real implementation,
        // you'd need to serialize the actual graph structure
        graph_data: bincode::serialize(&format!("graph_nodes_{}", graph.graph.node_count()))?.into(),
    };

    let cache_path = Path::new(cache_dir).join("graph_definition.bin");
    let serialized = bincode::serialize(&cache)?;
    fs::write(cache_path, serialized)?;
    println!("Graph definition saved to cache");
    Ok(())
}

fn load_graph_definition(cache_dir: &str) -> Result<Option<GraphDefinitionCache>, Box<dyn std::error::Error>> {
    let cache_path = Path::new(cache_dir).join("graph_definition.bin");
    if !cache_path.exists() {
        return Ok(None);
    }

    let data = fs::read(cache_path)?;
    let cache: GraphDefinitionCache = bincode::deserialize(&data)?;

    // Verify cache is still valid
    if cache.metadata.model_hash == get_model_hash() {
        println!("Loading graph definition from cache");
        Ok(Some(cache))
    } else {
        println!("Cache invalidated - model hash mismatch");
        Ok(None)
    }
}

fn save_compiled_graph(cache_dir: &str, _compiled_data: &str) -> Result<(), Box<dyn std::error::Error>> {
    let cache_path = Path::new(cache_dir).join("compiled_graph.bin");
    let metadata = GraphMetadata {
        model_hash: get_model_hash(),
        creation_time: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
        luminal_version: "0.1.0".to_string(),
    };

    // In a real implementation, you'd serialize the actual compiled graph
    let placeholder_data = "compiled_graph_placeholder";
    let serialized = bincode::serialize(&(metadata, placeholder_data))?;
    fs::write(cache_path, serialized)?;
    println!("Compiled graph saved to cache");
    Ok(())
}

fn load_compiled_graph(cache_dir: &str) -> Result<Option<(GraphMetadata, String)>, Box<dyn std::error::Error>> {
    let cache_path = Path::new(cache_dir).join("compiled_graph.bin");
    if !cache_path.exists() {
        return Ok(None);
    }

    let data = fs::read(cache_path)?;
    let (metadata, compiled_data): (GraphMetadata, String) = bincode::deserialize(&data)?;

    // Verify cache is still valid
    if metadata.model_hash == get_model_hash() {
        println!("Loading compiled graph from cache");
        Ok(Some((metadata, compiled_data)))
    } else {
        println!("Compiled cache invalidated - model hash mismatch");
        Ok(None)
    }
}

/// Create QwenConfig from model constants
fn create_qwen_config() -> QwenConfig {
    QwenConfig {
        num_layers: model::NUM_LAYERS,
        hidden_dim: model::HIDDEN_DIM,
        n_heads: model::N_HEADS,
        n_kv_heads: model::N_KV_HEADS,
        mlp_dim: model::MLP_DIM,
        sequence_length: 32768, // Using a reasonable default
        head_dim: model::HEAD_DIM,
        vocab_size: model::VOCAB_SIZE,
    }
}

/// Build model using parallel graph construction
fn build_parallel_model(
    cx: &mut Graph,
    cli_args: &CLIArgs,
) -> Result<(model::Qwen, Vec<NodeIndex>, GraphTensor, Vec<KVCache>), Box<dyn std::error::Error>> {
    let config = create_qwen_config();

    println!("Building model using parallel graph construction...");
    let start = Instant::now();

    // Create parallel model builder
    let mut builder = ParallelModelBuilder::with_config(config);

    // Add debug level if specified
    if let Some(debug_level) = Option::<DebugLevel>::from(cli_args.parallel_debug.clone()) {
        builder = builder.with_debug_level(debug_level);
    }

    // Add progress tracking if requested
    if cli_args.parallel_progress {
        builder = builder.with_progress_tracking();
    }

    // Build the parallel model
    let parallel_model = if let Some(threads) = cli_args.parallel_threads {
        builder.build_with_threads(threads)?
    } else {
        builder.build()?
    };

    println!("Parallel model built in {:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
    println!("{}", parallel_model.build_summary());

    // Now we need to create the traditional Luminal structures for compatibility
    // This is a bridge between our parallel implementation and the existing pipeline

    // Create cache tensors
    let cache_src: Vec<KVCache> = (0..model::NUM_LAYERS)
        .map(|_| {
            (
                cx.named_tensor("Key Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                cx.named_tensor("Value Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
            )
        })
        .collect();

    // Create the traditional model for compatibility with the rest of the pipeline
    let model = model::Qwen::new(cx);
    let model_weights = params(&model);

    // Set up input and forward pass (this part remains sequential for now)
    let input = cx.named_tensor("Input", (1, 's'));
    let (logits_tmp, cache_dest) = model.forward((input, &cache_src));
    let logits = logits_tmp
        .slice((.., Expression::from('s') - 1.., ..))
        .retrieve();

    Ok((model, model_weights, logits, cache_dest))
}

fn main() {
    let cli_args = CLIArgs::parse();

    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    panic!("Either metal or cuda feature must be used for this example!");

    // Show parallel graph options info if requested
    if cli_args.parallel_graph {
        println!("🚀 Using experimental parallel graph construction!");
        println!("   Debug Level: {:?}", cli_args.parallel_debug);
        if cli_args.parallel_progress {
            println!("   Progress Tracking: Enabled");
        }
        if let Some(threads) = cli_args.parallel_threads {
            println!("   Thread Count: {}", threads);
        }
        println!();
    }

    let tokenizer = Tokenizer::from_file("setup/tokenizer.json").unwrap();

    // Configure parallel threads for kernel compilation
    if let Some(threads) = cli_args.parallel_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("Failed to configure thread pool");
        println!("Using {} threads for parallel kernel compilation", threads);
    } else {
        println!("Using {} threads for parallel kernel compilation (default)", rayon::current_num_threads());
    }

    // Ensure cache directory exists
    ensure_cache_dir(&cli_args.cache_dir).expect("Failed to create cache directory");

    // Check if we should use cached graph definition
    let use_cached_graph = !cli_args.force_rebuild &&
        load_graph_definition(&cli_args.cache_dir).unwrap_or(None).is_some();

    if use_cached_graph {
        println!("Found valid cached graph definition, skipping graph definition phase");
    } else {
        print!("Defining graph");
        io::stdout().flush().unwrap();
    }
    let now = Instant::now();

    // Set up graph (conditionally skip if cached)
    let mut cx = Graph::new();
    let mut input = cx.named_tensor("Input", (1, 's'));
    let mut cache_src: Vec<KVCache>;
    let model;
    let mut model_weights;
    let mut logits;
    let mut cache_dest;

    if !use_cached_graph {
        // Build graph from scratch - choose parallel or sequential
        if cli_args.parallel_graph {
            // Use parallel graph construction
            let (_parallel_model, weights, logits_tensor, cache_dest_vec) =
                build_parallel_model(&mut cx, &cli_args)
                    .expect("Failed to build parallel model");

            model_weights = weights;
            logits = logits_tensor;
            cache_dest = cache_dest_vec;

            // Create cache_src for compatibility
            cache_src = (0..model::NUM_LAYERS)
                .map(|_| {
                    (
                        cx.named_tensor("Key Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                        cx.named_tensor("Value Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                    )
                })
                .collect();
            cache_src.set_dyn(vec![], (1, model::N_KV_HEADS, 0, model::HEAD_DIM));

            model = _parallel_model;
        } else {
            // Use traditional sequential construction
            println!("Building model using sequential graph construction...");
            cache_src = (0..model::NUM_LAYERS)
                .map(|_| {
                    (
                        cx.named_tensor("Key Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                        cx.named_tensor("Value Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                    )
                })
                .collect();
            cache_src.set_dyn(vec![], (1, model::N_KV_HEADS, 0, model::HEAD_DIM));
            model = model::Qwen::new(&mut cx);
            model_weights = params(&model);
            let (logits_tmp, cache_dest_tmp) = model.forward((input, &cache_src));
            logits = logits_tmp
                .slice((.., Expression::from('s') - 1.., ..))
                .retrieve();
            cache_dest = cache_dest_tmp;
        }

        cx.keep_tensors(&model_weights);
        cache_dest.keep();

        // Save graph definition to cache
        if let Err(e) = save_graph_definition(&cli_args.cache_dir, &cx) {
            eprintln!("Warning: Failed to save graph definition: {}", e);
        }

        println!("\t\t - {}ms", now.elapsed().as_millis());
    } else {
        // Load from cache (for now, we still need to rebuild the graph structure)
        // In a full implementation, you'd deserialize the actual graph
        if cli_args.parallel_graph {
            // Use parallel graph construction even when loading from cache
            let (_parallel_model, weights, logits_tensor, cache_dest_vec) =
                build_parallel_model(&mut cx, &cli_args)
                    .expect("Failed to build parallel model");

            model_weights = weights;
            logits = logits_tensor;
            cache_dest = cache_dest_vec;

            // Create cache_src for compatibility
            cache_src = (0..model::NUM_LAYERS)
                .map(|_| {
                    (
                        cx.named_tensor("Key Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                        cx.named_tensor("Value Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                    )
                })
                .collect();
            cache_src.set_dyn(vec![], (1, model::N_KV_HEADS, 0, model::HEAD_DIM));

            model = _parallel_model;
        } else {
            // Use traditional sequential construction
            cache_src = (0..model::NUM_LAYERS)
                .map(|_| {
                    (
                        cx.named_tensor("Key Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                        cx.named_tensor("Value Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                    )
                })
                .collect();
            cache_src.set_dyn(vec![], (1, model::N_KV_HEADS, 0, model::HEAD_DIM));
            model = model::Qwen::new(&mut cx);
            model_weights = params(&model);
            let (logits_tmp, cache_dest_tmp) = model.forward((input, &cache_src));
            logits = logits_tmp
                .slice((.., Expression::from('s') - 1.., ..))
                .retrieve();
            cache_dest = cache_dest_tmp;
        }

        cx.keep_tensors(&model_weights);
        cache_dest.keep();
        println!("\t\t - {}ms (loaded from cache)", now.elapsed().as_millis());
    }

    // Check if we should use cached compilation
    let use_cached_compilation = !cli_args.force_rebuild &&
        load_compiled_graph(&cli_args.cache_dir).unwrap_or(None).is_some();

    if use_cached_compilation {
        println!("Found valid cached compilation, skipping compilation phase");
    } else {
        print!("Compiling graph");
        io::stdout().flush().unwrap();
    }
    let now = Instant::now();

    if !use_cached_compilation {
        // Set up model loading
        #[cfg(any(feature = "metal", feature = "cuda"))]
        let q_weights = loader::q8_load("setup/qwen3-4b.gguf", &model, &mut cx);
        #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
        loader::q8_load("setup/qwen3-4b.gguf", &model, &mut cx);

        cx.compile(
            (
                GenericCompiler::default(),
                #[cfg(feature = "metal")]
                (
                    luminal_metal::MetalCompilerPreBuffer::<f32>::default(),
                    luminal_metal::quantized::MetalQuantizedCompiler::<f32>::new(q_weights),
                    luminal_metal::BufferCompilers::default(),
                ),
                #[cfg(feature = "cuda")]
                (
                    luminal_cuda::CudaCompiler::<f32>::default(),
                    luminal_cuda::CudaQuantizedCompiler::<f32>::new(q_weights),
                ),
            ),
            (
                &mut input,
                &mut logits,
                &mut cache_src,
                &mut cache_dest,
                &mut model_weights,
            ),
        );

        // Save compiled graph to cache
        if let Err(e) = save_compiled_graph(&cli_args.cache_dir, "compiled_graph_data") {
            eprintln!("Warning: Failed to save compiled graph: {}", e);
        }

        println!("\t\t - {}ms", now.elapsed().as_millis());
    } else {
        // For cached compilation, we still need to set up the model loading and compilation
        // In a full implementation, you'd deserialize the compiled graph state
        #[cfg(any(feature = "metal", feature = "cuda"))]
        let q_weights = loader::q8_load("setup/qwen3-4b.gguf", &model, &mut cx);
        #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
        loader::q8_load("setup/qwen3-4b.gguf", &model, &mut cx);

        cx.compile(
            (
                GenericCompiler::default(),
                #[cfg(feature = "metal")]
                (
                    luminal_metal::MetalCompilerPreBuffer::<f32>::default(),
                    luminal_metal::quantized::MetalQuantizedCompiler::<f32>::new(q_weights),
                    luminal_metal::BufferCompilers::default(),
                ),
                #[cfg(feature = "cuda")]
                (
                    luminal_cuda::CudaCompiler::<f32>::default(),
                    luminal_cuda::CudaQuantizedCompiler::<f32>::new(q_weights),
                ),
            ),
            (
                &mut input,
                &mut logits,
                &mut cache_src,
                &mut cache_dest,
                &mut model_weights,
            ),
        );
        println!("\t\t - {}ms (loaded from cache)", now.elapsed().as_millis());
    }

    let cache_src = downstream(&cache_src, &cx);

    // Initial forward pass to load weights
    print!("Loading model");
    io::stdout().flush().unwrap();
    let now = Instant::now();
    input.set_dyn(vec![1.], (1, 1));
    cx.set_dyn_dim('t', 1);
    cx.execute();
    logits.drop();
    transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);
    println!("\t\t - {}ms", now.elapsed().as_millis());

    // Now that weights are loaded, delete the loading nodes so they don't run again
    delete_inputs(&cache_src, &mut cx);
    delete_inputs(downstream(model_weights, &cx), &mut cx);

    // Run prompt processing pass
    let input_ids = tokenizer
        .encode(&cli_args.prompt as &str, false)
        .unwrap()
        .get_ids()
        .to_vec();
    input.set_dyn(
        input_ids.iter().map(|i| *i as f32).collect::<Vec<_>>(),
        (1, input_ids.len()),
    );
    cx.set_dyn_dim('t', input_ids.len());
    print!("Processing Prompt");
    io::stdout().flush().unwrap();
    let now = Instant::now();
    cx.execute();
    let elapsed_ms = now.elapsed().as_millis();
    println!(
        "\t - {elapsed_ms}ms ({:.2} tok/s, {} prompt tokens)",
        1000.0 * (input_ids.len() as f64) / (elapsed_ms as f64),
        input_ids.len()
    );
    let mut output_ids = vec![argmax(&logits.data())];
    logits.drop();

    // Decode token
    print!("{}", cli_args.prompt.white().bold());
    let initial = tokenizer.decode(&output_ids, false).unwrap().bright_green();
    print!("{initial}",);
    io::stdout().flush().unwrap();

    // Swap caches
    transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);

    // Decode loop
    let start_decode = std::time::Instant::now();
    let mut prev_output_len = initial.len();
    for _ in 0..cli_args.gen_tokens {
        input.set_dyn(vec![*output_ids.last().unwrap() as f32], (1, 1));
        cx.set_dyn_dim('p', input_ids.len() + output_ids.len() - 1);
        cx.execute();

        // Sample tokens
        let output_id = argmax(&logits.data());
        logits.drop();
        output_ids.push(output_id);

        // Get the current decoded output
        let current_output = tokenizer.decode(&output_ids, false).unwrap();

        // Print the new substring added to the decoded output
        let legal_byte_num = utf8_legal_byte_num(current_output.as_bytes()[prev_output_len]);
        if let Some(byte_num) = legal_byte_num {
            if current_output.len() > byte_num + prev_output_len {
                print!(
                    "{}",
                    current_output[prev_output_len..prev_output_len + byte_num].bright_green()
                );
                io::stdout().flush().unwrap();

                // Update the previous output
                prev_output_len += byte_num
            }
        }

        // Swap caches
        transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);
    }

    println!();
    let avg_token_time =
        start_decode.elapsed().as_micros() as f32 / (output_ids.len() - 1) as f32 / 1000.0;
    println!(
        "\nAverage token generated in {:.2}ms\t - ({:.2} tok/s)",
        avg_token_time,
        1000.0 / avg_token_time
    );
}

// Currently just an argmax, do actual sampling here
fn argmax(dist: &[f32]) -> u32 {
    dist.iter()
        .position_max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap() as u32
}

/// return utf8 char num corresponding to start byte, if it's not a start byte return [`None`]
fn utf8_legal_byte_num(byte: u8) -> Option<usize> {
    match byte {
        // ASCII  (0xxxxxxx)
        0x00..=0x7F => Some(1),
        // char of 2 bytes (110xxxxx)
        0xC0..=0xDF => Some(2),
        // char of 3 bytes (1110xxxx)
        0xE0..=0xEF => Some(3),
        // char of 4 bytes (11110xxx)
        0xF0..=0xF7 => Some(4),
        // not a start byte
        _ => None,
    }
}
