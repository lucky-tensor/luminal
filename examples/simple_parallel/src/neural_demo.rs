use itertools::Itertools;
use luminal::prelude::*;
use luminal_nn::Linear;
use rand::{rng, Rng};
use rayon::prelude::*;
use std::time::Instant;

fn create_and_run_model(id: usize, input_size: usize, output_size: usize) -> Vec<f32> {
    let thread_id = rayon::current_thread_index().unwrap_or(0);
    println!("🧵 Thread {} creating model {} ({}x{})", thread_id, id, input_size, output_size);

    let mut rng = rng();
    let weight = (0..input_size * output_size).map(|_| rng.random()).collect_vec();
    let input_data = (0..input_size).map(|i| (i + id) as f32 * 0.1).collect_vec();

    // Create a new graph
    let mut cx = Graph::new();

    // Create linear layer
    let model = Linear::new(input_size, output_size, false, &mut cx);
    model.weight.set(weight);

    // Make an input tensor
    let input_tensor = cx.tensor(input_size).set(input_data);

    // Feed tensor through model
    let mut output = model.forward(input_tensor).retrieve();

    // Compile the graph for optimal execution
    cx.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "cpu")]
            luminal_cpu::CPUCompiler::default(),
        ),
        &mut output,
    );

    // Execute the graph
    cx.execute();

    println!("✅ Thread {} completed model {} -> [{:.3}, {:.3}, {:.3}, ...]",
             thread_id, id, output.data()[0], output.data()[1], output.data().get(2).unwrap_or(&0.0));

    output.data()
}

fn main() {
    println!("🔬 Parallel Neural Network Demo with Luminal");
    println!("CPU cores available: {}", num_cpus::get());

    let num_models = 8;
    let input_size = 10;
    let output_size = 5;

    println!("\n🚀 Running {} neural network models in parallel...", num_models);

    let start = Instant::now();

    let results: Vec<Vec<f32>> = (0..num_models).into_par_iter()
        .map(|i| create_and_run_model(i, input_size, output_size))
        .collect();

    let elapsed = start.elapsed();

    println!("\n📊 Results Summary:");
    println!("⏱️  Total parallel execution time: {:.2}ms", elapsed.as_millis());
    println!("📈 Average per model: {:.2}ms", elapsed.as_millis() as f64 / num_models as f64);
    println!("🔢 Models processed: {}", results.len());
    println!("🧵 Using {} CPU threads", rayon::current_num_threads());

    println!("\n✅ All {} models executed successfully in parallel!", num_models);
}