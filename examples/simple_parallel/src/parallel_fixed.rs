use itertools::Itertools;
use luminal::prelude::*;
use luminal_nn::Linear;
use rayon::prelude::*;
use std::time::Instant;

fn create_and_run_fixed_model(id: usize) -> Vec<f32> {
    let thread_id = rayon::current_thread_index().unwrap_or(0);

    // Use exactly the same fixed weights as ../simple/fixed
    let weight: Vec<f32> = (0..4 * 5).map(|i| i as f32 * 0.05).collect_vec();

    // Create a new graph
    let mut cx = Graph::new();
    let model = Linear::new(4, 5, false, &mut cx);
    model.weight.set(weight);

    // Make an input tensor (same as original)
    let a = cx.tensor(4).set(vec![1., 2., 3., 4.]);
    let mut b = model.forward(a).retrieve();

    // Compile and execute
    cx.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "cpu")]
            luminal_cpu::CPUCompiler::default(),
        ),
        &mut b,
    );
    cx.execute();

    println!("🧵 Thread {} processed model {} -> B: {:?}", thread_id, id, b.data());

    b.data()
}

fn main() {
    println!("🚀 Parallel Fixed Weights: Multiple threads, identical outputs");
    println!("Expected result: [5.0, 5.5, 6.0, 6.5, 7.0]");
    println!("CPU cores available: {}", num_cpus::get());
    println!("");

    let num_models = 6;
    let start = Instant::now();

    let results: Vec<Vec<f32>> = (0..num_models).into_par_iter()
        .map(|i| create_and_run_fixed_model(i))
        .collect();

    let elapsed = start.elapsed();

    println!("\n📊 Verification:");
    let expected = vec![5.0, 5.5, 6.0, 6.5, 7.0];

    let mut all_match = true;
    for (i, result) in results.iter().enumerate() {
        if *result == expected {
            println!("✅ Model {} output matches expected result", i);
        } else {
            println!("❌ Model {} output differs: {:?}", i, result);
            all_match = false;
        }
    }

    if all_match {
        println!("\n🎉 SUCCESS: All {} parallel models produced identical results!", num_models);
        println!("⏱️  Parallel execution time: {:.2}ms", elapsed.as_millis());
        println!("📈 Average per model: {:.2}ms", elapsed.as_millis() as f64 / num_models as f64);
        println!("🔍 Output: {:?}", expected);
    } else {
        println!("\n❌ Some models produced different results");
    }
}