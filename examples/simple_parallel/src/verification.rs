use itertools::Itertools;
use luminal::prelude::*;
use luminal_nn::Linear;
use rand::{rng, Rng, SeedableRng};
use rand::rngs::StdRng;

fn main() {
    println!("🔍 Verification: Comparing with ../simple output");

    // Use the same seed for reproducible results
    let mut rng = StdRng::seed_from_u64(12345); // Fixed seed
    let weight = (0..4 * 5).map(|_| rng.gen()).collect_vec();

    // Create a new graph (same as original simple)
    let mut cx = Graph::new();

    // Randomly initialize a linear layer with an input size of 4 and an output size of 5
    let model = Linear::new(4, 5, false, &mut cx);
    model.weight.set(weight.clone());

    // Make an input tensor (same as original simple)
    let a = cx.tensor(4).set(vec![1., 2., 3., 4.]);

    // Feed tensor through model
    let mut b = model.forward(a).retrieve();

    // Compile the graph for optimal execution (same as original simple)
    cx.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "cpu")]
            luminal_cpu::CPUCompiler::default(),
        ),
        &mut b,
    );

    // Execute the graph
    cx.execute();
    println!("Our result: {:?}", b.data());

    // Expected result from ../simple (with different random seed)
    println!("Original simple result: [5.6734047, 5.4501333, 4.3345704, 4.8373823, 5.453207]");

    // Now test with the exact same setup but deterministic weights
    println!("\n🔬 Testing with deterministic weights for exact comparison...");

    // Create deterministic weights (0.0 to 0.95 in 0.05 increments)
    let deterministic_weight: Vec<f32> = (0..20).map(|i| i as f32 * 0.05).collect();

    let mut cx2 = Graph::new();
    let model2 = Linear::new(4, 5, false, &mut cx2);
    model2.weight.set(deterministic_weight);

    let a2 = cx2.tensor(4).set(vec![1., 2., 3., 4.]);
    let mut b2 = model2.forward(a2).retrieve();

    cx2.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "cpu")]
            luminal_cpu::CPUCompiler::default(),
        ),
        &mut b2,
    );

    cx2.execute();
    println!("Deterministic result: {:?}", b2.data());

    // Manual calculation to verify:
    // Input: [1, 2, 3, 4]
    // Weights (4x5 matrix):
    // [0.0, 0.05, 0.10, 0.15, 0.20]
    // [0.25, 0.30, 0.35, 0.40, 0.45]
    // [0.50, 0.55, 0.60, 0.65, 0.70]
    // [0.75, 0.80, 0.85, 0.90, 0.95]
    //
    // Output[0] = 1*0.0 + 2*0.25 + 3*0.50 + 4*0.75 = 0 + 0.5 + 1.5 + 3.0 = 5.0
    // Output[1] = 1*0.05 + 2*0.30 + 3*0.55 + 4*0.80 = 0.05 + 0.6 + 1.65 + 3.2 = 5.5
    // Output[2] = 1*0.10 + 2*0.35 + 3*0.60 + 4*0.85 = 0.1 + 0.7 + 1.8 + 3.4 = 6.0
    // Output[3] = 1*0.15 + 2*0.40 + 3*0.65 + 4*0.90 = 0.15 + 0.8 + 1.95 + 3.6 = 6.5
    // Output[4] = 1*0.20 + 2*0.45 + 3*0.70 + 4*0.95 = 0.2 + 0.9 + 2.1 + 3.8 = 7.0

    let expected = vec![5.0, 5.5, 6.0, 6.5, 7.0];
    println!("Expected manual calc: {:?}", expected);

    // Compare with tolerance
    let tolerance = 1e-6;
    let mut matches = true;
    for (i, (&actual, &expected_val)) in b2.data().iter().zip(expected.iter()).enumerate() {
        if (actual - expected_val).abs() > tolerance {
            println!("❌ Mismatch at index {}: got {}, expected {}", i, actual, expected_val);
            matches = false;
        }
    }

    if matches {
        println!("✅ All values match within tolerance!");
        println!("🎉 Our implementation produces mathematically correct results!");
    }
}