use itertools::Itertools;
use luminal::prelude::*;
use luminal_nn::Linear;

fn main() {
    println!("🎯 Fixed Exact Match: Using identical weights as ../simple/fixed");

    // Use exactly the same fixed weights as the original simple/fixed
    let weight: Vec<f32> = (0..4 * 5).map(|i| i as f32 * 0.05).collect_vec();

    // Create a new graph (identical to original)
    let mut cx = Graph::new();
    // Initialize a linear layer with an input size of 4 and an output size of 5 with fixed weights
    let model = Linear::new(4, 5, false, &mut cx);
    model.weight.set(weight.clone());

    // Make an input tensor (identical to original)
    let a = cx.tensor(4).set(vec![1., 2., 3., 4.]);
    // Feed tensor through model
    let mut b = model.forward(a).retrieve();

    // Compile the graph for optimal execution (identical to original)
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
        &mut b,
    );

    // Execute the graph
    cx.execute();
    println!("B: {:?}", b.data());

    // Compare with expected
    let expected = vec![5.0, 5.5, 6.0, 6.5, 7.0];
    println!("Expected: {:?}", expected);

    // Verify exact match
    let tolerance = 1e-10; // Very strict tolerance for exact match
    let mut exact_match = true;
    for (i, (&actual, &expected_val)) in b.data().iter().zip(expected.iter()).enumerate() {
        if (actual - expected_val).abs() > tolerance {
            println!("❌ Difference at index {}: got {}, expected {}, diff: {}",
                     i, actual, expected_val, (actual - expected_val).abs());
            exact_match = false;
        }
    }

    if exact_match {
        println!("✅ PERFECT MATCH! Our output exactly matches ../simple/fixed");
    } else {
        println!("❌ Output differs from ../simple/fixed");
    }
}