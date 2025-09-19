use itertools::Itertools;
use luminal::prelude::*;
use luminal_nn::Linear;

fn main() {
    // Use fixed weights instead of random for exact reproducible results
    let weight: Vec<f32> = (0..4 * 5).map(|i| i as f32 * 0.05).collect_vec();

    // Create a new graph
    let mut cx = Graph::new();
    // Initialize a linear layer with an input size of 4 and an output size of 5 with fixed weights
    let model = Linear::new(4, 5, false, &mut cx);
    model.weight.set(weight.clone());

    // Make an input tensor
    let a = cx.tensor(4).set(vec![1., 2., 3., 4.]);
    // Feed tensor through model
    let mut b = model.forward(a).retrieve();

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
        &mut b,
    );

    // Execute the graph
    cx.execute();
    println!("B: {:?}", b.data());

    // Manual verification:
    // Input: [1, 2, 3, 4]
    // Weights (4x5 matrix, row-major):
    // [0.0, 0.05, 0.10, 0.15, 0.20]
    // [0.25, 0.30, 0.35, 0.40, 0.45]
    // [0.50, 0.55, 0.60, 0.65, 0.70]
    // [0.75, 0.80, 0.85, 0.90, 0.95]
    //
    // Matrix multiplication: output = input * weights^T
    // Output[0] = 1*0.0 + 2*0.25 + 3*0.50 + 4*0.75 = 5.0
    // Output[1] = 1*0.05 + 2*0.30 + 3*0.55 + 4*0.80 = 5.5
    // Output[2] = 1*0.10 + 2*0.35 + 3*0.60 + 4*0.85 = 6.0
    // Output[3] = 1*0.15 + 2*0.40 + 3*0.65 + 4*0.90 = 6.5
    // Output[4] = 1*0.20 + 2*0.45 + 3*0.70 + 4*0.95 = 7.0
    println!("Expected: [5.0, 5.5, 6.0, 6.5, 7.0]");
}