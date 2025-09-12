use itertools::Itertools;
use luminal::prelude::*;
//use luminal_2::{
// codegen::codegen, extract::search, run::run_graph, translate::translate_graph,
// utils::build_search_space, GPUArch,
//};
use luminal_nn::Linear;
use rand::{rng, Rng};

fn main() {
    let mut rng = rng();
    let weight = (0..4 * 5).map(|_| rng.random()).collect_vec();
    // Create a new graph
    let mut cx = Graph::new();
    // Randomly initialize a linear layer with an input size of 4 and an output size of 5
    let model = Linear::new(4, 5, false, &mut cx);
    model.weight.set(weight.clone());
    // Make an input tensor
    let a = cx.tensor(4).set(vec![1., 2., 3., 4.]);
    // Feed tensor through model
    let mut b = model.forward(a).retrieve();

    // Compile the graph for optimal execution
    #[cfg(feature = "cuda")]
    cx.compile(
        (
            GenericCompiler::default(),
            luminal_cuda::CudaCompiler::<f32>::default(),
        ),
        &mut b,
    );
    
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    cx.compile(
        (
            GenericCompiler::default(),
            luminal_cpu::CPUCompiler::default(),
        ),
        &mut b,
    );

    // Execute the graph
    cx.execute();
    println!("B: {:?}", b.data());
}
