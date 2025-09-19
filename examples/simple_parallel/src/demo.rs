use rayon::prelude::*;
use std::time::Instant;

fn main() {
    println!("🔬 Simple Parallelization Demo");
    println!("CPU cores available: {}", num_cpus::get());

    // Simple computation to demonstrate parallelization
    let data: Vec<i32> = (0..1000).collect();

    // Standard sequential processing
    let start = Instant::now();
    let sequential_result: Vec<i32> = data.iter()
        .map(|&x| {
            // Simulate some work
            let mut sum = 0;
            for i in 0..1000 {
                sum += x * i;
            }
            sum % 1000000
        })
        .collect();
    let sequential_time = start.elapsed();

    // Parallel processing
    let start = Instant::now();
    let parallel_result: Vec<i32> = data.par_iter()
        .map(|&x| {
            let thread_id = rayon::current_thread_index().unwrap_or(0);
            if x < 5 {
                println!("Thread {} processing element {}", thread_id, x);
            }

            // Same computation as sequential
            let mut sum = 0;
            for i in 0..1000 {
                sum += x * i;
            }
            sum % 1000000
        })
        .collect();
    let parallel_time = start.elapsed();

    // Verify results are the same
    assert_eq!(sequential_result, parallel_result);

    println!("Sequential time: {:.2}ms", sequential_time.as_millis());
    println!("Parallel time: {:.2}ms", parallel_time.as_millis());
    println!("Speedup: {:.2}x", sequential_time.as_millis() as f64 / parallel_time.as_millis() as f64);
    println!("✅ Results verified identical!");
}