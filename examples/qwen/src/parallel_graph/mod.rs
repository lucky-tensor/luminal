//! Parallel graph definition implementation
//!
//! This module provides parallel graph construction capabilities to speed up
//! the graph definition phase by parallelizing tensor and layer specification
//! creation while maintaining thread safety and correctness.

pub mod specs;
pub mod builder;
pub mod assembly;
pub mod model_builder;
pub mod debug;
pub mod progress;

#[cfg(test)]
pub mod integration_tests;

pub mod benchmarks;

// Export key types for parallel graph operations
pub use specs::{TensorSpec, LayerSpec, TensorType, LayerType, LayerParameters};
pub use builder::{ParallelGraphBuilder, ParallelGraphError, ParallelGraphResult};
pub use debug::{ParallelDebugger, DebugLevel};
pub use progress::{ProgressTracker, ProgressPhase, ProgressCounts};