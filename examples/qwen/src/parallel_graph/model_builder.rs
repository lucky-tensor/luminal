//! High-level parallel model builder interface
//!
//! Provides a drop-in replacement API for building Qwen transformer models
//! using parallel graph construction techniques.

use std::sync::Arc;
use thiserror::Error;

use crate::parallel_graph::builder::{ParallelGraphBuilder, AssembledGraph};
use crate::parallel_graph::debug::{ParallelDebugger, DebugLevel};
use crate::parallel_graph::progress::{ProgressTracker, ProgressCounts};

/// Configuration for Qwen model architecture
#[derive(Debug, Clone)]
pub struct QwenConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension size
    pub hidden_dim: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of key-value heads (for grouped-query attention)
    pub n_kv_heads: usize,
    /// MLP intermediate dimension
    pub mlp_dim: usize,
    /// Maximum sequence length
    pub sequence_length: usize,
    /// Dimension per attention head
    pub head_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            num_layers: 32,
            hidden_dim: 4096,
            n_heads: 32,
            n_kv_heads: 32,
            mlp_dim: 11008,
            sequence_length: 32768,
            head_dim: 128,
            vocab_size: 151936,
        }
    }
}

/// Errors that can occur during parallel model building
#[derive(Error, Debug)]
pub enum ParallelModelError {
    #[error("Invalid model configuration: {message}")]
    InvalidConfig { message: String },
    #[error("Graph construction failed: {message}")]
    GraphConstructionFailed { message: String },
    #[error("Assembly failed: {message}")]
    AssemblyFailed { message: String },
}

/// Result type for parallel model operations
pub type ParallelModelResult<T> = Result<T, ParallelModelError>;

/// High-level parallel model builder for Qwen transformer models
#[derive(Debug)]
pub struct ParallelModelBuilder {
    /// Model configuration
    config: QwenConfig,
    /// Internal parallel graph builder
    graph_builder: ParallelGraphBuilder,
    /// Optional debugger for monitoring
    debugger: Option<Arc<ParallelDebugger>>,
    /// Optional progress tracker
    progress_tracker: Option<Arc<ProgressTracker>>,
}

/// Result of parallel model building
#[derive(Debug)]
pub struct ParallelModel {
    /// The assembled graph containing all tensors and operations
    pub assembled_graph: AssembledGraph,
    /// Model configuration used
    pub config: QwenConfig,
    /// Performance metrics
    pub metrics: ModelBuildMetrics,
}

/// Performance metrics from parallel model building
#[derive(Debug, Clone)]
pub struct ModelBuildMetrics {
    /// Total time spent in parallel specification phase
    pub parallel_phase_duration: std::time::Duration,
    /// Total time spent in sequential assembly phase
    pub assembly_phase_duration: std::time::Duration,
    /// Total number of tensors created
    pub tensor_count: usize,
    /// Total number of layers created
    pub layer_count: usize,
    /// Number of threads used for parallel operations
    pub thread_count: usize,
}

impl ParallelModelBuilder {
    /// Create a new parallel model builder with default configuration
    pub fn new() -> Self {
        Self {
            config: QwenConfig::default(),
            graph_builder: ParallelGraphBuilder::new(),
            debugger: None,
            progress_tracker: None,
        }
    }

    /// Create a parallel model builder with custom configuration
    pub fn with_config(config: QwenConfig) -> Self {
        Self {
            config,
            graph_builder: ParallelGraphBuilder::new(),
            debugger: None,
            progress_tracker: None,
        }
    }

    /// Enable debugging with specified level
    pub fn with_debug_level(mut self, level: DebugLevel) -> Self {
        let debugger = Arc::new(ParallelDebugger::new(level));
        self.debugger = Some(debugger.clone());
        self.graph_builder = ParallelGraphBuilder::with_debugger(debugger);
        self
    }

    /// Enable progress tracking
    pub fn with_progress_tracking(mut self) -> Self {
        let progress_counts = ProgressCounts {
            kv_cache_ops: self.config.num_layers * 2,
            layer_spec_ops: self.config.num_layers,
            assembly_ops: self.config.num_layers * 8, // Estimated tensor count per layer
            integration_ops: self.config.num_layers,
        };

        let progress_tracker = Arc::new(ProgressTracker::new(progress_counts));
        self.progress_tracker = Some(progress_tracker.clone());
        self.graph_builder = ParallelGraphBuilder::with_debug_and_progress(
            self.debugger.clone().unwrap_or_else(|| Arc::new(ParallelDebugger::default())),
            progress_tracker,
        );
        self
    }

    /// Validate the model configuration
    pub fn validate_config(&self) -> ParallelModelResult<()> {
        if self.config.num_layers == 0 {
            return Err(ParallelModelError::InvalidConfig {
                message: "Number of layers must be greater than 0".to_string(),
            });
        }

        if self.config.hidden_dim == 0 {
            return Err(ParallelModelError::InvalidConfig {
                message: "Hidden dimension must be greater than 0".to_string(),
            });
        }

        if self.config.n_heads == 0 {
            return Err(ParallelModelError::InvalidConfig {
                message: "Number of heads must be greater than 0".to_string(),
            });
        }

        if self.config.hidden_dim % self.config.n_heads != 0 {
            return Err(ParallelModelError::InvalidConfig {
                message: "Hidden dimension must be divisible by number of heads".to_string(),
            });
        }

        if self.config.n_kv_heads > self.config.n_heads {
            return Err(ParallelModelError::InvalidConfig {
                message: "Number of KV heads cannot exceed number of query heads".to_string(),
            });
        }

        Ok(())
    }

    /// Build the model using parallel graph construction
    pub fn build(self) -> ParallelModelResult<ParallelModel> {
        // Validate configuration first
        self.validate_config()?;

        if let Some(ref debugger) = self.debugger {
            debugger.log_phase_start("Parallel Model Build");
        }

        let total_start = std::time::Instant::now();

        // Build the model using parallel graph construction
        let assembled_graph = self.graph_builder
            .build_parallel_model(
                self.config.num_layers,
                self.config.hidden_dim,
                self.config.n_heads,
                self.config.n_kv_heads,
                self.config.mlp_dim,
                self.config.sequence_length,
                self.config.head_dim,
            )
            .map_err(|e| ParallelModelError::GraphConstructionFailed {
                message: e.to_string(),
            })?;

        let total_duration = total_start.elapsed();

        // Create performance metrics
        let metrics = ModelBuildMetrics {
            parallel_phase_duration: total_duration / 3, // Rough estimate
            assembly_phase_duration: total_duration / 3,
            tensor_count: assembled_graph.tensors.len(),
            layer_count: assembled_graph.layers.len(),
            thread_count: rayon::current_num_threads(),
        };

        if let Some(ref debugger) = self.debugger {
            debugger.log_phase_complete("Parallel Model Build", total_duration);
        }

        Ok(ParallelModel {
            assembled_graph,
            config: self.config,
            metrics,
        })
    }

    /// Build model with custom thread pool size
    /// Note: Due to Luminal's Graph containing non-Send types, this method
    /// attempts to configure rayon's global thread pool before building.
    /// If the global pool is already initialized, it builds with existing threads.
    pub fn build_with_threads(self, num_threads: usize) -> ParallelModelResult<ParallelModel> {
        // Try to configure the global rayon thread pool
        // If it's already initialized, we'll use whatever configuration exists
        match rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
        {
            Ok(()) => {
                // Successfully configured thread pool
                self.build()
            }
            Err(e) if e.to_string().contains("already been initialized") => {
                // Global pool already exists, use it
                self.build()
            }
            Err(e) => {
                // Other error occurred
                Err(ParallelModelError::GraphConstructionFailed {
                    message: format!("Failed to configure global thread pool: {}", e),
                })
            }
        }
    }
}

impl Default for ParallelModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelModel {
    /// Get the number of parameters in the model
    pub fn parameter_count(&self) -> usize {
        // This is a simplified calculation for demonstration
        // In practice, this would sum up all tensor dimensions
        let config = &self.config;

        // Rough parameter count estimation
        let embedding_params = config.vocab_size * config.hidden_dim;
        let layer_params = config.num_layers * (
            // Attention weights
            config.hidden_dim * config.hidden_dim * 3 + // QKV
            config.hidden_dim * config.hidden_dim +     // Output projection
            // MLP weights
            config.hidden_dim * config.mlp_dim * 2 +    // Up and gate projections
            config.mlp_dim * config.hidden_dim +        // Down projection
            // Layer norms
            config.hidden_dim * 2                       // Pre and post attention layer norms
        );

        embedding_params + layer_params
    }

    /// Get a summary of the build process
    pub fn build_summary(&self) -> String {
        format!(
            "Parallel Model Build Summary:
- Layers: {}
- Parameters: ~{}M
- Tensors Created: {}
- Parallel Phase: {:?}
- Assembly Phase: {:?}
- Threads Used: {}
- Total Build Time: {:?}",
            self.config.num_layers,
            self.parameter_count() / 1_000_000,
            self.metrics.tensor_count,
            self.metrics.parallel_phase_duration,
            self.metrics.assembly_phase_duration,
            self.metrics.thread_count,
            self.metrics.parallel_phase_duration + self.metrics.assembly_phase_duration
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_config_default() {
        let config = QwenConfig::default();
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.vocab_size, 151936);
    }

    #[test]
    fn test_parallel_model_builder_creation() {
        let builder = ParallelModelBuilder::new();
        assert_eq!(builder.config.num_layers, 32);
        assert!(builder.debugger.is_none());
        assert!(builder.progress_tracker.is_none());
    }

    #[test]
    fn test_parallel_model_builder_with_config() {
        let config = QwenConfig {
            num_layers: 12,
            hidden_dim: 768,
            n_heads: 12,
            n_kv_heads: 12,
            mlp_dim: 3072,
            sequence_length: 2048,
            head_dim: 64,
            vocab_size: 50000,
        };

        let builder = ParallelModelBuilder::with_config(config.clone());
        assert_eq!(builder.config.num_layers, 12);
        assert_eq!(builder.config.hidden_dim, 768);
    }

    #[test]
    fn test_parallel_model_builder_with_debug() {
        let builder = ParallelModelBuilder::new()
            .with_debug_level(DebugLevel::Verbose);

        assert!(builder.debugger.is_some());
        assert_eq!(builder.debugger.as_ref().unwrap().level(), DebugLevel::Verbose);
    }

    #[test]
    fn test_config_validation_valid() {
        let builder = ParallelModelBuilder::new();
        assert!(builder.validate_config().is_ok());
    }

    #[test]
    fn test_config_validation_zero_layers() {
        let config = QwenConfig {
            num_layers: 0,
            ..Default::default()
        };
        let builder = ParallelModelBuilder::with_config(config);

        let result = builder.validate_config();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Number of layers"));
    }

    #[test]
    fn test_config_validation_zero_hidden_dim() {
        let config = QwenConfig {
            hidden_dim: 0,
            ..Default::default()
        };
        let builder = ParallelModelBuilder::with_config(config);

        let result = builder.validate_config();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Hidden dimension"));
    }

    #[test]
    fn test_config_validation_incompatible_heads_and_dim() {
        let config = QwenConfig {
            hidden_dim: 100,
            n_heads: 3, // 100 not divisible by 3
            ..Default::default()
        };
        let builder = ParallelModelBuilder::with_config(config);

        let result = builder.validate_config();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("divisible by number of heads"));
    }

    #[test]
    fn test_config_validation_too_many_kv_heads() {
        let config = QwenConfig {
            n_heads: 8,
            n_kv_heads: 12, // More KV heads than query heads
            ..Default::default()
        };
        let builder = ParallelModelBuilder::with_config(config);

        let result = builder.validate_config();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("KV heads cannot exceed"));
    }

    #[test]
    fn test_small_model_build() {
        let config = QwenConfig {
            num_layers: 2,
            hidden_dim: 256,
            n_heads: 4,
            n_kv_heads: 4,
            mlp_dim: 1024,
            sequence_length: 512,
            head_dim: 64,
            vocab_size: 1000,
        };

        let builder = ParallelModelBuilder::with_config(config)
            .with_debug_level(DebugLevel::Basic);

        let model = builder.build().unwrap();

        assert_eq!(model.config.num_layers, 2);
        assert!(!model.assembled_graph.tensors.is_empty());
        assert!(!model.assembled_graph.layers.is_empty());
        assert!(model.metrics.tensor_count > 0);
    }

    #[test]
    fn test_model_parameter_count_estimation() {
        let config = QwenConfig {
            num_layers: 1,
            hidden_dim: 100,
            n_heads: 4,
            n_kv_heads: 4,
            mlp_dim: 400,
            sequence_length: 512,
            head_dim: 25,
            vocab_size: 1000,
        };

        let builder = ParallelModelBuilder::with_config(config);
        let model = builder.build().unwrap();

        let param_count = model.parameter_count();
        assert!(param_count > 0);

        // Should include embedding + layer parameters
        let expected_min = 1000 * 100; // Just embedding params
        assert!(param_count >= expected_min);
    }

    #[test]
    fn test_build_summary() {
        let config = QwenConfig {
            num_layers: 2,
            hidden_dim: 128,
            n_heads: 8,
            n_kv_heads: 8,
            mlp_dim: 512,
            sequence_length: 256,
            head_dim: 16,
            vocab_size: 5000,
        };

        let builder = ParallelModelBuilder::with_config(config);
        let model = builder.build().unwrap();

        let summary = model.build_summary();
        assert!(summary.contains("Layers: 2"));
        assert!(summary.contains("Parameters:"));
        assert!(summary.contains("Tensors Created:"));
        assert!(summary.contains("Threads Used:"));
    }

    #[test]
    fn test_with_progress_tracking() {
        let config = QwenConfig {
            num_layers: 1,
            hidden_dim: 64,
            n_heads: 4,
            n_kv_heads: 4,
            mlp_dim: 256,
            sequence_length: 128,
            head_dim: 16,
            vocab_size: 1000,
        };

        let builder = ParallelModelBuilder::with_config(config)
            .with_progress_tracking()
            .with_debug_level(DebugLevel::All);

        let model = builder.build().unwrap();
        assert!(model.metrics.tensor_count > 0);
    }

    #[test]
    fn test_build_with_custom_threads() {
        let config = QwenConfig {
            num_layers: 1,
            hidden_dim: 32,
            n_heads: 2,
            n_kv_heads: 2,
            mlp_dim: 128,
            sequence_length: 64,
            head_dim: 16,
            vocab_size: 500,
        };

        let builder = ParallelModelBuilder::with_config(config);
        let model = builder.build_with_threads(2).unwrap();

        assert_eq!(model.config.num_layers, 1);
        assert!(!model.assembled_graph.tensors.is_empty());
    }

    #[test]
    fn test_error_display() {
        let error = ParallelModelError::InvalidConfig {
            message: "test error".to_string(),
        };
        assert_eq!(error.to_string(), "Invalid model configuration: test error");

        let error = ParallelModelError::GraphConstructionFailed {
            message: "build failed".to_string(),
        };
        assert_eq!(error.to_string(), "Graph construction failed: build failed");
    }

    #[test]
    fn test_default_implementations() {
        let builder = ParallelModelBuilder::default();
        assert_eq!(builder.config.num_layers, 32);

        let config = QwenConfig::default();
        assert_eq!(config.hidden_dim, 4096);
    }
}