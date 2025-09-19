//! Integration tests for the parallel graph implementation
//!
//! These tests verify that the parallel implementation produces correct
//! and consistent results compared to expected behavior.

#[cfg(test)]
mod tests {
    use crate::parallel_graph::model_builder::{
        ParallelModelBuilder, QwenConfig
    };
    use crate::parallel_graph::debug::DebugLevel;

    /// Test that parallel model building produces consistent results
    #[test]
    fn test_parallel_model_consistency() {
        let config = QwenConfig {
            num_layers: 2,
            hidden_dim: 256,
            n_heads: 8,
            n_kv_heads: 8,
            mlp_dim: 1024,
            sequence_length: 512,
            head_dim: 32,
            vocab_size: 10000,
        };

        // Build the same model multiple times to ensure consistency
        let model1 = ParallelModelBuilder::with_config(config.clone())
            .with_debug_level(DebugLevel::Basic)
            .build()
            .expect("First model build should succeed");

        let model2 = ParallelModelBuilder::with_config(config.clone())
            .with_debug_level(DebugLevel::Basic)
            .build()
            .expect("Second model build should succeed");

        // Models should have the same configuration
        assert_eq!(model1.config.num_layers, model2.config.num_layers);
        assert_eq!(model1.config.hidden_dim, model2.config.hidden_dim);
        assert_eq!(model1.config.n_heads, model2.config.n_heads);

        // Models should have the same structure
        assert_eq!(model1.assembled_graph.tensors.len(), model2.assembled_graph.tensors.len());
        assert_eq!(model1.assembled_graph.layers.len(), model2.assembled_graph.layers.len());

        // Parameter counts should match
        assert_eq!(model1.parameter_count(), model2.parameter_count());
    }

    /// Test that models built with different thread counts produce the same results
    #[test]
    fn test_thread_count_consistency() {
        let config = QwenConfig {
            num_layers: 3,
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 4,
            mlp_dim: 512,
            sequence_length: 256,
            head_dim: 32,
            vocab_size: 5000,
        };

        // Build with default threading
        let model_default = ParallelModelBuilder::with_config(config.clone())
            .build()
            .expect("Default model build should succeed");

        // Build with specific thread count
        let model_threaded = ParallelModelBuilder::with_config(config)
            .build_with_threads(2)
            .expect("Threaded model build should succeed");

        // Should produce the same structure
        assert_eq!(model_default.assembled_graph.tensors.len(), model_threaded.assembled_graph.tensors.len());
        assert_eq!(model_default.assembled_graph.layers.len(), model_threaded.assembled_graph.layers.len());
        assert_eq!(model_default.parameter_count(), model_threaded.parameter_count());
    }

    /// Test that models with debugging produce correct results
    #[test]
    fn test_debugging_consistency() {
        let config = QwenConfig {
            num_layers: 2,
            hidden_dim: 64,
            n_heads: 2,
            n_kv_heads: 2,
            mlp_dim: 256,
            sequence_length: 128,
            head_dim: 32,
            vocab_size: 1000,
        };

        // Build without debugging
        let model_no_debug = ParallelModelBuilder::with_config(config.clone())
            .build()
            .expect("Model without debugging should succeed");

        // Build with different debug levels
        let model_basic_debug = ParallelModelBuilder::with_config(config.clone())
            .with_debug_level(DebugLevel::Basic)
            .build()
            .expect("Model with basic debugging should succeed");

        let model_verbose_debug = ParallelModelBuilder::with_config(config.clone())
            .with_debug_level(DebugLevel::Verbose)
            .build()
            .expect("Model with verbose debugging should succeed");

        // All models should have the same structure regardless of debugging
        let tensor_count = model_no_debug.assembled_graph.tensors.len();
        let layer_count = model_no_debug.assembled_graph.layers.len();
        let param_count = model_no_debug.parameter_count();

        assert_eq!(model_basic_debug.assembled_graph.tensors.len(), tensor_count);
        assert_eq!(model_basic_debug.assembled_graph.layers.len(), layer_count);
        assert_eq!(model_basic_debug.parameter_count(), param_count);

        assert_eq!(model_verbose_debug.assembled_graph.tensors.len(), tensor_count);
        assert_eq!(model_verbose_debug.assembled_graph.layers.len(), layer_count);
        assert_eq!(model_verbose_debug.parameter_count(), param_count);
    }

    /// Test that models with progress tracking work correctly
    #[test]
    fn test_progress_tracking_consistency() {
        let config = QwenConfig {
            num_layers: 3,
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 4,
            mlp_dim: 512,
            sequence_length: 256,
            head_dim: 32,
            vocab_size: 2000,
        };

        // Build without progress tracking
        let model_no_progress = ParallelModelBuilder::with_config(config.clone())
            .build()
            .expect("Model without progress tracking should succeed");

        // Build with progress tracking
        let model_with_progress = ParallelModelBuilder::with_config(config)
            .with_progress_tracking()
            .build()
            .expect("Model with progress tracking should succeed");

        // Models should have the same structure regardless of progress tracking
        assert_eq!(
            model_no_progress.assembled_graph.tensors.len(),
            model_with_progress.assembled_graph.tensors.len()
        );
        assert_eq!(
            model_no_progress.assembled_graph.layers.len(),
            model_with_progress.assembled_graph.layers.len()
        );
        assert_eq!(
            model_no_progress.parameter_count(),
            model_with_progress.parameter_count()
        );
    }

    /// Test that larger models are built correctly
    #[test]
    fn test_large_model_integration() {
        let config = QwenConfig {
            num_layers: 8,
            hidden_dim: 512,
            n_heads: 8,
            n_kv_heads: 8,
            mlp_dim: 2048,
            sequence_length: 1024,
            head_dim: 64,
            vocab_size: 50000,
        };

        let model = ParallelModelBuilder::with_config(config)
            .with_debug_level(DebugLevel::Basic)
            .with_progress_tracking()
            .build()
            .expect("Large model build should succeed");

        // Should have created many tensors and layers
        assert!(model.assembled_graph.tensors.len() > 50);
        assert!(model.assembled_graph.layers.len() > 0);

        // Should have a significant parameter count
        assert!(model.parameter_count() > 1_000_000);

        // Build metrics should be reasonable
        assert!(model.metrics.tensor_count > 0);
        assert!(model.metrics.thread_count > 0);
        assert!(model.metrics.parallel_phase_duration.as_nanos() > 0);
        assert!(model.metrics.assembly_phase_duration.as_nanos() > 0);
    }

    /// Test edge cases and error conditions
    #[test]
    fn test_error_conditions() {
        // Test with invalid configurations that should be caught by validation
        let invalid_config = QwenConfig {
            num_layers: 0, // Invalid: zero layers
            hidden_dim: 256,
            n_heads: 8,
            n_kv_heads: 8,
            mlp_dim: 1024,
            sequence_length: 512,
            head_dim: 32,
            vocab_size: 10000,
        };

        let result = ParallelModelBuilder::with_config(invalid_config)
            .build();

        assert!(result.is_err(), "Build with zero layers should fail");

        // Test with incompatible head dimensions
        let incompatible_config = QwenConfig {
            num_layers: 2,
            hidden_dim: 100, // Not divisible by n_heads
            n_heads: 3,
            n_kv_heads: 3,
            mlp_dim: 400,
            sequence_length: 512,
            head_dim: 33,
            vocab_size: 10000,
        };

        let result = ParallelModelBuilder::with_config(incompatible_config)
            .build();

        assert!(result.is_err(), "Build with incompatible dimensions should fail");
    }

    /// Test that the parallel implementation scales correctly
    #[test]
    fn test_scaling_behavior() {
        // Test small model
        let small_config = QwenConfig {
            num_layers: 1,
            hidden_dim: 64,
            n_heads: 2,
            n_kv_heads: 2,
            mlp_dim: 256,
            sequence_length: 128,
            head_dim: 32,
            vocab_size: 1000,
        };

        // Test medium model
        let medium_config = QwenConfig {
            num_layers: 4,
            hidden_dim: 256,
            n_heads: 8,
            n_kv_heads: 8,
            mlp_dim: 1024,
            sequence_length: 512,
            head_dim: 32,
            vocab_size: 10000,
        };

        let small_model = ParallelModelBuilder::with_config(small_config)
            .build()
            .expect("Small model should build successfully");

        let medium_model = ParallelModelBuilder::with_config(medium_config)
            .build()
            .expect("Medium model should build successfully");

        // Medium model should have significantly more parameters
        assert!(medium_model.parameter_count() > small_model.parameter_count() * 4);

        // Medium model should have more tensors and layers
        assert!(medium_model.assembled_graph.tensors.len() > small_model.assembled_graph.tensors.len());
        assert!(medium_model.assembled_graph.layers.len() >= small_model.assembled_graph.layers.len());
    }

    /// Test that build summaries contain expected information
    #[test]
    fn test_build_summary_completeness() {
        let config = QwenConfig {
            num_layers: 2,
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 4,
            mlp_dim: 512,
            sequence_length: 256,
            head_dim: 32,
            vocab_size: 5000,
        };

        let model = ParallelModelBuilder::with_config(config)
            .with_debug_level(DebugLevel::Verbose)
            .build()
            .expect("Model build should succeed");

        let summary = model.build_summary();

        // Summary should contain key information
        assert!(summary.contains("Layers: 2"));
        assert!(summary.contains("Parameters:"));
        assert!(summary.contains("Tensors Created:"));
        assert!(summary.contains("Parallel Phase:"));
        assert!(summary.contains("Assembly Phase:"));
        assert!(summary.contains("Threads Used:"));
        assert!(summary.contains("Total Build Time:"));
    }
}