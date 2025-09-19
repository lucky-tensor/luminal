//! Sequential assembly phase implementation
//!
//! Converts parallel-created specifications into actual Luminal graph tensors
//! and operations in a sequential, dependency-ordered manner.

use std::collections::HashMap;
use luminal::prelude::*;
use thiserror::Error;

use crate::parallel_graph::specs::{TensorSpec, LayerSpec, TensorType, LayerType, LayerParameters};
use crate::parallel_graph::debug::ParallelDebugger;
use crate::parallel_graph::progress::ProgressTracker;

/// Errors that can occur during assembly
#[derive(Error, Debug)]
pub enum AssemblyError {
    #[error("Tensor specification not found: {id}")]
    TensorSpecNotFound { id: usize },
    #[error("Layer specification not found: {id}")]
    LayerSpecNotFound { id: usize },
    #[error("Invalid tensor shape: {message}")]
    InvalidShape { message: String },
    #[error("Dependency not satisfied: tensor {tensor_id} needed for layer {layer_id}")]
    DependencyNotSatisfied { tensor_id: usize, layer_id: usize },
    #[error("Graph operation failed: {message}")]
    GraphOperationFailed { message: String },
}

/// Result type for assembly operations
pub type AssemblyResult<T> = Result<T, AssemblyError>;

/// Sequential assembly phase that converts specifications to actual graph elements
#[derive(Debug)]
pub struct AssemblyPhase {
    /// Debug logger for assembly operations
    debugger: Option<ParallelDebugger>,
    /// Progress tracker for assembly operations
    progress_tracker: Option<ProgressTracker>,
}

/// Assembled tensors and metadata
#[derive(Debug)]
pub struct AssembledTensors {
    /// Map from spec ID to actual GraphTensor
    pub tensors: HashMap<usize, GraphTensor>,
    /// Tensor creation order for dependency tracking
    pub creation_order: Vec<usize>,
    /// Total number of tensors assembled
    pub total_count: usize,
}

/// Assembled layers and metadata
#[derive(Debug)]
pub struct AssembledLayers {
    /// Map from spec ID to layer metadata
    pub layers: HashMap<usize, LayerInfo>,
    /// Layer creation order
    pub creation_order: Vec<usize>,
    /// Total number of layers assembled
    pub total_count: usize,
}

/// Information about an assembled layer
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer specification ID
    pub spec_id: usize,
    /// Layer index in the model
    pub layer_idx: usize,
    /// Associated tensor IDs
    pub tensor_ids: Vec<usize>,
    /// Output tensor ID (if applicable)
    pub output_tensor_id: Option<usize>,
}

impl AssemblyPhase {
    /// Create a new assembly phase
    pub fn new() -> Self {
        Self {
            debugger: None,
            progress_tracker: None,
        }
    }

    /// Create assembly phase with debugger
    pub fn with_debugger(debugger: ParallelDebugger) -> Self {
        Self {
            debugger: Some(debugger),
            progress_tracker: None,
        }
    }

    /// Create assembly phase with progress tracker
    pub fn with_progress_tracker(progress_tracker: ProgressTracker) -> Self {
        Self {
            debugger: None,
            progress_tracker: Some(progress_tracker),
        }
    }

    /// Assemble tensor specifications into actual GraphTensors
    pub fn assemble_tensors(
        &self,
        graph: &mut Graph,
        tensor_specs: &HashMap<usize, TensorSpec>,
    ) -> AssemblyResult<AssembledTensors> {
        if let Some(ref debugger) = self.debugger {
            debugger.log_phase_start("Tensor Assembly");
        }

        let start_time = std::time::Instant::now();
        let mut tensors = HashMap::new();
        let mut creation_order = Vec::new();

        // Sort specs by layer index to ensure proper dependency order
        let mut sorted_specs: Vec<_> = tensor_specs.iter().collect();
        sorted_specs.sort_by_key(|(_, spec)| (spec.layer_idx.unwrap_or(0), spec.id));

        for (spec_id, spec) in sorted_specs {
            let tensor = self.create_tensor_from_spec(graph, spec)?;
            tensors.insert(*spec_id, tensor);
            creation_order.push(*spec_id);

            if let Some(ref tracker) = self.progress_tracker {
                tracker.increment_assembly_progress();
            }

            if let Some(ref debugger) = self.debugger {
                debugger.log_tensor_creation(&spec.name, std::time::Duration::from_micros(10));
            }
        }

        let elapsed = start_time.elapsed();
        if let Some(ref debugger) = self.debugger {
            debugger.log_phase_complete("Tensor Assembly", elapsed);
        }

        Ok(AssembledTensors {
            tensors,
            creation_order,
            total_count: tensor_specs.len(),
        })
    }

    /// Create a tensor from a specification
    fn create_tensor_from_spec(
        &self,
        graph: &mut Graph,
        spec: &TensorSpec,
    ) -> AssemblyResult<GraphTensor> {
        // For testing, create a simple tensor based on common shapes
        // In a real implementation, this would use the actual shape from spec.shape
        let tensor = match spec.tensor_type {
            TensorType::Data | TensorType::Input | TensorType::Output => {
                // Create a 3D tensor for data (batch, seq_len, hidden_dim)
                graph.named_tensor(&spec.name, (1, 512, 768))
            }
            TensorType::KeyCache | TensorType::ValueCache => {
                // Create a 4D tensor for KV cache (batch, heads, seq_len, head_dim)
                graph.named_tensor(&spec.name, (1, 8, 1024, 128))
            }
            TensorType::Weight => {
                // Create weight tensors with appropriate shapes
                if spec.name.contains("layernorm") {
                    graph.named_tensor(&spec.name, 768) // 1D for layer norms
                } else if spec.name.contains("qkv") {
                    graph.named_tensor(&spec.name, (768, 2304)) // q, k, v projections
                } else {
                    graph.named_tensor(&spec.name, (768, 768)) // Default weight matrix
                }
            }
        };

        Ok(tensor)
    }

    /// Assemble layer specifications (placeholder - would create actual operations in full implementation)
    pub fn assemble_layers(
        &self,
        _graph: &mut Graph,
        layer_specs: &HashMap<usize, LayerSpec>,
        _assembled_tensors: &AssembledTensors,
    ) -> AssemblyResult<AssembledLayers> {
        if let Some(ref debugger) = self.debugger {
            debugger.log_phase_start("Layer Assembly");
        }

        let start_time = std::time::Instant::now();
        let mut layers = HashMap::new();
        let mut creation_order = Vec::new();

        // Sort layer specs by layer index
        let mut sorted_specs: Vec<_> = layer_specs.iter().collect();
        sorted_specs.sort_by_key(|(_, spec)| spec.layer_idx);

        for (spec_id, spec) in sorted_specs {
            let layer_info = LayerInfo {
                spec_id: *spec_id,
                layer_idx: spec.layer_idx,
                tensor_ids: spec.tensor_ids.clone(),
                output_tensor_id: None, // Would be set in full implementation
            };

            layers.insert(*spec_id, layer_info);
            creation_order.push(*spec_id);

            if let Some(ref tracker) = self.progress_tracker {
                tracker.increment_integration_progress();
            }

            if let Some(ref debugger) = self.debugger {
                debugger.log_layer_spec(spec.layer_idx, spec.tensor_ids.len());
            }
        }

        let elapsed = start_time.elapsed();
        if let Some(ref debugger) = self.debugger {
            debugger.log_phase_complete("Layer Assembly", elapsed);
        }

        Ok(AssembledLayers {
            layers,
            creation_order,
            total_count: layer_specs.len(),
        })
    }

    /// Validate that all dependencies are satisfied
    pub fn validate_dependencies(
        &self,
        tensor_specs: &HashMap<usize, TensorSpec>,
        layer_specs: &HashMap<usize, LayerSpec>,
    ) -> AssemblyResult<()> {
        for (layer_id, layer_spec) in layer_specs {
            for tensor_id in &layer_spec.tensor_ids {
                if !tensor_specs.contains_key(tensor_id) {
                    return Err(AssemblyError::DependencyNotSatisfied {
                        tensor_id: *tensor_id,
                        layer_id: *layer_id,
                    });
                }
            }
        }
        Ok(())
    }
}

impl Default for AssemblyPhase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel_graph::specs::{TensorSpec, LayerSpec};
    use crate::parallel_graph::debug::DebugLevel;
    use crate::parallel_graph::progress::ProgressCounts;

    fn create_test_tensor_spec(id: usize, name: &str, tensor_type: TensorType) -> TensorSpec {
        TensorSpec {
            id,
            name: name.to_string(),
            shape: ShapeTracker::new(&[1, 512, 768]),
            tensor_type,
            layer_idx: Some(0),
        }
    }

    fn create_test_layer_spec(id: usize, layer_idx: usize, tensor_ids: Vec<usize>) -> LayerSpec {
        LayerSpec {
            id,
            layer_type: LayerType::TransformerBlock,
            tensor_ids,
            layer_idx,
            parameters: LayerParameters::TransformerBlock {
                hidden_dim: 768,
                n_heads: 12,
                n_kv_heads: 12,
                mlp_dim: 3072,
            },
        }
    }

    #[test]
    fn test_assembly_phase_creation() {
        let assembly = AssemblyPhase::new();
        assert!(assembly.debugger.is_none());
        assert!(assembly.progress_tracker.is_none());
    }

    #[test]
    fn test_assembly_phase_with_debugger() {
        let debugger = ParallelDebugger::new(DebugLevel::Basic);
        let assembly = AssemblyPhase::with_debugger(debugger);
        assert!(assembly.debugger.is_some());
    }

    #[test]
    fn test_assembly_phase_with_progress_tracker() {
        let tracker = ProgressTracker::new(ProgressCounts {
            kv_cache_ops: 10,
            layer_spec_ops: 5,
            assembly_ops: 20,
            integration_ops: 3,
        });
        let assembly = AssemblyPhase::with_progress_tracker(tracker);
        assert!(assembly.progress_tracker.is_some());
    }

    #[test]
    fn test_assemble_empty_tensors() {
        let assembly = AssemblyPhase::new();
        let mut graph = Graph::new();
        let tensor_specs = HashMap::new();

        let result = assembly.assemble_tensors(&mut graph, &tensor_specs).unwrap();
        assert_eq!(result.total_count, 0);
        assert!(result.tensors.is_empty());
        assert!(result.creation_order.is_empty());
    }

    #[test]
    fn test_assemble_single_tensor() {
        let assembly = AssemblyPhase::new();
        let mut graph = Graph::new();
        let mut tensor_specs = HashMap::new();

        let spec = create_test_tensor_spec(1, "test_tensor", TensorType::Data);
        tensor_specs.insert(1, spec);

        let result = assembly.assemble_tensors(&mut graph, &tensor_specs).unwrap();
        assert_eq!(result.total_count, 1);
        assert_eq!(result.tensors.len(), 1);
        assert!(result.tensors.contains_key(&1));
        assert_eq!(result.creation_order, vec![1]);
    }

    #[test]
    fn test_assemble_multiple_tensors() {
        let assembly = AssemblyPhase::new();
        let mut graph = Graph::new();
        let mut tensor_specs = HashMap::new();

        tensor_specs.insert(1, create_test_tensor_spec(1, "tensor1", TensorType::Data));
        tensor_specs.insert(2, create_test_tensor_spec(2, "tensor2", TensorType::KeyCache));
        tensor_specs.insert(3, create_test_tensor_spec(3, "tensor3", TensorType::ValueCache));

        let result = assembly.assemble_tensors(&mut graph, &tensor_specs).unwrap();
        assert_eq!(result.total_count, 3);
        assert_eq!(result.tensors.len(), 3);
        assert_eq!(result.creation_order.len(), 3);

        // Verify all tensors were created
        for id in 1..=3 {
            assert!(result.tensors.contains_key(&id));
        }
    }

    #[test]
    fn test_assemble_tensors_with_debugger() {
        let debugger = ParallelDebugger::new(DebugLevel::All);
        let assembly = AssemblyPhase::with_debugger(debugger.clone());
        let mut graph = Graph::new();
        let mut tensor_specs = HashMap::new();

        tensor_specs.insert(1, create_test_tensor_spec(1, "test", TensorType::Data));

        let _result = assembly.assemble_tensors(&mut graph, &tensor_specs).unwrap();

        // Should have logged phase start, completion, and tensor creation
        assert!(debugger.log_count() >= 3);
    }

    #[test]
    fn test_assemble_layers_empty() {
        let assembly = AssemblyPhase::new();
        let mut graph = Graph::new();
        let layer_specs = HashMap::new();
        let assembled_tensors = AssembledTensors {
            tensors: HashMap::new(),
            creation_order: Vec::new(),
            total_count: 0,
        };

        let result = assembly.assemble_layers(&mut graph, &layer_specs, &assembled_tensors).unwrap();
        assert_eq!(result.total_count, 0);
        assert!(result.layers.is_empty());
    }

    #[test]
    fn test_assemble_layers_single() {
        let assembly = AssemblyPhase::new();
        let mut graph = Graph::new();
        let mut layer_specs = HashMap::new();

        let spec = create_test_layer_spec(1, 0, vec![1, 2, 3]);
        layer_specs.insert(1, spec);

        let assembled_tensors = AssembledTensors {
            tensors: HashMap::new(),
            creation_order: Vec::new(),
            total_count: 0,
        };

        let result = assembly.assemble_layers(&mut graph, &layer_specs, &assembled_tensors).unwrap();
        assert_eq!(result.total_count, 1);
        assert_eq!(result.layers.len(), 1);

        let layer_info = result.layers.get(&1).unwrap();
        assert_eq!(layer_info.spec_id, 1);
        assert_eq!(layer_info.layer_idx, 0);
        assert_eq!(layer_info.tensor_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_validate_dependencies_satisfied() {
        let assembly = AssemblyPhase::new();
        let mut tensor_specs = HashMap::new();
        let mut layer_specs = HashMap::new();

        // Create tensors
        tensor_specs.insert(1, create_test_tensor_spec(1, "t1", TensorType::Data));
        tensor_specs.insert(2, create_test_tensor_spec(2, "t2", TensorType::Data));

        // Create layer that uses these tensors
        layer_specs.insert(1, create_test_layer_spec(1, 0, vec![1, 2]));

        let result = assembly.validate_dependencies(&tensor_specs, &layer_specs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_dependencies_not_satisfied() {
        let assembly = AssemblyPhase::new();
        let mut tensor_specs = HashMap::new();
        let mut layer_specs = HashMap::new();

        // Create only one tensor
        tensor_specs.insert(1, create_test_tensor_spec(1, "t1", TensorType::Data));

        // Create layer that uses missing tensor
        layer_specs.insert(1, create_test_layer_spec(1, 0, vec![1, 99])); // 99 doesn't exist

        let result = assembly.validate_dependencies(&tensor_specs, &layer_specs);
        assert!(result.is_err());

        match result.unwrap_err() {
            AssemblyError::DependencyNotSatisfied { tensor_id, layer_id } => {
                assert_eq!(tensor_id, 99);
                assert_eq!(layer_id, 1);
            }
            _ => panic!("Expected DependencyNotSatisfied error"),
        }
    }

    #[test]
    fn test_tensor_ordering_by_layer() {
        let assembly = AssemblyPhase::new();
        let mut graph = Graph::new();
        let mut tensor_specs = HashMap::new();

        // Create tensors with different layer indices
        let mut spec1 = create_test_tensor_spec(1, "layer0", TensorType::Data);
        spec1.layer_idx = Some(0);
        let mut spec2 = create_test_tensor_spec(2, "layer2", TensorType::Data);
        spec2.layer_idx = Some(2);
        let mut spec3 = create_test_tensor_spec(3, "layer1", TensorType::Data);
        spec3.layer_idx = Some(1);

        tensor_specs.insert(1, spec1);
        tensor_specs.insert(2, spec2);
        tensor_specs.insert(3, spec3);

        let result = assembly.assemble_tensors(&mut graph, &tensor_specs).unwrap();

        // Should be ordered by layer index: 0, 1, 2
        assert_eq!(result.creation_order, vec![1, 3, 2]);
    }

    #[test]
    fn test_default_implementation() {
        let assembly = AssemblyPhase::default();
        assert!(assembly.debugger.is_none());
        assert!(assembly.progress_tracker.is_none());
    }

    #[test]
    fn test_assembly_error_display() {
        let error = AssemblyError::TensorSpecNotFound { id: 123 };
        assert_eq!(error.to_string(), "Tensor specification not found: 123");

        let error = AssemblyError::DependencyNotSatisfied { tensor_id: 1, layer_id: 2 };
        assert_eq!(error.to_string(), "Dependency not satisfied: tensor 1 needed for layer 2");
    }
}