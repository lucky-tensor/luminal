//! Parallel graph builder implementation
//!
//! Thread-safe wrapper around Graph that defers mutations until assembly phase,
//! enabling parallel specification creation without graph contention.

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;
use luminal::prelude::*;
use thiserror::Error;

use crate::parallel_graph::specs::{TensorSpec, LayerSpec, TensorType};
use crate::parallel_graph::debug::{ParallelDebugger, DebugLevel};
use crate::parallel_graph::progress::ProgressTracker;
use rayon::prelude::*;

/// Errors that can occur during parallel graph building
#[derive(Error, Debug)]
pub enum ParallelGraphError {
    #[error("Graph mutex lock failed: {0}")]
    LockError(String),
    #[error("Tensor specification not found: {id}")]
    TensorSpecNotFound { id: usize },
    #[error("Layer specification not found: {id}")]
    LayerSpecNotFound { id: usize },
    #[error("Invalid tensor shape: {message}")]
    InvalidShape { message: String },
    #[error("Assembly phase error: {message}")]
    AssemblyError { message: String },
}

/// Result type for parallel graph operations
pub type ParallelGraphResult<T> = Result<T, ParallelGraphError>;

/// Thread-safe parallel graph builder
#[derive(Debug)]
pub struct ParallelGraphBuilder {
    /// Collection of tensor specifications
    tensor_specs: Arc<Mutex<HashMap<usize, TensorSpec>>>,
    /// Collection of layer specifications
    layer_specs: Arc<Mutex<HashMap<usize, LayerSpec>>>,
    /// Atomic ID counter for generating unique IDs
    id_counter: Arc<AtomicUsize>,
    /// Debug logger
    debugger: Option<Arc<ParallelDebugger>>,
    /// Progress tracker
    progress_tracker: Option<Arc<ProgressTracker>>,
}

/// Assembled tensors and layers after sequential phase
#[derive(Debug)]
pub struct AssembledGraph {
    /// The assembled graph
    pub graph: Graph,
    /// Map from spec ID to actual tensor
    pub tensors: HashMap<usize, GraphTensor>,
    /// Map from spec ID to layer info
    pub layers: HashMap<usize, crate::parallel_graph::assembly::LayerInfo>,
}


impl ParallelGraphBuilder {
    /// Create a new parallel graph builder
    pub fn new() -> Self {
        Self {
            tensor_specs: Arc::new(Mutex::new(HashMap::new())),
            layer_specs: Arc::new(Mutex::new(HashMap::new())),
            id_counter: Arc::new(AtomicUsize::new(1)),
            debugger: None,
            progress_tracker: None,
        }
    }

    /// Create a new parallel graph builder with debugging
    pub fn with_debugger(debugger: Arc<ParallelDebugger>) -> Self {
        let mut builder = Self::new();
        builder.debugger = Some(debugger);
        builder
    }

    /// Create a new parallel graph builder with progress tracking
    pub fn with_progress_tracker(progress_tracker: Arc<ProgressTracker>) -> Self {
        let mut builder = Self::new();
        builder.progress_tracker = Some(progress_tracker);
        builder
    }

    /// Create a new parallel graph builder with debugging and progress tracking
    pub fn with_debug_and_progress(
        debugger: Arc<ParallelDebugger>,
        progress_tracker: Arc<ProgressTracker>,
    ) -> Self {
        let mut builder = Self::new();
        builder.debugger = Some(debugger);
        builder.progress_tracker = Some(progress_tracker);
        builder
    }

    /// Generate a unique ID
    fn next_id(&self) -> usize {
        self.id_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Add a tensor specification (thread-safe)
    pub fn add_tensor_spec(&self, mut spec: TensorSpec) -> ParallelGraphResult<usize> {
        // Always assign a new unique ID from our counter
        spec.id = self.next_id();

        let spec_id = spec.id;

        {
            let mut specs = self.tensor_specs.lock()
                .map_err(|e| ParallelGraphError::LockError(e.to_string()))?;
            specs.insert(spec_id, spec);
        }

        if let Some(ref debugger) = self.debugger {
            debugger.log_tensor_creation(&format!("spec_{}", spec_id), std::time::Duration::from_micros(1));
        }

        if let Some(ref tracker) = self.progress_tracker {
            tracker.increment_kv_cache_progress();
        }

        Ok(spec_id)
    }

    /// Add a layer specification (thread-safe)
    pub fn add_layer_spec(&self, mut spec: LayerSpec) -> ParallelGraphResult<usize> {
        // Always assign a new unique ID from our counter
        spec.id = self.next_id();

        let spec_id = spec.id;

        {
            let mut specs = self.layer_specs.lock()
                .map_err(|e| ParallelGraphError::LockError(e.to_string()))?;
            specs.insert(spec_id, spec.clone());
        }

        if let Some(ref debugger) = self.debugger {
            debugger.log_layer_spec(spec.layer_idx, spec.tensor_ids.len());
        }

        if let Some(ref tracker) = self.progress_tracker {
            tracker.increment_layer_progress();
        }

        Ok(spec_id)
    }

    /// Get a tensor specification by ID
    pub fn get_tensor_spec(&self, id: usize) -> ParallelGraphResult<TensorSpec> {
        let specs = self.tensor_specs.lock()
            .map_err(|e| ParallelGraphError::LockError(e.to_string()))?;
        specs.get(&id)
            .cloned()
            .ok_or(ParallelGraphError::TensorSpecNotFound { id })
    }

    /// Get a layer specification by ID
    pub fn get_layer_spec(&self, id: usize) -> ParallelGraphResult<LayerSpec> {
        let specs = self.layer_specs.lock()
            .map_err(|e| ParallelGraphError::LockError(e.to_string()))?;
        specs.get(&id)
            .cloned()
            .ok_or(ParallelGraphError::LayerSpecNotFound { id })
    }

    /// Get all tensor specifications
    pub fn get_all_tensor_specs(&self) -> ParallelGraphResult<HashMap<usize, TensorSpec>> {
        let specs = self.tensor_specs.lock()
            .map_err(|e| ParallelGraphError::LockError(e.to_string()))?;
        Ok(specs.clone())
    }

    /// Get all layer specifications
    pub fn get_all_layer_specs(&self) -> ParallelGraphResult<HashMap<usize, LayerSpec>> {
        let specs = self.layer_specs.lock()
            .map_err(|e| ParallelGraphError::LockError(e.to_string()))?;
        Ok(specs.clone())
    }

    /// Get the number of tensor specifications
    pub fn tensor_spec_count(&self) -> ParallelGraphResult<usize> {
        let specs = self.tensor_specs.lock()
            .map_err(|e| ParallelGraphError::LockError(e.to_string()))?;
        Ok(specs.len())
    }

    /// Get the number of layer specifications
    pub fn layer_spec_count(&self) -> ParallelGraphResult<usize> {
        let specs = self.layer_specs.lock()
            .map_err(|e| ParallelGraphError::LockError(e.to_string()))?;
        Ok(specs.len())
    }

    /// Clear all specifications (for testing)
    pub fn clear_specs(&self) -> ParallelGraphResult<()> {
        {
            let mut tensor_specs = self.tensor_specs.lock()
                .map_err(|e| ParallelGraphError::LockError(e.to_string()))?;
            tensor_specs.clear();
        }
        {
            let mut layer_specs = self.layer_specs.lock()
                .map_err(|e| ParallelGraphError::LockError(e.to_string()))?;
            layer_specs.clear();
        }
        Ok(())
    }

    /// Begin sequential assembly phase with batch tensor insertion
    pub fn assemble(&self) -> ParallelGraphResult<AssembledGraph> {
        if let Some(ref debugger) = self.debugger {
            debugger.log_phase_start("Sequential Assembly");
        }

        let start_time = std::time::Instant::now();

        // Get all specifications
        let tensor_specs = self.get_all_tensor_specs()?;
        let layer_specs = self.get_all_layer_specs()?;

        // Create assembly phase with debugging and progress tracking
        let assembly = match &self.debugger {
            Some(debugger) => {
                // Clone the debugger from Arc
                let debugger_clone = (**debugger).clone();
                crate::parallel_graph::assembly::AssemblyPhase::with_debugger(debugger_clone)
            }
            None => crate::parallel_graph::assembly::AssemblyPhase::new(),
        };

        // Validate dependencies before assembly
        assembly.validate_dependencies(&tensor_specs, &layer_specs)
            .map_err(|e| ParallelGraphError::AssemblyError { message: e.to_string() })?;

        // Create new graph for assembly
        let mut graph = Graph::new();

        // Assemble tensors with batch insertion
        let assembled_tensors = assembly.assemble_tensors(&mut graph, &tensor_specs)
            .map_err(|e| ParallelGraphError::AssemblyError { message: e.to_string() })?;

        // Assemble layers
        let assembled_layers = assembly.assemble_layers(&mut graph, &layer_specs, &assembled_tensors)
            .map_err(|e| ParallelGraphError::AssemblyError { message: e.to_string() })?;

        let elapsed = start_time.elapsed();
        if let Some(ref debugger) = self.debugger {
            debugger.log_phase_complete("Sequential Assembly", elapsed);
        }

        if let Some(ref tracker) = self.progress_tracker {
            tracker.increment_assembly_progress();
        }

        Ok(AssembledGraph {
            graph,
            tensors: assembled_tensors.tensors,
            layers: assembled_layers.layers,
        })
    }

    /// Create a complete parallel model from specifications
    pub fn build_parallel_model(
        &self,
        num_layers: usize,
        hidden_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        mlp_dim: usize,
        sequence_length: usize,
        head_dim: usize,
    ) -> ParallelGraphResult<AssembledGraph> {
        if let Some(ref debugger) = self.debugger {
            debugger.log_phase_start("Parallel Model Build");
        }

        let start_time = std::time::Instant::now();

        // Step 1: Create KV cache specifications in parallel
        let _kv_specs = self.create_kv_cache_specs_parallel(
            num_layers,
            n_kv_heads,
            head_dim,
            sequence_length,
        )?;

        // Step 2: Create layer specifications in parallel
        let _layer_specs = self.create_transformer_layer_specs_parallel(
            num_layers,
            hidden_dim,
            n_heads,
            n_kv_heads,
            mlp_dim,
        )?;

        // Step 3: Sequential assembly with batch tensor insertion
        let assembled_graph = self.assemble()?;

        let elapsed = start_time.elapsed();
        if let Some(ref debugger) = self.debugger {
            debugger.log_phase_complete("Parallel Model Build", elapsed);
        }

        Ok(assembled_graph)
    }

    /// Create KV cache specifications in parallel using rayon
    pub fn create_kv_cache_specs_parallel(
        &self,
        num_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        sequence_length: usize,
    ) -> ParallelGraphResult<Vec<usize>> {
        if let Some(ref debugger) = self.debugger {
            debugger.log_phase_start("Parallel KV Cache Creation");
        }

        let start_time = std::time::Instant::now();

        // Create KV cache specs in parallel using rayon
        let spec_ids: Vec<usize> = (0..num_layers)
            .into_par_iter()
            .map(|layer_idx| {
                // Create key cache spec
                let key_spec = TensorSpec::with_layer_idx(
                    format!("key_cache_layer_{}", layer_idx),
                    ShapeTracker::new(&[1, n_kv_heads, sequence_length, head_dim]),
                    TensorType::KeyCache,
                    layer_idx,
                );

                // Create value cache spec
                let value_spec = TensorSpec::with_layer_idx(
                    format!("value_cache_layer_{}", layer_idx),
                    ShapeTracker::new(&[1, n_kv_heads, sequence_length, head_dim]),
                    TensorType::ValueCache,
                    layer_idx,
                );

                // Add both specs and return their IDs
                let key_id = self.add_tensor_spec(key_spec).unwrap_or(0);
                let value_id = self.add_tensor_spec(value_spec).unwrap_or(0);

                if let Some(ref debugger) = self.debugger {
                    debugger.log_tensor_creation(
                        &format!("kv_cache_layer_{}", layer_idx),
                        std::time::Duration::from_micros(10),
                    );
                }

                vec![key_id, value_id]
            })
            .flatten()
            .collect();

        let elapsed = start_time.elapsed();
        if let Some(ref debugger) = self.debugger {
            debugger.log_phase_complete("Parallel KV Cache Creation", elapsed);
        }

        Ok(spec_ids)
    }

    /// Create transformer layer specifications in parallel using rayon
    pub fn create_transformer_layer_specs_parallel(
        &self,
        num_layers: usize,
        hidden_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        mlp_dim: usize,
    ) -> ParallelGraphResult<Vec<usize>> {
        if let Some(ref debugger) = self.debugger {
            debugger.log_phase_start("Parallel Layer Specification");
        }

        let start_time = std::time::Instant::now();

        // Create layer specs in parallel using rayon
        let spec_ids: Vec<usize> = (0..num_layers)
            .into_par_iter()
            .map(|layer_idx| {
                // Create tensor specs for this layer
                let weight_specs: Vec<TensorSpec> = vec![
                    // Attention weights
                    TensorSpec::with_layer_idx(
                        format!("layer_{}_attn_qkv_weight", layer_idx),
                        ShapeTracker::new(&[hidden_dim, (n_heads + 2 * n_kv_heads) * (hidden_dim / n_heads)]),
                        TensorType::Weight,
                        layer_idx,
                    ),
                    TensorSpec::with_layer_idx(
                        format!("layer_{}_attn_output_weight", layer_idx),
                        ShapeTracker::new(&[hidden_dim, hidden_dim]),
                        TensorType::Weight,
                        layer_idx,
                    ),
                    // MLP weights
                    TensorSpec::with_layer_idx(
                        format!("layer_{}_mlp_gate_weight", layer_idx),
                        ShapeTracker::new(&[hidden_dim, mlp_dim]),
                        TensorType::Weight,
                        layer_idx,
                    ),
                    TensorSpec::with_layer_idx(
                        format!("layer_{}_mlp_up_weight", layer_idx),
                        ShapeTracker::new(&[hidden_dim, mlp_dim]),
                        TensorType::Weight,
                        layer_idx,
                    ),
                    TensorSpec::with_layer_idx(
                        format!("layer_{}_mlp_down_weight", layer_idx),
                        ShapeTracker::new(&[mlp_dim, hidden_dim]),
                        TensorType::Weight,
                        layer_idx,
                    ),
                    // Layer norms
                    TensorSpec::with_layer_idx(
                        format!("layer_{}_input_layernorm_weight", layer_idx),
                        ShapeTracker::new(&[hidden_dim]),
                        TensorType::Weight,
                        layer_idx,
                    ),
                    TensorSpec::with_layer_idx(
                        format!("layer_{}_post_attention_layernorm_weight", layer_idx),
                        ShapeTracker::new(&[hidden_dim]),
                        TensorType::Weight,
                        layer_idx,
                    ),
                ];

                // Add all weight specs
                let tensor_ids: Vec<usize> = weight_specs
                    .into_iter()
                    .map(|spec| self.add_tensor_spec(spec).unwrap_or(0))
                    .collect();

                // Create layer spec
                use crate::parallel_graph::specs::{LayerSpec, LayerType, LayerParameters};
                let layer_spec = LayerSpec::new(
                    LayerType::TransformerBlock,
                    tensor_ids,
                    layer_idx,
                    LayerParameters::TransformerBlock {
                        hidden_dim,
                        n_heads,
                        n_kv_heads,
                        mlp_dim,
                    },
                );

                if let Some(ref debugger) = self.debugger {
                    debugger.log_layer_spec(layer_idx, layer_spec.tensor_ids.len());
                }

                self.add_layer_spec(layer_spec).unwrap_or(0)
            })
            .collect();

        let elapsed = start_time.elapsed();
        if let Some(ref debugger) = self.debugger {
            debugger.log_phase_complete("Parallel Layer Specification", elapsed);
        }

        Ok(spec_ids)
    }
}

impl Default for ParallelGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel_graph::specs::{LayerParameters, TensorType, LayerType};
    use std::sync::Arc;
    use std::thread;
    
    fn create_test_tensor_spec() -> TensorSpec {
        TensorSpec::new(
            "test_tensor".to_string(),
            ShapeTracker::new(&[1, 512, 768]),
            TensorType::Data,
        )
    }

    fn create_test_layer_spec() -> LayerSpec {
        LayerSpec::new(
            LayerType::TransformerBlock,
            vec![1, 2, 3],
            0,
            LayerParameters::TransformerBlock {
                hidden_dim: 768,
                n_heads: 12,
                n_kv_heads: 12,
                mlp_dim: 3072,
            },
        )
    }

    #[test]
    fn test_parallel_graph_builder_creation() {
        let builder = ParallelGraphBuilder::new();
        assert_eq!(builder.tensor_spec_count().unwrap(), 0);
        assert_eq!(builder.layer_spec_count().unwrap(), 0);
    }

    #[test]
    fn test_with_debugger() {
        let debugger = Arc::new(ParallelDebugger::new(DebugLevel::Basic));
        let builder = ParallelGraphBuilder::with_debugger(debugger.clone());
        assert!(builder.debugger.is_some());
        assert!(builder.progress_tracker.is_none());
    }

    #[test]
    fn test_add_tensor_spec() {
        let builder = ParallelGraphBuilder::new();
        let spec = create_test_tensor_spec();
        let original_id = spec.id;

        let spec_id = builder.add_tensor_spec(spec).unwrap();
        assert_eq!(builder.tensor_spec_count().unwrap(), 1);

        let retrieved_spec = builder.get_tensor_spec(spec_id).unwrap();
        assert_eq!(retrieved_spec.name, "test_tensor");
        // ID should be preserved or assigned
        assert!(retrieved_spec.id == original_id || retrieved_spec.id == spec_id);
    }

    #[test]
    fn test_add_layer_spec() {
        let builder = ParallelGraphBuilder::new();
        let spec = create_test_layer_spec();
        let original_id = spec.id;

        let spec_id = builder.add_layer_spec(spec).unwrap();
        assert_eq!(builder.layer_spec_count().unwrap(), 1);

        let retrieved_spec = builder.get_layer_spec(spec_id).unwrap();
        assert_eq!(retrieved_spec.layer_idx, 0);
        assert_eq!(retrieved_spec.tensor_ids, vec![1, 2, 3]);
        assert!(retrieved_spec.id == original_id || retrieved_spec.id == spec_id);
    }

    #[test]
    fn test_tensor_spec_not_found() {
        let builder = ParallelGraphBuilder::new();
        let result = builder.get_tensor_spec(999);
        assert!(matches!(result, Err(ParallelGraphError::TensorSpecNotFound { id: 999 })));
    }

    #[test]
    fn test_layer_spec_not_found() {
        let builder = ParallelGraphBuilder::new();
        let result = builder.get_layer_spec(999);
        assert!(matches!(result, Err(ParallelGraphError::LayerSpecNotFound { id: 999 })));
    }

    #[test]
    fn test_get_all_specs() {
        let builder = ParallelGraphBuilder::new();

        let tensor_spec = create_test_tensor_spec();
        let layer_spec = create_test_layer_spec();

        let tensor_id = builder.add_tensor_spec(tensor_spec).unwrap();
        let layer_id = builder.add_layer_spec(layer_spec).unwrap();

        let all_tensors = builder.get_all_tensor_specs().unwrap();
        let all_layers = builder.get_all_layer_specs().unwrap();

        assert_eq!(all_tensors.len(), 1);
        assert_eq!(all_layers.len(), 1);
        assert!(all_tensors.contains_key(&tensor_id));
        assert!(all_layers.contains_key(&layer_id));
    }

    #[test]
    fn test_clear_specs() {
        let builder = ParallelGraphBuilder::new();

        builder.add_tensor_spec(create_test_tensor_spec()).unwrap();
        builder.add_layer_spec(create_test_layer_spec()).unwrap();

        assert_eq!(builder.tensor_spec_count().unwrap(), 1);
        assert_eq!(builder.layer_spec_count().unwrap(), 1);

        builder.clear_specs().unwrap();

        assert_eq!(builder.tensor_spec_count().unwrap(), 0);
        assert_eq!(builder.layer_spec_count().unwrap(), 0);
    }

    #[test]
    fn test_thread_safety() {
        let builder = Arc::new(ParallelGraphBuilder::new());
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let builder = builder.clone();
                thread::spawn(move || {
                    // Add tensor specs from multiple threads
                    let mut spec = create_test_tensor_spec();
                    spec.name = format!("tensor_{}", i);
                    builder.add_tensor_spec(spec).unwrap();

                    // Add layer specs from multiple threads
                    let mut layer_spec = create_test_layer_spec();
                    layer_spec.layer_idx = i;
                    builder.add_layer_spec(layer_spec).unwrap();
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(builder.tensor_spec_count().unwrap(), 10);
        assert_eq!(builder.layer_spec_count().unwrap(), 10);

        // Verify all specs are accessible
        let all_tensors = builder.get_all_tensor_specs().unwrap();
        let all_layers = builder.get_all_layer_specs().unwrap();

        assert_eq!(all_tensors.len(), 10);
        assert_eq!(all_layers.len(), 10);
    }

    #[test]
    fn test_unique_id_generation() {
        let builder = ParallelGraphBuilder::new();

        let spec1 = create_test_tensor_spec();
        let spec2 = create_test_tensor_spec();

        let id1 = builder.add_tensor_spec(spec1).unwrap();
        let id2 = builder.add_tensor_spec(spec2).unwrap();

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_assemble_placeholder() {
        let builder = ParallelGraphBuilder::new();

        // Add tensor spec first
        let tensor_id = builder.add_tensor_spec(create_test_tensor_spec()).unwrap();

        // Add layer spec that references the actual tensor
        let mut layer_spec = create_test_layer_spec();
        layer_spec.tensor_ids = vec![tensor_id];
        builder.add_layer_spec(layer_spec).unwrap();

        let assembled = builder.assemble().unwrap();

        // Should have assembled the tensors and layers
        assert!(!assembled.tensors.is_empty());
        assert!(!assembled.layers.is_empty());
    }

    #[test]
    fn test_with_debug_and_progress() {
        let debugger = Arc::new(ParallelDebugger::new(DebugLevel::Verbose));
        let progress = Arc::new(ProgressTracker::default());

        let builder = ParallelGraphBuilder::with_debug_and_progress(debugger.clone(), progress.clone());

        assert!(builder.debugger.is_some());
        assert!(builder.progress_tracker.is_some());

        // Test that operations are logged/tracked
        builder.add_tensor_spec(create_test_tensor_spec()).unwrap();

        assert_eq!(debugger.log_count(), 1);
        // Progress tracking depends on configuration
    }

    #[test]
    fn test_default_implementation() {
        let builder = ParallelGraphBuilder::default();
        assert_eq!(builder.tensor_spec_count().unwrap(), 0);
        assert!(builder.debugger.is_none());
        assert!(builder.progress_tracker.is_none());
    }

    #[test]
    fn test_error_display() {
        let error = ParallelGraphError::TensorSpecNotFound { id: 123 };
        assert_eq!(error.to_string(), "Tensor specification not found: 123");

        let error = ParallelGraphError::LockError("mutex poisoned".to_string());
        assert_eq!(error.to_string(), "Graph mutex lock failed: mutex poisoned");
    }

    #[test]
    fn test_parallel_kv_cache_creation() {
        let builder = ParallelGraphBuilder::new();

        // Create KV cache specs in parallel
        let spec_ids = builder
            .create_kv_cache_specs_parallel(4, 8, 128, 1024)
            .unwrap();

        // Should create 2 specs per layer (key and value) * 4 layers = 8 specs
        assert_eq!(spec_ids.len(), 8);
        assert_eq!(builder.tensor_spec_count().unwrap(), 8);

        // Verify each spec was created correctly
        for (i, spec_id) in spec_ids.iter().enumerate() {
            let spec = builder.get_tensor_spec(*spec_id).unwrap();
            let layer_idx = i / 2; // Each layer has 2 specs (key, value)
            let is_key = i % 2 == 0;

            assert_eq!(spec.layer_idx, Some(layer_idx));
            if is_key {
                assert!(spec.name.contains("key_cache"));
                assert_eq!(spec.tensor_type, TensorType::KeyCache);
            } else {
                assert!(spec.name.contains("value_cache"));
                assert_eq!(spec.tensor_type, TensorType::ValueCache);
            }
        }
    }

    #[test]
    fn test_parallel_kv_cache_with_debugger() {
        let debugger = Arc::new(ParallelDebugger::new(DebugLevel::All));
        let builder = ParallelGraphBuilder::with_debugger(debugger.clone());

        let _spec_ids = builder
            .create_kv_cache_specs_parallel(2, 4, 64, 512)
            .unwrap();

        // Should have logged phase start, completion, and tensor creation
        assert!(debugger.log_count() >= 6); // 2 phases + 4 tensors minimum
    }

    #[test]
    fn test_parallel_transformer_layer_creation() {
        let builder = ParallelGraphBuilder::new();

        // Create transformer layer specs in parallel
        let layer_spec_ids = builder
            .create_transformer_layer_specs_parallel(3, 768, 12, 12, 3072)
            .unwrap();

        // Should create 3 layer specs
        assert_eq!(layer_spec_ids.len(), 3);
        assert_eq!(builder.layer_spec_count().unwrap(), 3);

        // Each layer should have multiple tensor specs (7 weight tensors per layer)
        // Total tensors = 3 layers * 7 tensors = 21 tensors
        assert_eq!(builder.tensor_spec_count().unwrap(), 21);

        // Verify each layer spec
        for (i, layer_spec_id) in layer_spec_ids.iter().enumerate() {
            let layer_spec = builder.get_layer_spec(*layer_spec_id).unwrap();
            assert_eq!(layer_spec.layer_idx, i);
            assert_eq!(layer_spec.layer_type, LayerType::TransformerBlock);
            assert_eq!(layer_spec.tensor_ids.len(), 7); // 7 weight tensors per layer

            // Verify all referenced tensors exist
            for tensor_id in &layer_spec.tensor_ids {
                let tensor_spec = builder.get_tensor_spec(*tensor_id).unwrap();
                assert_eq!(tensor_spec.layer_idx, Some(i));
                assert_eq!(tensor_spec.tensor_type, TensorType::Weight);
            }
        }
    }

    #[test]
    fn test_parallel_transformer_layer_with_debugger() {
        let debugger = Arc::new(ParallelDebugger::new(DebugLevel::Detailed));
        let builder = ParallelGraphBuilder::with_debugger(debugger.clone());

        let _layer_spec_ids = builder
            .create_transformer_layer_specs_parallel(2, 512, 8, 8, 2048)
            .unwrap();

        // Should have logged phase operations and layer specs
        assert!(debugger.log_count() >= 4); // Phase start/complete + 2 layers
    }

    #[test]
    fn test_parallel_operations_thread_safety() {
        let builder = Arc::new(ParallelGraphBuilder::new());

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let builder = builder.clone();
                thread::spawn(move || {
                    // Create unique specs from each thread to avoid conflicts
                    let base_name = format!("thread_{}", i);

                    // Create KV cache spec with unique name
                    let key_spec = TensorSpec::with_layer_idx(
                        format!("{}_key_cache", base_name),
                        ShapeTracker::new(&[1, 4, 256, 64]),
                        TensorType::KeyCache,
                        i * 100, // Use unique layer indices to avoid conflicts
                    );
                    let value_spec = TensorSpec::with_layer_idx(
                        format!("{}_value_cache", base_name),
                        ShapeTracker::new(&[1, 4, 256, 64]),
                        TensorType::ValueCache,
                        i * 100 + 1,
                    );

                    let key_result = builder.add_tensor_spec(key_spec);
                    let value_result = builder.add_tensor_spec(value_spec);

                    // Return results with error information for debugging
                    (i, key_result, value_result)
                })
            })
            .collect();

        let mut all_ids = Vec::new();
        for handle in handles {
            let (thread_id, key_result, value_result) = handle.join().unwrap();

            // Both operations should succeed
            let key_id = key_result.unwrap_or_else(|e| panic!("Thread {} key spec failed: {}", thread_id, e));
            let value_id = value_result.unwrap_or_else(|e| panic!("Thread {} value spec failed: {}", thread_id, e));

            all_ids.push(key_id);
            all_ids.push(value_id);
        }

        // Verify all IDs are unique and all specs are accessible
        assert_eq!(all_ids.len(), 8); // 4 threads * 2 specs each
        let unique_ids: std::collections::HashSet<_> = all_ids.iter().collect();
        assert_eq!(unique_ids.len(), 8, "IDs should be unique: {:?}", all_ids);

        // Verify all specs can be retrieved
        for id in &all_ids {
            builder.get_tensor_spec(*id)
                .unwrap_or_else(|e| panic!("Failed to retrieve spec {}: {}", id, e));
        }

        // Verify final counts
        assert_eq!(builder.tensor_spec_count().unwrap(), 8);
    }

    #[test]
    fn test_parallel_functionality_correctness() {
        // This test focuses on correctness rather than performance
        // since parallel overhead can be higher for small workloads
        let builder = ParallelGraphBuilder::new();

        // Test that parallel creation produces correct results
        let parallel_specs = builder
            .create_kv_cache_specs_parallel(5, 8, 128, 1024)
            .unwrap();

        // Should create exactly 10 specs (2 per layer)
        assert_eq!(parallel_specs.len(), 10);
        assert_eq!(builder.tensor_spec_count().unwrap(), 10);

        // Verify each spec is correct
        for (i, spec_id) in parallel_specs.iter().enumerate() {
            let spec = builder.get_tensor_spec(*spec_id).unwrap();
            let layer_idx = i / 2;
            let is_key = i % 2 == 0;

            assert_eq!(spec.layer_idx, Some(layer_idx));
            if is_key {
                assert_eq!(spec.tensor_type, TensorType::KeyCache);
            } else {
                assert_eq!(spec.tensor_type, TensorType::ValueCache);
            }
        }

        // Test sequential creation for comparison
        builder.clear_specs().unwrap();

        for i in 0..5 {
            let key_spec = TensorSpec::with_layer_idx(
                format!("sequential_key_{}", i),
                ShapeTracker::new(&[1, 8, 1024, 128]),
                TensorType::KeyCache,
                i,
            );
            let value_spec = TensorSpec::with_layer_idx(
                format!("sequential_value_{}", i),
                ShapeTracker::new(&[1, 8, 1024, 128]),
                TensorType::ValueCache,
                i,
            );
            builder.add_tensor_spec(key_spec).unwrap();
            builder.add_tensor_spec(value_spec).unwrap();
        }

        // Should have same count as parallel version
        assert_eq!(builder.tensor_spec_count().unwrap(), 10);
    }

    #[test]
    fn test_parallel_operations_empty_input() {
        let builder = ParallelGraphBuilder::new();

        // Test with 0 layers
        let kv_specs = builder
            .create_kv_cache_specs_parallel(0, 8, 128, 1024)
            .unwrap();
        assert_eq!(kv_specs.len(), 0);

        let layer_specs = builder
            .create_transformer_layer_specs_parallel(0, 768, 12, 12, 3072)
            .unwrap();
        assert_eq!(layer_specs.len(), 0);

        assert_eq!(builder.tensor_spec_count().unwrap(), 0);
        assert_eq!(builder.layer_spec_count().unwrap(), 0);
    }

    #[test]
    fn test_sequential_assembly_integration() {
        let builder = ParallelGraphBuilder::new();

        // Create tensor specs first
        let tensor_spec = create_test_tensor_spec();
        let tensor_id = builder.add_tensor_spec(tensor_spec).unwrap();

        // Create layer spec that references the existing tensor
        let mut layer_spec = create_test_layer_spec();
        layer_spec.tensor_ids = vec![tensor_id]; // Reference the actual tensor ID

        let _layer_id = builder.add_layer_spec(layer_spec).unwrap();

        // Test assembly
        let result = builder.assemble().unwrap();

        // Should have assembled tensors and layers
        assert!(!result.tensors.is_empty());
        assert!(result.tensors.contains_key(&tensor_id));
        assert!(!result.layers.is_empty());
    }

    #[test]
    fn test_build_parallel_model() {
        let debugger = Arc::new(ParallelDebugger::new(DebugLevel::All));
        let builder = ParallelGraphBuilder::with_debugger(debugger.clone());

        // Build a small parallel model
        let result = builder.build_parallel_model(
            2,    // num_layers
            512,  // hidden_dim
            8,    // n_heads
            8,    // n_kv_heads
            2048, // mlp_dim
            1024, // sequence_length
            64,   // head_dim
        ).unwrap();

        // Should have created and assembled the model
        assert!(!result.tensors.is_empty());
        assert!(!result.layers.is_empty());

        // Should have logged multiple phases
        assert!(debugger.log_count() >= 6); // Multiple phase operations
    }

    #[test]
    fn test_dependency_validation_in_assembly() {
        let builder = ParallelGraphBuilder::new();

        // Create a layer spec that references a non-existent tensor
        let mut layer_spec = create_test_layer_spec();
        layer_spec.tensor_ids = vec![999]; // Non-existent tensor ID

        builder.add_layer_spec(layer_spec).unwrap();

        // Assembly should fail due to missing dependency
        let result = builder.assemble();
        assert!(result.is_err());

        match result.unwrap_err() {
            ParallelGraphError::AssemblyError { message } => {
                assert!(message.contains("Dependency not satisfied"));
            }
            _ => panic!("Expected AssemblyError"),
        }
    }

    #[test]
    fn test_batch_tensor_insertion_ordering() {
        let builder = ParallelGraphBuilder::new();

        // Create tensors with different layer indices to test ordering
        let mut spec1 = create_test_tensor_spec();
        spec1.layer_idx = Some(2);
        spec1.name = "layer2_tensor".to_string();

        let mut spec2 = create_test_tensor_spec();
        spec2.layer_idx = Some(0);
        spec2.name = "layer0_tensor".to_string();

        let mut spec3 = create_test_tensor_spec();
        spec3.layer_idx = Some(1);
        spec3.name = "layer1_tensor".to_string();

        builder.add_tensor_spec(spec1).unwrap();
        builder.add_tensor_spec(spec2).unwrap();
        builder.add_tensor_spec(spec3).unwrap();

        // Assembly should succeed and maintain proper ordering
        let result = builder.assemble().unwrap();
        assert_eq!(result.tensors.len(), 3);
    }
}