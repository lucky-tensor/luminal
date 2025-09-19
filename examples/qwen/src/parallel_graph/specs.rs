//! Tensor and Layer specification structures
//!
//! These structures capture the parameters needed to create tensors and layers
//! without requiring exclusive graph access, enabling parallel specification creation.

use std::sync::atomic::{AtomicUsize, Ordering};
use luminal::prelude::*;

/// Atomic counter for generating unique tensor IDs
static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Type of tensor to be created
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TensorType {
    /// Regular tensor with data
    Data,
    /// Key cache tensor
    KeyCache,
    /// Value cache tensor
    ValueCache,
    /// Weight tensor
    Weight,
    /// Input tensor
    Input,
    /// Output tensor
    Output,
}

/// Specification for creating a tensor without immediate graph mutation
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSpec {
    /// Unique identifier for this tensor spec
    pub id: usize,
    /// Name of the tensor
    pub name: String,
    /// Shape of the tensor
    pub shape: ShapeTracker,
    /// Type of tensor
    pub tensor_type: TensorType,
    /// Layer index (if applicable)
    pub layer_idx: Option<usize>,
}

/// Type of layer to be created
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LayerType {
    /// Transformer block layer
    TransformerBlock,
    /// Multi-layer perceptron
    Mlp,
    /// Attention layer
    Attention,
    /// Layer normalization
    LayerNorm,
    /// Embedding layer
    Embedding,
    /// Linear layer
    Linear,
}

/// Specification for creating a layer without immediate graph mutation
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSpec {
    /// Unique identifier for this layer spec
    pub id: usize,
    /// Type of layer
    pub layer_type: LayerType,
    /// References to tensor specs this layer uses
    pub tensor_ids: Vec<usize>,
    /// Layer index in the model
    pub layer_idx: usize,
    /// Additional layer-specific parameters
    pub parameters: LayerParameters,
}

/// Layer-specific parameters
#[derive(Debug, Clone, PartialEq)]
pub enum LayerParameters {
    /// Parameters for transformer block
    TransformerBlock {
        hidden_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        mlp_dim: usize,
    },
    /// Parameters for MLP
    Mlp {
        hidden_dim: usize,
        intermediate_dim: usize,
    },
    /// Parameters for attention
    Attention {
        hidden_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    },
    /// Parameters for layer norm
    LayerNorm {
        dim: usize,
    },
    /// Parameters for embedding
    Embedding {
        vocab_size: usize,
        embed_dim: usize,
    },
    /// Parameters for linear layer
    Linear {
        in_dim: usize,
        out_dim: usize,
        bias: bool,
    },
}

impl TensorSpec {
    /// Create a new tensor specification
    pub fn new(name: String, shape: ShapeTracker, tensor_type: TensorType) -> Self {
        Self {
            id: TENSOR_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            name,
            shape,
            tensor_type,
            layer_idx: None,
        }
    }

    /// Create a new tensor specification with layer index
    pub fn with_layer_idx(
        name: String,
        shape: ShapeTracker,
        tensor_type: TensorType,
        layer_idx: usize,
    ) -> Self {
        Self {
            id: TENSOR_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            name,
            shape,
            tensor_type,
            layer_idx: Some(layer_idx),
        }
    }
}

impl LayerSpec {
    /// Create a new layer specification
    pub fn new(
        layer_type: LayerType,
        tensor_ids: Vec<usize>,
        layer_idx: usize,
        parameters: LayerParameters,
    ) -> Self {
        static LAYER_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self {
            id: LAYER_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            layer_type,
            tensor_ids,
            layer_idx,
            parameters,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_spec_creation() {
        let shape = ShapeTracker::new(&[1, 512, 768]);
        let spec = TensorSpec::new(
            "test_tensor".to_string(),
            shape,
            TensorType::Data,
        );

        assert_eq!(spec.name, "test_tensor");
        assert_eq!(spec.tensor_type, TensorType::Data);
        assert_eq!(spec.layer_idx, None);
        assert!(spec.id > 0);
    }

    #[test]
    fn test_tensor_spec_with_layer_idx() {
        let shape = ShapeTracker::new(&[1, 8, 1024, 128]);
        let spec = TensorSpec::with_layer_idx(
            "key_cache_layer_5".to_string(),
            shape,
            TensorType::KeyCache,
            5,
        );

        assert_eq!(spec.name, "key_cache_layer_5");
        assert_eq!(spec.tensor_type, TensorType::KeyCache);
        assert_eq!(spec.layer_idx, Some(5));
    }

    #[test]
    fn test_tensor_spec_unique_ids() {
        let shape = ShapeTracker::new(&[1, 512]);
        let spec1 = TensorSpec::new("tensor1".to_string(), shape, TensorType::Data);
        let spec2 = TensorSpec::new("tensor2".to_string(), shape, TensorType::Data);

        assert_ne!(spec1.id, spec2.id);
    }

    #[test]
    fn test_layer_spec_creation() {
        let params = LayerParameters::TransformerBlock {
            hidden_dim: 768,
            n_heads: 12,
            n_kv_heads: 12,
            mlp_dim: 3072,
        };
        let spec = LayerSpec::new(
            LayerType::TransformerBlock,
            vec![1, 2, 3],
            0,
            params.clone(),
        );

        assert_eq!(spec.layer_type, LayerType::TransformerBlock);
        assert_eq!(spec.tensor_ids, vec![1, 2, 3]);
        assert_eq!(spec.layer_idx, 0);
        assert_eq!(spec.parameters, params);
    }

    #[test]
    fn test_layer_spec_unique_ids() {
        let params = LayerParameters::Mlp {
            hidden_dim: 768,
            intermediate_dim: 3072,
        };
        let spec1 = LayerSpec::new(LayerType::Mlp, vec![1], 0, params.clone());
        let spec2 = LayerSpec::new(LayerType::Mlp, vec![2], 1, params);

        assert_ne!(spec1.id, spec2.id);
    }

    #[test]
    fn test_tensor_type_debug_display() {
        assert_eq!(format!("{:?}", TensorType::KeyCache), "KeyCache");
        assert_eq!(format!("{:?}", TensorType::ValueCache), "ValueCache");
    }

    #[test]
    fn test_layer_type_debug_display() {
        assert_eq!(format!("{:?}", LayerType::TransformerBlock), "TransformerBlock");
        assert_eq!(format!("{:?}", LayerType::Attention), "Attention");
    }

    #[test]
    fn test_tensor_spec_clone() {
        let shape = ShapeTracker::new(&[1, 512, 768]);
        let spec = TensorSpec::new("test".to_string(), shape, TensorType::Data);
        let cloned = spec.clone();

        assert_eq!(spec, cloned);
        assert_eq!(spec.id, cloned.id);
    }

    #[test]
    fn test_layer_parameters_equality() {
        let params1 = LayerParameters::Linear {
            in_dim: 768,
            out_dim: 256,
            bias: true,
        };
        let params2 = LayerParameters::Linear {
            in_dim: 768,
            out_dim: 256,
            bias: true,
        };
        let params3 = LayerParameters::Linear {
            in_dim: 768,
            out_dim: 256,
            bias: false,
        };

        assert_eq!(params1, params2);
        assert_ne!(params1, params3);
    }
}