//! Progress tracking for parallel operations
//!
//! Provides thread-safe progress tracking with atomic counters and
//! real-time progress indicators for parallel graph operations.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};

/// Phase of parallel graph construction
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProgressPhase {
    /// KV cache specification creation
    KVCacheCreation,
    /// Layer specification creation
    LayerSpecification,
    /// Sequential assembly phase
    SequentialAssembly,
    /// Model integration
    ModelIntegration,
}

impl fmt::Display for ProgressPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProgressPhase::KVCacheCreation => write!(f, "KV Cache Creation"),
            ProgressPhase::LayerSpecification => write!(f, "Layer Specification"),
            ProgressPhase::SequentialAssembly => write!(f, "Sequential Assembly"),
            ProgressPhase::ModelIntegration => write!(f, "Model Integration"),
        }
    }
}

/// Thread-safe progress tracker using atomic counters
#[derive(Debug)]
pub struct ProgressTracker {
    /// Progress for KV cache creation
    kv_cache_progress: Arc<AtomicUsize>,
    /// Progress for layer specification
    layer_progress: Arc<AtomicUsize>,
    /// Progress for sequential assembly
    assembly_progress: Arc<AtomicUsize>,
    /// Progress for model integration
    integration_progress: Arc<AtomicUsize>,
    /// Total expected operations for each phase
    expected_counts: ProgressCounts,
    /// Start time for overall tracking
    start_time: Instant,
    /// Multi-progress bar for terminal display
    multi_progress: Option<Arc<MultiProgress>>,
    /// Individual progress bars
    progress_bars: Option<ProgressBars>,
}

/// Expected operation counts for each phase
#[derive(Debug, Clone)]
pub struct ProgressCounts {
    /// Expected KV cache operations (typically num_layers * 2)
    pub kv_cache_ops: usize,
    /// Expected layer specification operations (typically num_layers)
    pub layer_spec_ops: usize,
    /// Expected assembly operations (varies by implementation)
    pub assembly_ops: usize,
    /// Expected integration operations
    pub integration_ops: usize,
}

/// Progress bars for terminal display
#[derive(Debug)]
struct ProgressBars {
    kv_cache_bar: ProgressBar,
    layer_spec_bar: ProgressBar,
    assembly_bar: ProgressBar,
    integration_bar: ProgressBar,
}

impl ProgressTracker {
    /// Create a new progress tracker with expected operation counts
    pub fn new(expected_counts: ProgressCounts) -> Self {
        Self {
            kv_cache_progress: Arc::new(AtomicUsize::new(0)),
            layer_progress: Arc::new(AtomicUsize::new(0)),
            assembly_progress: Arc::new(AtomicUsize::new(0)),
            integration_progress: Arc::new(AtomicUsize::new(0)),
            expected_counts,
            start_time: Instant::now(),
            multi_progress: None,
            progress_bars: None,
        }
    }

    /// Create a new progress tracker with terminal progress bars
    pub fn with_progress_bars(expected_counts: ProgressCounts) -> Self {
        let multi_progress = Arc::new(MultiProgress::new());

        let style = ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .expect("Invalid progress bar template")
            .progress_chars("##-");

        let kv_cache_bar = multi_progress.add(ProgressBar::new(expected_counts.kv_cache_ops as u64));
        kv_cache_bar.set_style(style.clone());
        kv_cache_bar.set_message("KV Cache Creation");

        let layer_spec_bar = multi_progress.add(ProgressBar::new(expected_counts.layer_spec_ops as u64));
        layer_spec_bar.set_style(style.clone());
        layer_spec_bar.set_message("Layer Specification");

        let assembly_bar = multi_progress.add(ProgressBar::new(expected_counts.assembly_ops as u64));
        assembly_bar.set_style(style.clone());
        assembly_bar.set_message("Sequential Assembly");

        let integration_bar = multi_progress.add(ProgressBar::new(expected_counts.integration_ops as u64));
        integration_bar.set_style(style);
        integration_bar.set_message("Model Integration");

        let progress_bars = ProgressBars {
            kv_cache_bar,
            layer_spec_bar,
            assembly_bar,
            integration_bar,
        };

        Self {
            kv_cache_progress: Arc::new(AtomicUsize::new(0)),
            layer_progress: Arc::new(AtomicUsize::new(0)),
            assembly_progress: Arc::new(AtomicUsize::new(0)),
            integration_progress: Arc::new(AtomicUsize::new(0)),
            expected_counts,
            start_time: Instant::now(),
            multi_progress: Some(multi_progress),
            progress_bars: Some(progress_bars),
        }
    }

    /// Update KV cache creation progress
    pub fn increment_kv_cache_progress(&self) -> usize {
        let new_count = self.kv_cache_progress.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(ref bars) = self.progress_bars {
            bars.kv_cache_bar.set_position(new_count as u64);
        }
        new_count
    }

    /// Update layer specification progress
    pub fn increment_layer_progress(&self) -> usize {
        let new_count = self.layer_progress.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(ref bars) = self.progress_bars {
            bars.layer_spec_bar.set_position(new_count as u64);
        }
        new_count
    }

    /// Update sequential assembly progress
    pub fn increment_assembly_progress(&self) -> usize {
        let new_count = self.assembly_progress.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(ref bars) = self.progress_bars {
            bars.assembly_bar.set_position(new_count as u64);
        }
        new_count
    }

    /// Update model integration progress
    pub fn increment_integration_progress(&self) -> usize {
        let new_count = self.integration_progress.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(ref bars) = self.progress_bars {
            bars.integration_bar.set_position(new_count as u64);
        }
        new_count
    }

    /// Get current progress for a specific phase
    pub fn get_progress(&self, phase: ProgressPhase) -> (usize, usize) {
        match phase {
            ProgressPhase::KVCacheCreation => {
                let current = self.kv_cache_progress.load(Ordering::SeqCst);
                (current, self.expected_counts.kv_cache_ops)
            }
            ProgressPhase::LayerSpecification => {
                let current = self.layer_progress.load(Ordering::SeqCst);
                (current, self.expected_counts.layer_spec_ops)
            }
            ProgressPhase::SequentialAssembly => {
                let current = self.assembly_progress.load(Ordering::SeqCst);
                (current, self.expected_counts.assembly_ops)
            }
            ProgressPhase::ModelIntegration => {
                let current = self.integration_progress.load(Ordering::SeqCst);
                (current, self.expected_counts.integration_ops)
            }
        }
    }

    /// Get progress percentage for a specific phase
    pub fn get_progress_percentage(&self, phase: ProgressPhase) -> f64 {
        let (current, total) = self.get_progress(phase);
        if total == 0 {
            100.0
        } else {
            (current as f64 / total as f64) * 100.0
        }
    }

    /// Check if a phase is complete
    pub fn is_phase_complete(&self, phase: ProgressPhase) -> bool {
        let (current, total) = self.get_progress(phase);
        current >= total
    }

    /// Check if all phases are complete
    pub fn is_complete(&self) -> bool {
        self.is_phase_complete(ProgressPhase::KVCacheCreation) &&
        self.is_phase_complete(ProgressPhase::LayerSpecification) &&
        self.is_phase_complete(ProgressPhase::SequentialAssembly) &&
        self.is_phase_complete(ProgressPhase::ModelIntegration)
    }

    /// Get total elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Reset all progress counters
    pub fn reset(&self) {
        self.kv_cache_progress.store(0, Ordering::SeqCst);
        self.layer_progress.store(0, Ordering::SeqCst);
        self.assembly_progress.store(0, Ordering::SeqCst);
        self.integration_progress.store(0, Ordering::SeqCst);

        if let Some(ref bars) = self.progress_bars {
            bars.kv_cache_bar.set_position(0);
            bars.layer_spec_bar.set_position(0);
            bars.assembly_bar.set_position(0);
            bars.integration_bar.set_position(0);
        }
    }

    /// Finish a phase and mark its progress bar as complete
    pub fn finish_phase(&self, phase: ProgressPhase) {
        if let Some(ref bars) = self.progress_bars {
            let bar = match phase {
                ProgressPhase::KVCacheCreation => &bars.kv_cache_bar,
                ProgressPhase::LayerSpecification => &bars.layer_spec_bar,
                ProgressPhase::SequentialAssembly => &bars.assembly_bar,
                ProgressPhase::ModelIntegration => &bars.integration_bar,
            };
            bar.finish_with_message(format!("{} complete", phase));
        }
    }

    /// Finish all progress bars
    pub fn finish(&self) {
        if let Some(ref bars) = self.progress_bars {
            bars.kv_cache_bar.finish();
            bars.layer_spec_bar.finish();
            bars.assembly_bar.finish();
            bars.integration_bar.finish();
        }
    }

    /// Get a formatted progress summary
    pub fn summary(&self) -> String {
        let kv_pct = self.get_progress_percentage(ProgressPhase::KVCacheCreation);
        let layer_pct = self.get_progress_percentage(ProgressPhase::LayerSpecification);
        let assembly_pct = self.get_progress_percentage(ProgressPhase::SequentialAssembly);
        let integration_pct = self.get_progress_percentage(ProgressPhase::ModelIntegration);

        format!(
            "Progress: KV Cache {:.1}%, Layers {:.1}%, Assembly {:.1}%, Integration {:.1}% | Elapsed: {:?}",
            kv_pct, layer_pct, assembly_pct, integration_pct, self.elapsed()
        )
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new(ProgressCounts {
            kv_cache_ops: 0,
            layer_spec_ops: 0,
            assembly_ops: 0,
            integration_ops: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_progress_counts_creation() {
        let counts = ProgressCounts {
            kv_cache_ops: 72,
            layer_spec_ops: 36,
            assembly_ops: 100,
            integration_ops: 10,
        };

        assert_eq!(counts.kv_cache_ops, 72);
        assert_eq!(counts.layer_spec_ops, 36);
    }

    #[test]
    fn test_progress_tracker_creation() {
        let counts = ProgressCounts {
            kv_cache_ops: 10,
            layer_spec_ops: 5,
            assembly_ops: 20,
            integration_ops: 3,
        };

        let tracker = ProgressTracker::new(counts);
        assert_eq!(tracker.get_progress(ProgressPhase::KVCacheCreation), (0, 10));
        assert_eq!(tracker.get_progress(ProgressPhase::LayerSpecification), (0, 5));
        assert!(!tracker.is_complete());
    }

    #[test]
    fn test_increment_kv_cache_progress() {
        let counts = ProgressCounts {
            kv_cache_ops: 5,
            layer_spec_ops: 3,
            assembly_ops: 10,
            integration_ops: 2,
        };

        let tracker = ProgressTracker::new(counts);
        assert_eq!(tracker.increment_kv_cache_progress(), 1);
        assert_eq!(tracker.increment_kv_cache_progress(), 2);
        assert_eq!(tracker.get_progress(ProgressPhase::KVCacheCreation), (2, 5));
    }

    #[test]
    fn test_increment_layer_progress() {
        let counts = ProgressCounts {
            kv_cache_ops: 5,
            layer_spec_ops: 3,
            assembly_ops: 10,
            integration_ops: 2,
        };

        let tracker = ProgressTracker::new(counts);
        assert_eq!(tracker.increment_layer_progress(), 1);
        assert_eq!(tracker.increment_layer_progress(), 2);
        assert_eq!(tracker.get_progress(ProgressPhase::LayerSpecification), (2, 3));
    }

    #[test]
    fn test_progress_percentage() {
        let counts = ProgressCounts {
            kv_cache_ops: 10,
            layer_spec_ops: 4,
            assembly_ops: 20,
            integration_ops: 5,
        };

        let tracker = ProgressTracker::new(counts);

        // Test 0% progress
        assert_eq!(tracker.get_progress_percentage(ProgressPhase::KVCacheCreation), 0.0);

        // Test 50% progress
        for _ in 0..2 {
            tracker.increment_layer_progress();
        }
        assert_eq!(tracker.get_progress_percentage(ProgressPhase::LayerSpecification), 50.0);

        // Test 100% progress
        for _ in 0..2 {
            tracker.increment_layer_progress();
        }
        assert_eq!(tracker.get_progress_percentage(ProgressPhase::LayerSpecification), 100.0);
    }

    #[test]
    fn test_is_phase_complete() {
        let counts = ProgressCounts {
            kv_cache_ops: 2,
            layer_spec_ops: 3,
            assembly_ops: 1,
            integration_ops: 1,
        };

        let tracker = ProgressTracker::new(counts);
        assert!(!tracker.is_phase_complete(ProgressPhase::KVCacheCreation));

        // Complete KV cache phase
        tracker.increment_kv_cache_progress();
        tracker.increment_kv_cache_progress();
        assert!(tracker.is_phase_complete(ProgressPhase::KVCacheCreation));
    }

    #[test]
    fn test_is_complete() {
        let counts = ProgressCounts {
            kv_cache_ops: 1,
            layer_spec_ops: 1,
            assembly_ops: 1,
            integration_ops: 1,
        };

        let tracker = ProgressTracker::new(counts);
        assert!(!tracker.is_complete());

        // Complete all phases
        tracker.increment_kv_cache_progress();
        tracker.increment_layer_progress();
        tracker.increment_assembly_progress();
        tracker.increment_integration_progress();

        assert!(tracker.is_complete());
    }

    #[test]
    fn test_reset() {
        let counts = ProgressCounts {
            kv_cache_ops: 5,
            layer_spec_ops: 5,
            assembly_ops: 5,
            integration_ops: 5,
        };

        let tracker = ProgressTracker::new(counts);

        // Make some progress
        tracker.increment_kv_cache_progress();
        tracker.increment_layer_progress();
        assert_eq!(tracker.get_progress(ProgressPhase::KVCacheCreation), (1, 5));

        // Reset
        tracker.reset();
        assert_eq!(tracker.get_progress(ProgressPhase::KVCacheCreation), (0, 5));
        assert_eq!(tracker.get_progress(ProgressPhase::LayerSpecification), (0, 5));
    }

    #[test]
    fn test_elapsed_time() {
        let counts = ProgressCounts {
            kv_cache_ops: 1,
            layer_spec_ops: 1,
            assembly_ops: 1,
            integration_ops: 1,
        };

        let tracker = ProgressTracker::new(counts);
        thread::sleep(Duration::from_millis(10));
        assert!(tracker.elapsed() >= Duration::from_millis(10));
    }

    #[test]
    fn test_thread_safety() {
        let counts = ProgressCounts {
            kv_cache_ops: 100,
            layer_spec_ops: 100,
            assembly_ops: 100,
            integration_ops: 100,
        };

        let tracker = Arc::new(ProgressTracker::new(counts));
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let tracker = tracker.clone();
                thread::spawn(move || {
                    for _ in 0..10 {
                        tracker.increment_kv_cache_progress();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(tracker.get_progress(ProgressPhase::KVCacheCreation), (100, 100));
        assert!(tracker.is_phase_complete(ProgressPhase::KVCacheCreation));
    }

    #[test]
    fn test_progress_phase_display() {
        assert_eq!(format!("{}", ProgressPhase::KVCacheCreation), "KV Cache Creation");
        assert_eq!(format!("{}", ProgressPhase::LayerSpecification), "Layer Specification");
        assert_eq!(format!("{}", ProgressPhase::SequentialAssembly), "Sequential Assembly");
        assert_eq!(format!("{}", ProgressPhase::ModelIntegration), "Model Integration");
    }

    #[test]
    fn test_summary() {
        let counts = ProgressCounts {
            kv_cache_ops: 4,
            layer_spec_ops: 4,
            assembly_ops: 4,
            integration_ops: 4,
        };

        let tracker = ProgressTracker::new(counts);
        tracker.increment_kv_cache_progress(); // 25%
        tracker.increment_layer_progress(); // 25%
        tracker.increment_layer_progress(); // 50%

        let summary = tracker.summary();
        assert!(summary.contains("25.0%")); // KV Cache
        assert!(summary.contains("50.0%")); // Layer
        assert!(summary.contains("0.0%"));  // Assembly and Integration
        assert!(summary.contains("Elapsed"));
    }

    #[test]
    fn test_default_implementation() {
        let tracker = ProgressTracker::default();
        assert_eq!(tracker.get_progress(ProgressPhase::KVCacheCreation), (0, 0));
        assert!(tracker.is_complete()); // All phases complete when expected count is 0
    }

    #[test]
    fn test_zero_expected_count_percentage() {
        let counts = ProgressCounts {
            kv_cache_ops: 0,
            layer_spec_ops: 1,
            assembly_ops: 1,
            integration_ops: 1,
        };

        let tracker = ProgressTracker::new(counts);
        // When expected count is 0, percentage should be 100%
        assert_eq!(tracker.get_progress_percentage(ProgressPhase::KVCacheCreation), 100.0);
    }
}