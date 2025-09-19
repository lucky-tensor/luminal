//! Parallel debugging infrastructure
//!
//! Provides thread-safe logging and debugging capabilities for parallel graph operations.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;
use log::{debug, info, warn, error};

/// Debug level for controlling verbosity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DebugLevel {
    /// No debug output
    None = 0,
    /// Basic progress information
    Basic = 1,
    /// Detailed operation logging
    Detailed = 2,
    /// Verbose logging including timings
    Verbose = 3,
    /// All debug information including internal state
    All = 4,
}

/// Thread-specific logging information
#[derive(Debug, Clone)]
pub struct ThreadLogEntry {
    /// Thread ID
    pub thread_id: String,
    /// Timestamp when the entry was created
    pub timestamp: Instant,
    /// Log level
    pub level: DebugLevel,
    /// Log message
    pub message: String,
    /// Optional duration for timed operations
    pub duration: Option<Duration>,
}

/// Thread-safe logger for parallel operations
#[derive(Debug)]
pub struct ThreadLogger {
    /// Log entries from all threads
    entries: Vec<ThreadLogEntry>,
    /// Thread names mapping
    thread_names: HashMap<String, String>,
}

impl ThreadLogger {
    /// Create a new thread logger
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            thread_names: HashMap::new(),
        }
    }

    /// Log a message from a specific thread
    pub fn log(&mut self, thread_id: String, level: DebugLevel, message: String) {
        self.entries.push(ThreadLogEntry {
            thread_id,
            timestamp: Instant::now(),
            level,
            message,
            duration: None,
        });
    }

    /// Log a message with duration
    pub fn log_with_duration(
        &mut self,
        thread_id: String,
        level: DebugLevel,
        message: String,
        duration: Duration,
    ) {
        self.entries.push(ThreadLogEntry {
            thread_id,
            timestamp: Instant::now(),
            level,
            message,
            duration: Some(duration),
        });
    }

    /// Set a friendly name for a thread
    pub fn set_thread_name(&mut self, thread_id: String, name: String) {
        self.thread_names.insert(thread_id, name);
    }

    /// Get all log entries for a specific thread
    pub fn get_thread_entries(&self, thread_id: &str) -> Vec<&ThreadLogEntry> {
        self.entries.iter()
            .filter(|entry| entry.thread_id == thread_id)
            .collect()
    }

    /// Get all log entries at or above a certain level
    pub fn get_entries_by_level(&self, min_level: DebugLevel) -> Vec<&ThreadLogEntry> {
        self.entries.iter()
            .filter(|entry| entry.level >= min_level)
            .collect()
    }

    /// Clear all log entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get the total number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the logger is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for ThreadLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel debugger for thread-safe logging and debugging
#[derive(Debug, Clone)]
pub struct ParallelDebugger {
    /// Current debug level
    level: DebugLevel,
    /// Thread-safe logger
    logger: Arc<Mutex<ThreadLogger>>,
    /// Start time for relative timestamps
    start_time: Instant,
}

impl ParallelDebugger {
    /// Create a new parallel debugger with the specified debug level
    pub fn new(level: DebugLevel) -> Self {
        Self {
            level,
            logger: Arc::new(Mutex::new(ThreadLogger::new())),
            start_time: Instant::now(),
        }
    }

    /// Get the current debug level
    pub fn level(&self) -> DebugLevel {
        self.level
    }

    /// Set the debug level
    pub fn set_level(&mut self, level: DebugLevel) {
        self.level = level;
    }

    /// Get current thread ID as string
    fn current_thread_id() -> String {
        format!("{:?}", thread::current().id())
    }

    /// Log tensor creation with timing
    pub fn log_tensor_creation(&self, tensor_name: &str, elapsed: Duration) {
        if self.level >= DebugLevel::Verbose {
            let thread_id = Self::current_thread_id();
            let message = format!("Created tensor '{}' in {:?}", tensor_name, elapsed);

            if let Ok(mut logger) = self.logger.lock() {
                logger.log_with_duration(thread_id.clone(), DebugLevel::Verbose, message.clone(), elapsed);
            }

            debug!("[{}] {}", thread_id, message);
        }
    }

    /// Log layer specification creation
    pub fn log_layer_spec(&self, layer_idx: usize, tensor_count: usize) {
        if self.level >= DebugLevel::Detailed {
            let thread_id = Self::current_thread_id();
            let message = format!("Layer {} spec created with {} tensors", layer_idx, tensor_count);

            if let Ok(mut logger) = self.logger.lock() {
                logger.log(thread_id.clone(), DebugLevel::Detailed, message.clone());
            }

            info!("[{}] {}", thread_id, message);
        }
    }

    /// Log parallel phase start
    pub fn log_phase_start(&self, phase_name: &str) {
        if self.level >= DebugLevel::Basic {
            let thread_id = Self::current_thread_id();
            let message = format!("Starting phase: {}", phase_name);

            if let Ok(mut logger) = self.logger.lock() {
                logger.log(thread_id.clone(), DebugLevel::Basic, message.clone());
            }

            info!("[{}] {}", thread_id, message);
        }
    }

    /// Log parallel phase completion
    pub fn log_phase_complete(&self, phase_name: &str, elapsed: Duration) {
        if self.level >= DebugLevel::Basic {
            let thread_id = Self::current_thread_id();
            let message = format!("Completed phase: {} in {:?}", phase_name, elapsed);

            if let Ok(mut logger) = self.logger.lock() {
                logger.log_with_duration(thread_id.clone(), DebugLevel::Basic, message.clone(), elapsed);
            }

            info!("[{}] {}", thread_id, message);
        }
    }

    /// Log an error
    pub fn log_error(&self, error_msg: &str) {
        let thread_id = Self::current_thread_id();
        let message = format!("ERROR: {}", error_msg);

        if let Ok(mut logger) = self.logger.lock() {
            logger.log(thread_id.clone(), DebugLevel::Basic, message.clone());
        }

        error!("[{}] {}", thread_id, message);
    }

    /// Log a warning
    pub fn log_warning(&self, warning_msg: &str) {
        if self.level >= DebugLevel::Basic {
            let thread_id = Self::current_thread_id();
            let message = format!("WARNING: {}", warning_msg);

            if let Ok(mut logger) = self.logger.lock() {
                logger.log(thread_id.clone(), DebugLevel::Basic, message.clone());
            }

            warn!("[{}] {}", thread_id, message);
        }
    }

    /// Get a copy of the current log entries
    pub fn get_log_entries(&self) -> Vec<ThreadLogEntry> {
        if let Ok(logger) = self.logger.lock() {
            logger.entries.clone()
        } else {
            Vec::new()
        }
    }

    /// Get log entries for a specific thread
    pub fn get_thread_log_entries(&self, thread_id: &str) -> Vec<ThreadLogEntry> {
        if let Ok(logger) = self.logger.lock() {
            logger.get_thread_entries(thread_id)
                .into_iter()
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Clear all log entries
    pub fn clear_logs(&self) {
        if let Ok(mut logger) = self.logger.lock() {
            logger.clear();
        }
    }

    /// Get the total number of log entries
    pub fn log_count(&self) -> usize {
        if let Ok(logger) = self.logger.lock() {
            logger.len()
        } else {
            0
        }
    }
}

impl Default for ParallelDebugger {
    fn default() -> Self {
        Self::new(DebugLevel::Basic)
    }
}

impl fmt::Display for DebugLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DebugLevel::None => write!(f, "None"),
            DebugLevel::Basic => write!(f, "Basic"),
            DebugLevel::Detailed => write!(f, "Detailed"),
            DebugLevel::Verbose => write!(f, "Verbose"),
            DebugLevel::All => write!(f, "All"),
        }
    }
}

impl fmt::Display for ThreadLogEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(duration) = self.duration {
            write!(
                f,
                "[{}] {:?} [{}] {} (took {:?})",
                self.thread_id, self.timestamp, self.level, self.message, duration
            )
        } else {
            write!(
                f,
                "[{}] {:?} [{}] {}",
                self.thread_id, self.timestamp, self.level, self.message
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_debug_level_ordering() {
        assert!(DebugLevel::None < DebugLevel::Basic);
        assert!(DebugLevel::Basic < DebugLevel::Detailed);
        assert!(DebugLevel::Detailed < DebugLevel::Verbose);
        assert!(DebugLevel::Verbose < DebugLevel::All);
    }

    #[test]
    fn test_thread_logger_creation() {
        let logger = ThreadLogger::new();
        assert!(logger.is_empty());
        assert_eq!(logger.len(), 0);
    }

    #[test]
    fn test_thread_logger_log() {
        let mut logger = ThreadLogger::new();
        logger.log(
            "thread-1".to_string(),
            DebugLevel::Basic,
            "Test message".to_string(),
        );

        assert_eq!(logger.len(), 1);
        let entries = logger.get_thread_entries("thread-1");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].message, "Test message");
        assert_eq!(entries[0].level, DebugLevel::Basic);
    }

    #[test]
    fn test_thread_logger_log_with_duration() {
        let mut logger = ThreadLogger::new();
        let duration = Duration::from_millis(100);
        logger.log_with_duration(
            "thread-1".to_string(),
            DebugLevel::Verbose,
            "Timed operation".to_string(),
            duration,
        );

        let entries = logger.get_thread_entries("thread-1");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].duration, Some(duration));
    }

    #[test]
    fn test_thread_logger_filter_by_level() {
        let mut logger = ThreadLogger::new();
        logger.log("t1".to_string(), DebugLevel::Basic, "Basic msg".to_string());
        logger.log("t1".to_string(), DebugLevel::Verbose, "Verbose msg".to_string());
        logger.log("t1".to_string(), DebugLevel::None, "None msg".to_string());

        let verbose_entries = logger.get_entries_by_level(DebugLevel::Verbose);
        assert_eq!(verbose_entries.len(), 1);
        assert_eq!(verbose_entries[0].message, "Verbose msg");

        let basic_entries = logger.get_entries_by_level(DebugLevel::Basic);
        assert_eq!(basic_entries.len(), 2); // Basic and Verbose
    }

    #[test]
    fn test_thread_logger_clear() {
        let mut logger = ThreadLogger::new();
        logger.log("t1".to_string(), DebugLevel::Basic, "Test".to_string());
        assert_eq!(logger.len(), 1);

        logger.clear();
        assert!(logger.is_empty());
        assert_eq!(logger.len(), 0);
    }

    #[test]
    fn test_parallel_debugger_creation() {
        let debugger = ParallelDebugger::new(DebugLevel::Verbose);
        assert_eq!(debugger.level(), DebugLevel::Verbose);
        assert_eq!(debugger.log_count(), 0);
    }

    #[test]
    fn test_parallel_debugger_set_level() {
        let mut debugger = ParallelDebugger::new(DebugLevel::None);
        assert_eq!(debugger.level(), DebugLevel::None);

        debugger.set_level(DebugLevel::All);
        assert_eq!(debugger.level(), DebugLevel::All);
    }

    #[test]
    fn test_parallel_debugger_log_tensor_creation() {
        let debugger = ParallelDebugger::new(DebugLevel::Verbose);
        debugger.log_tensor_creation("test_tensor", Duration::from_millis(50));

        assert_eq!(debugger.log_count(), 1);
        let entries = debugger.get_log_entries();
        assert!(entries[0].message.contains("test_tensor"));
        assert!(entries[0].duration.is_some());
    }

    #[test]
    fn test_parallel_debugger_log_layer_spec() {
        let debugger = ParallelDebugger::new(DebugLevel::Detailed);
        debugger.log_layer_spec(5, 10);

        assert_eq!(debugger.log_count(), 1);
        let entries = debugger.get_log_entries();
        assert!(entries[0].message.contains("Layer 5"));
        assert!(entries[0].message.contains("10 tensors"));
    }

    #[test]
    fn test_parallel_debugger_respects_debug_level() {
        let debugger = ParallelDebugger::new(DebugLevel::Basic);

        // This should not log because it requires Verbose level
        debugger.log_tensor_creation("test", Duration::from_millis(10));
        assert_eq!(debugger.log_count(), 0);

        // This should log because it requires Detailed level but we have Basic
        debugger.log_layer_spec(1, 5); // This actually requires Detailed level
        assert_eq!(debugger.log_count(), 0);

        // This should log because it requires Basic level
        debugger.log_phase_start("test_phase");
        assert_eq!(debugger.log_count(), 1);
    }

    #[test]
    fn test_parallel_debugger_clear_logs() {
        let debugger = ParallelDebugger::new(DebugLevel::All);
        debugger.log_phase_start("test");
        assert_eq!(debugger.log_count(), 1);

        debugger.clear_logs();
        assert_eq!(debugger.log_count(), 0);
    }

    #[test]
    fn test_parallel_debugger_thread_safety() {
        let debugger = Arc::new(ParallelDebugger::new(DebugLevel::All));
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let debugger = debugger.clone();
                thread::spawn(move || {
                    debugger.log_phase_start(&format!("phase_{}", i));
                    thread::sleep(Duration::from_millis(10));
                    debugger.log_phase_complete(&format!("phase_{}", i), Duration::from_millis(10));
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(debugger.log_count(), 8); // 4 threads * 2 logs each
    }

    #[test]
    fn test_debug_level_display() {
        assert_eq!(format!("{}", DebugLevel::None), "None");
        assert_eq!(format!("{}", DebugLevel::Verbose), "Verbose");
    }

    #[test]
    fn test_thread_log_entry_display() {
        let entry = ThreadLogEntry {
            thread_id: "test-thread".to_string(),
            timestamp: Instant::now(),
            level: DebugLevel::Basic,
            message: "Test message".to_string(),
            duration: Some(Duration::from_millis(100)),
        };

        let display = format!("{}", entry);
        assert!(display.contains("test-thread"));
        assert!(display.contains("Test message"));
        assert!(display.contains("took"));
    }

    #[test]
    fn test_default_implementations() {
        let logger = ThreadLogger::default();
        assert!(logger.is_empty());

        let debugger = ParallelDebugger::default();
        assert_eq!(debugger.level(), DebugLevel::Basic);
    }
}