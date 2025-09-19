use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

/// Represents a compiled cache artifact
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheArtifact {
    pub name: String,
    pub content_hash: String,
    pub size_bytes: usize,
    pub metadata: HashMap<String, String>,
}

/// Cache comparison results
#[derive(Debug)]
pub struct CacheComparisonResult {
    pub identical: bool,
    pub mismatched_files: Vec<String>,
    pub sequential_only: Vec<String>,
    pub parallel_only: Vec<String>,
    pub detailed_diff: Vec<FileDifference>,
}

/// Detailed file difference information
#[derive(Debug)]
pub struct FileDifference {
    pub file_name: String,
    pub sequential_hash: String,
    pub parallel_hash: String,
    pub size_difference: i64,
}

/// Cache artifact manager
pub struct CacheManager {
    base_dir: PathBuf,
}

impl CacheManager {
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_path_buf(),
        }
    }

    /// Create separate cache directories for different compilation methods
    pub fn setup_test_directories(&self) -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
        let seq_dir = self.base_dir.join("sequential");
        let par_dir = self.base_dir.join("parallel");

        fs::create_dir_all(&seq_dir)?;
        fs::create_dir_all(&par_dir)?;

        // Clean existing cache files
        self.clean_directory(&seq_dir)?;
        self.clean_directory(&par_dir)?;

        Ok((seq_dir, par_dir))
    }

    /// Clean all files in a directory
    fn clean_directory(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if dir.exists() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    fs::remove_file(path)?;
                }
            }
        }
        Ok(())
    }

    /// Scan a directory and extract cache artifacts
    pub fn extract_artifacts<P: AsRef<Path>>(&self, cache_dir: P) -> Result<Vec<CacheArtifact>, Box<dyn std::error::Error>> {
        let mut artifacts = Vec::new();
        let cache_path = cache_dir.as_ref();

        if !cache_path.exists() {
            return Ok(artifacts);
        }

        for entry in fs::read_dir(cache_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                let artifact = self.analyze_cache_file(&path)?;
                artifacts.push(artifact);
            }
        }

        // Sort by name for consistent ordering
        artifacts.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(artifacts)
    }

    /// Analyze a single cache file
    fn analyze_cache_file(&self, file_path: &Path) -> Result<CacheArtifact, Box<dyn std::error::Error>> {
        let mut file = File::open(file_path)?;
        let mut content = Vec::new();
        file.read_to_end(&mut content)?;

        let mut hasher = Sha256::new();
        hasher.update(&content);
        let content_hash = format!("{:x}", hasher.finalize());

        let mut metadata = HashMap::new();

        // Add file metadata
        let file_name = file_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        metadata.insert("file_extension".to_string(),
            file_path.extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("none")
                .to_string()
        );

        metadata.insert("file_stem".to_string(),
            file_path.file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("unknown")
                .to_string()
        );

        // Try to detect file type based on content
        if self.is_binary_ptx(&content) {
            metadata.insert("type".to_string(), "ptx_binary".to_string());
        } else if self.is_metal_library(&content) {
            metadata.insert("type".to_string(), "metal_library".to_string());
        } else if self.looks_like_serialized_graph(&content) {
            metadata.insert("type".to_string(), "serialized_graph".to_string());
        } else {
            metadata.insert("type".to_string(), "unknown".to_string());
        }

        Ok(CacheArtifact {
            name: file_name,
            content_hash,
            size_bytes: content.len(),
            metadata,
        })
    }

    /// Heuristic to detect PTX binary content
    fn is_binary_ptx(&self, content: &[u8]) -> bool {
        // PTX files often contain specific headers or patterns
        content.windows(4).any(|w| w == b".ptx" || w == b"nvvm")
    }

    /// Heuristic to detect Metal library content
    fn is_metal_library(&self, content: &[u8]) -> bool {
        // Metal libraries have specific magic bytes
        content.len() > 8 && (
            content.starts_with(b"\x4d\x54\x4c\x42") || // MTLB magic
            content.windows(4).any(|w| w == b"MTLB")
        )
    }

    /// Heuristic to detect serialized graph data
    fn looks_like_serialized_graph(&self, content: &[u8]) -> bool {
        // Look for bincode or other serialization patterns
        content.len() > 0 && (
            // Common patterns in serialized data
            content[0] < 32 && content.len() > 100 ||
            content.windows(8).any(|w| w == b"graph_no")
        )
    }

    /// Compare cache artifacts from two compilation methods
    pub fn compare_caches(
        &self,
        sequential_artifacts: &[CacheArtifact],
        parallel_artifacts: &[CacheArtifact],
    ) -> CacheComparisonResult {
        let seq_map: HashMap<&String, &CacheArtifact> = sequential_artifacts
            .iter()
            .map(|a| (&a.name, a))
            .collect();

        let par_map: HashMap<&String, &CacheArtifact> = parallel_artifacts
            .iter()
            .map(|a| (&a.name, a))
            .collect();

        let mut mismatched_files = Vec::new();
        let mut detailed_diff = Vec::new();

        // Find files present in both caches but with differences
        for (name, seq_artifact) in &seq_map {
            if let Some(par_artifact) = par_map.get(name) {
                if seq_artifact.content_hash != par_artifact.content_hash {
                    mismatched_files.push((*name).clone());
                    detailed_diff.push(FileDifference {
                        file_name: (*name).clone(),
                        sequential_hash: seq_artifact.content_hash.clone(),
                        parallel_hash: par_artifact.content_hash.clone(),
                        size_difference: par_artifact.size_bytes as i64 - seq_artifact.size_bytes as i64,
                    });
                }
            }
        }

        // Find files only in sequential cache
        let sequential_only: Vec<String> = seq_map
            .keys()
            .filter(|name| !par_map.contains_key(*name))
            .map(|name| (*name).clone())
            .collect();

        // Find files only in parallel cache
        let parallel_only: Vec<String> = par_map
            .keys()
            .filter(|name| !seq_map.contains_key(*name))
            .map(|name| (*name).clone())
            .collect();

        let identical = mismatched_files.is_empty()
            && sequential_only.is_empty()
            && parallel_only.is_empty();

        CacheComparisonResult {
            identical,
            mismatched_files,
            sequential_only,
            parallel_only,
            detailed_diff,
        }
    }

    /// Save artifact analysis to JSON for debugging
    pub fn save_analysis<P: AsRef<Path>>(
        &self,
        artifacts: &[CacheArtifact],
        output_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(artifacts)?;
        let mut file = File::create(output_path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Generate a detailed comparison report
    pub fn generate_comparison_report(&self, result: &CacheComparisonResult) -> String {
        let mut report = String::new();

        report.push_str("# Cache Comparison Report\n\n");

        if result.identical {
            report.push_str("✅ **PASS**: All cache artifacts are identical between sequential and parallel compilation.\n\n");
        } else {
            report.push_str("❌ **FAIL**: Cache artifacts differ between compilation methods.\n\n");
        }

        if !result.mismatched_files.is_empty() {
            report.push_str("## Mismatched Files\n");
            for file in &result.mismatched_files {
                report.push_str(&format!("- {}\n", file));
            }
            report.push('\n');

            report.push_str("## Detailed Differences\n");
            for diff in &result.detailed_diff {
                report.push_str(&format!("### {}\n", diff.file_name));
                report.push_str(&format!("- Sequential hash: `{}`\n", diff.sequential_hash));
                report.push_str(&format!("- Parallel hash: `{}`\n", diff.parallel_hash));
                report.push_str(&format!("- Size difference: {} bytes\n\n", diff.size_difference));
            }
        }

        if !result.sequential_only.is_empty() {
            report.push_str("## Files Only in Sequential Cache\n");
            for file in &result.sequential_only {
                report.push_str(&format!("- {}\n", file));
            }
            report.push('\n');
        }

        if !result.parallel_only.is_empty() {
            report.push_str("## Files Only in Parallel Cache\n");
            for file in &result.parallel_only {
                report.push_str(&format!("- {}\n", file));
            }
            report.push('\n');
        }

        report.push_str("---\n");
        report.push_str(&format!("Generated on: {}\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_cache_manager_setup() {
        let temp_dir = tempdir().unwrap();
        let manager = CacheManager::new(temp_dir.path());

        let (seq_dir, par_dir) = manager.setup_test_directories().unwrap();

        assert!(seq_dir.exists());
        assert!(par_dir.exists());
        assert_eq!(seq_dir.file_name().unwrap(), "sequential");
        assert_eq!(par_dir.file_name().unwrap(), "parallel");
    }

    #[test]
    fn test_artifact_extraction() {
        let temp_dir = tempdir().unwrap();
        let manager = CacheManager::new(temp_dir.path());

        // Create a mock cache file
        let cache_dir = temp_dir.path().join("test_cache");
        fs::create_dir_all(&cache_dir).unwrap();

        let test_file = cache_dir.join("test.bin");
        fs::write(&test_file, b"mock cache data").unwrap();

        let artifacts = manager.extract_artifacts(&cache_dir).unwrap();

        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].name, "test.bin");
        assert_eq!(artifacts[0].size_bytes, 15);
        assert!(!artifacts[0].content_hash.is_empty());
    }

    #[test]
    fn test_cache_comparison() {
        let manager = CacheManager::new(".");

        let seq_artifacts = vec![
            CacheArtifact {
                name: "kernel1.bin".to_string(),
                content_hash: "abc123".to_string(),
                size_bytes: 100,
                metadata: HashMap::new(),
            },
            CacheArtifact {
                name: "kernel2.bin".to_string(),
                content_hash: "def456".to_string(),
                size_bytes: 200,
                metadata: HashMap::new(),
            },
        ];

        let par_artifacts = vec![
            CacheArtifact {
                name: "kernel1.bin".to_string(),
                content_hash: "abc123".to_string(), // Same
                size_bytes: 100,
                metadata: HashMap::new(),
            },
            CacheArtifact {
                name: "kernel2.bin".to_string(),
                content_hash: "xyz789".to_string(), // Different!
                size_bytes: 200,
                metadata: HashMap::new(),
            },
        ];

        let result = manager.compare_caches(&seq_artifacts, &par_artifacts);

        assert!(!result.identical);
        assert_eq!(result.mismatched_files, vec!["kernel2.bin"]);
        assert!(result.sequential_only.is_empty());
        assert!(result.parallel_only.is_empty());
        assert_eq!(result.detailed_diff.len(), 1);
    }
}