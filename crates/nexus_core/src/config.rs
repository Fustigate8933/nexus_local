//! Configuration file handling for Nexus Local.
//!
//! Loads settings from `nexus.config.toml` with the following search order:
//! 1. Current directory
//! 2. ~/.config/nexus/nexus.config.toml (Linux/macOS)
//! 3. %APPDATA%\nexus\nexus.config.toml (Windows)

use std::path::PathBuf;
use std::fs;
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// Main configuration structure.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct NexusConfig {
    pub index: IndexConfig,
    pub watch: WatchConfig,
    pub search: SearchConfig,
    pub gpu: GpuConfig,
    pub storage: StorageConfig,
}

/// Indexing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexConfig {
    /// Directories to index.
    pub roots: Vec<PathBuf>,
    /// File extensions to skip.
    pub skip_extensions: Vec<String>,
    /// Filename patterns to skip (substring match).
    pub skip_files: Vec<String>,
    /// Skip hidden files and directories.
    pub skip_hidden: bool,
    /// Maximum file size in MB.
    pub max_file_mb: u64,
    /// Maximum chunks per file (skip files exceeding this).
    pub max_chunks: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            roots: vec![],
            skip_extensions: vec!["exe".into(), "dll".into(), "so".into(), "o".into(), "pyc".into()],
            skip_files: vec!["node_modules".into(), ".git".into(), "target".into(), "__pycache__".into()],
            skip_hidden: true,
            max_file_mb: 50,
            max_chunks: 500,
        }
    }
}

/// Watch mode configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WatchConfig {
    /// Enable watch mode on startup.
    pub enabled: bool,
    /// Debounce delay in seconds.
    pub debounce_secs: u64,
    /// Patterns to ignore during watch (glob syntax).
    pub ignore_patterns: Vec<String>,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            debounce_secs: 2,
            ignore_patterns: vec![
                "*.tmp".into(),
                "*.swp".into(),
                "*~".into(),
                ".#*".into(),
                "*.lock".into(),
            ],
        }
    }
}

/// Search configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Default search mode.
    pub default_mode: String,
    /// Default number of results.
    pub results_count: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_mode: "hybrid".into(),
            results_count: 5,
        }
    }
}

/// GPU configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuConfig {
    /// Enable GPU acceleration.
    pub enabled: bool,
    /// CUDA device ID.
    pub device_id: u32,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device_id: 0,
        }
    }
}

/// Storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// Path to store index data.
    pub path: Option<PathBuf>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            path: None, // Will use default data_local_dir
        }
    }
}

impl NexusConfig {
    /// Config file name.
    pub const FILENAME: &'static str = "nexus.config.toml";

    /// Load configuration from file, searching standard locations.
    /// Returns default config if no file found.
    pub fn load() -> Result<Self> {
        if let Some(path) = Self::find_config_file() {
            Self::load_from(&path)
        } else {
            Ok(Self::default())
        }
    }

    /// Load configuration from a specific path.
    pub fn load_from(path: &PathBuf) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: NexusConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a file.
    pub fn save_to(&self, path: &PathBuf) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, content)?;
        Ok(())
    }

    /// Find config file in standard locations.
    pub fn find_config_file() -> Option<PathBuf> {
        // 1. Current directory
        let current = PathBuf::from(Self::FILENAME);
        if current.exists() {
            return Some(current);
        }

        // 2. Config directory (~/.config/nexus/ on Linux/macOS)
        if let Some(config_dir) = dirs::config_dir() {
            let path = config_dir.join("nexus").join(Self::FILENAME);
            if path.exists() {
                return Some(path);
            }
        }

        // 3. Home directory fallback
        if let Some(home) = dirs::home_dir() {
            let path = home.join(".nexus").join(Self::FILENAME);
            if path.exists() {
                return Some(path);
            }
        }

        None
    }

    /// Get the default config file path for the current platform.
    pub fn default_config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("nexus").join(Self::FILENAME))
    }

    /// Get the data directory path (uses storage.path or default).
    pub fn data_dir(&self) -> PathBuf {
        self.storage.path.clone().unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("nexus_local")
        })
    }

    /// Generate a default config file with comments.
    pub fn generate_default_config() -> String {
        r#"# Nexus Local Configuration
# Place this file at:
#   - ./nexus.config.toml (current directory)
#   - ~/.config/nexus/nexus.config.toml (Linux/macOS)
#   - %APPDATA%\nexus\nexus.config.toml (Windows)

[index]
# Directories to index (use ~ for home directory)
roots = ["~/Documents", "~/Projects"]

# File extensions to skip
skip_extensions = ["exe", "dll", "so", "o", "pyc", "class"]

# Directory/filename patterns to skip (substring match)
skip_files = ["node_modules", ".git", "target", "__pycache__", "venv", ".venv"]

# Skip hidden files (starting with .)
skip_hidden = true

# Maximum file size in MB
max_file_mb = 50

# Skip files that produce more than this many chunks
max_chunks = 500

[watch]
# Enable watch mode
enabled = false

# Debounce delay (seconds) - wait for file changes to settle
debounce_secs = 2

# Patterns to ignore during watch (glob syntax)
ignore_patterns = ["*.tmp", "*.swp", "*~", ".#*", "*.lock"]

[search]
# Default search mode: "hybrid", "semantic", or "lexical"
default_mode = "hybrid"

# Default number of results
results_count = 5

[gpu]
# Enable CUDA GPU acceleration
enabled = false

# CUDA device ID (for multi-GPU systems)
device_id = 0

[storage]
# Path for index data (default: ~/.local/share/nexus_local)
# path = "/custom/path/to/nexus_data"
"#.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = NexusConfig::default();
        assert_eq!(config.search.default_mode, "hybrid");
        assert_eq!(config.index.max_chunks, 500);
    }

    #[test]
    fn test_parse_config() {
        let toml_str = r#"
            [index]
            roots = ["/home/user/docs"]
            max_file_mb = 100

            [search]
            default_mode = "semantic"
        "#;
        
        let config: NexusConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.index.roots.len(), 1);
        assert_eq!(config.index.max_file_mb, 100);
        assert_eq!(config.search.default_mode, "semantic");
    }
}
