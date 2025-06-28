//! Configuration handling for libtree.
//!
//! This module provides functionality for parsing ld.so.conf files and
//! handling configuration options.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use glob::glob;

use crate::error::{Error, Result};
use crate::LibtreeState;

/// Configuration options for libtree
#[derive(Debug, Clone)]
pub struct LibtreeConfig {
    /// Verbosity level (0-3)
    pub verbosity: u32,
    /// Whether to show full paths
    pub show_path: bool,
    /// Whether to use color in output
    pub color: bool,
    /// Path to ld.so.conf file
    pub ld_conf_file: String,
    /// Maximum recursion depth
    pub max_depth: usize,
}

impl Default for LibtreeConfig {
    fn default() -> Self {
        Self {
            verbosity: 0,
            show_path: false,
            color: true,
            ld_conf_file: "/etc/ld.so.conf".to_string(),
            max_depth: 32,
        }
    }
}

/// Parse an ld.so.conf file and its included files
fn parse_ld_conf_file(state: &mut LibtreeState, path: &Path) -> Result<()> {
    let file = File::open(path).map_err(|e| Error::Io(e))?;
    let reader = BufReader::new(file);
    let current_dir = path.parent().unwrap_or(Path::new("/")).to_path_buf();

    for line in reader.lines() {
        let line = line.map_err(|e| Error::Io(e))?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Handle include directives
        if line.starts_with("include ") {
            let pattern = line["include ".len()..].trim();
            let pattern = if !pattern.starts_with('/') {
                // Relative path - prepend current directory
                current_dir.join(pattern).to_string_lossy().into_owned()
            } else {
                pattern.to_string()
            };

            // Expand glob pattern and parse included files
            for entry in glob(&pattern).map_err(|e| Error::Other(e.to_string()))? {
                match entry {
                    Ok(path) => {
                        parse_ld_conf_file(state, &path)?;
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to read glob entry: {}", e);
                    }
                }
            }
        } else {
            // Store the library path
            state.internal.append_ld_conf_path(line);
        }
    }

    Ok(())
}

/// Parse the system's ld.so.conf file and its includes
pub(crate) fn parse_ld_so_conf(state: &mut LibtreeState) -> Result<()> {
    // Initialize ld.so.conf paths
    state.internal.init_ld_conf_paths();

    // Standard library directories for different architectures
    #[cfg(target_arch = "x86_64")]
    {
        state.internal.append_ld_conf_path("/lib64");
        state.internal.append_ld_conf_path("/usr/lib64");
        state.internal.append_ld_conf_path("/lib/x86_64-linux-gnu");
        state.internal.append_ld_conf_path("/usr/lib/x86_64-linux-gnu");
    }

    #[cfg(target_arch = "aarch64")]
    {
        state.internal.append_ld_conf_path("/lib/aarch64-linux-gnu");
        state.internal.append_ld_conf_path("/usr/lib/aarch64-linux-gnu");
    }

    #[cfg(target_arch = "x86")]
    {
        state.internal.append_ld_conf_path("/lib");
        state.internal.append_ld_conf_path("/usr/lib");
        state.internal.append_ld_conf_path("/lib/i386-linux-gnu");
        state.internal.append_ld_conf_path("/usr/lib/i386-linux-gnu");
    }

    // Parse the main ld.so.conf file
    let conf_path = Path::new(&state.ld_conf_file);
    if conf_path.exists() {
        parse_ld_conf_file(state, conf_path)?;
    }

    Ok(())
}

/// Builder for LibtreeConfig
pub struct LibtreeConfigBuilder {
    config: LibtreeConfig,
}

impl LibtreeConfigBuilder {
    /// Create a new config builder with default values
    pub fn new() -> Self {
        Self {
            config: LibtreeConfig::default(),
        }
    }

    /// Set the verbosity level
    pub fn verbosity(mut self, level: u32) -> Self {
        self.config.verbosity = level;
        self
    }

    /// Set whether to show full paths
    pub fn show_path(mut self, show: bool) -> Self {
        self.config.show_path = show;
        self
    }

    /// Set whether to use color output
    pub fn color(mut self, color: bool) -> Self {
        self.config.color = color;
        self
    }

    /// Set the path to ld.so.conf
    pub fn ld_conf_file<P: Into<String>>(mut self, path: P) -> Self {
        self.config.ld_conf_file = path.into();
        self
    }

    /// Set the maximum recursion depth
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.config.max_depth = depth;
        self
    }

    /// Build the configuration
    pub fn build(self) -> LibtreeConfig {
        self.config
    }
}

impl Default for LibtreeConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}