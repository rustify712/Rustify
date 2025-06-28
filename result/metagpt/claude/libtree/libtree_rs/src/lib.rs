//! A library for analyzing ELF file dependencies.
//!
//! This library provides functionality to recursively analyze the dependencies
//! of ELF files (executables and shared libraries) and display them in a tree structure.
//! It supports various search paths including RPATH, RUNPATH, LD_LIBRARY_PATH, and
//! system default paths.

mod config;
mod elf;
mod error;
mod output;
mod search;
mod string_table;
mod utils;

use std::path::Path;

pub use config::{LibtreeConfig, LibtreeConfigBuilder};
pub use elf::{CompatType, ElfClass, ElfType};
pub use error::{Error, Result};
pub use output::OutputFormat;
pub use search::SearchMethod;

/// Main state structure for the libtree library.
pub struct LibtreeState {
    /// Verbosity level (0-3)
    pub verbosity: u32,
    /// Whether to show full paths
    pub show_path: bool,
    /// Whether to show search method
    pub config: LibtreeConfig,
    /// Whether to use color in output
    pub color: bool,
    /// Path to ld.so.conf file
    pub ld_conf_file: String,
    /// Maximum recursion depth
    pub max_depth: usize,
    /// Internal state
    pub(crate) internal: string_table::InternalState,
}

impl LibtreeState {
    /// Create a new LibtreeState with default settings.
    pub fn new() -> Self {
        let config = LibtreeConfig::default();
        Self {
            verbosity: 0,
            show_path: false,
            config,
            color: true,
            ld_conf_file: "/etc/ld.so.conf".to_string(),
            max_depth: 32,
            internal: string_table::InternalState::new(),
        }
    }

    /// Create a new LibtreeState with settings from a LibtreeConfig.
    pub fn from_config(config: &LibtreeConfig) -> Self {
        Self {
            verbosity: config.verbosity,
            show_path: config.show_path,
            config: config.clone(),
            color: config.color,
            ld_conf_file: config.ld_conf_file.clone(),
            max_depth: config.max_depth,
            internal: string_table::InternalState::new(),
        }
    }

    /// Analyze an ELF file and its dependencies.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ELF file to analyze
    /// * `output_format` - Format for the output
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Ok if successful, Err with an error code otherwise
    pub fn analyze<P: AsRef<Path>>(&mut self, path: P, output_format: OutputFormat) -> Result<()> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        search::analyze_file(&path_str, 0, self, None, SearchMethod::Input)?;
        output::print_results(self, output_format)
    }

    /// Set the LD_LIBRARY_PATH environment variable for dependency resolution.
    ///
    /// # Arguments
    ///
    /// * `ld_library_path` - Value of LD_LIBRARY_PATH
    pub fn set_ld_library_path(&mut self, ld_library_path: &str) {
        self.internal.set_ld_library_path(ld_library_path);
    }

    /// Initialize the state with system information.
    ///
    /// This function initializes platform-specific variables and loads
    /// the ld.so.conf configuration.
    pub fn initialize(&mut self) -> Result<()> {
        self.internal.initialize_platform_vars()?;
        config::parse_ld_so_conf(self)
    }
}

impl Default for LibtreeState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_state() {
        let state = LibtreeState::new();
        assert_eq!(state.verbosity, 0);
        assert_eq!(state.show_path, false);
        assert_eq!(state.color, true);
        assert_eq!(state.max_depth, 32);
        assert_eq!(state.ld_conf_file, "/etc/ld.so.conf");
    }

    #[test]
    fn test_from_config() {
        let config = LibtreeConfigBuilder::new()
            .verbosity(2)
            .show_path(true)
            .color(false)
            .max_depth(16)
            .ld_conf_file("/custom/ld.so.conf".to_string())
            .build();

        let state = LibtreeState::from_config(&config);
        assert_eq!(state.verbosity, 2);
        assert_eq!(state.show_path, true);
        assert_eq!(state.color, false);
        assert_eq!(state.max_depth, 16);
        assert_eq!(state.ld_conf_file, "/custom/ld.so.conf");
    }
}