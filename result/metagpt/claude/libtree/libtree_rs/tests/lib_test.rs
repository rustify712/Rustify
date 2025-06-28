I'll write a comprehensive test file for the libtree_rs library that tests all the core functionality. This will include tests for configuration, ELF parsing, dependency analysis, and error handling.

```rust
//! Integration tests for the libtree_rs library.
//!
//! These tests verify the core functionality of the library components.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

use libtree_rs::{
    LibtreeConfig, LibtreeConfigBuilder, LibtreeState, OutputFormat,
    error::{Error, Result},
    search::SearchMethod,
};

// Helper function to find a system ELF binary for testing
fn find_test_binary() -> PathBuf {
    let common_binaries = [
        "/bin/ls",
        "/bin/bash",
        "/usr/bin/env",
        "/bin/cat",
    ];

    for binary in &common_binaries {
        let path = Path::new(binary);
        if path.exists() {
            return path.to_path_buf();
        }
    }

    panic!("Could not find any system binary for testing");
}

// Helper function to create a simple test ELF file
fn create_test_elf_file() -> Result<PathBuf> {
    let dir = tempdir().map_err(|e| Error::Io(e))?;
    let test_file = dir.path().join("test_elf");
    
    // Create a minimal ELF header (not a valid ELF file, but enough to test error handling)
    let mut file = File::create(&test_file).map_err(|e| Error::Io(e))?;
    
    // ELF magic number
    file.write_all(&[0x7f, b'E', b'L', b'F']).map_err(|e| Error::Io(e))?;
    // 64-bit class
    file.write_all(&[2]).map_err(|e| Error::Io(e))?;
    // Little endian
    file.write_all(&[1]).map_err(|e| Error::Io(e))?;
    // ELF version
    file.write_all(&[1]).map_err(|e| Error::Io(e))?;
    // Padding to complete e_ident
    file.write_all(&[0; 9]).map_err(|e| Error::Io(e))?;
    
    // Minimal header data (not complete, but enough for basic tests)
    // e_type = ET_EXEC (2)
    file.write_all(&[2, 0]).map_err(|e| Error::Io(e))?;
    // e_machine = EM_X86_64 (62)
    file.write_all(&[62, 0]).map_err(|e| Error::Io(e))?;
    
    Ok(test_file)
}

// Helper function to create an invalid file for testing error handling
fn create_invalid_file() -> Result<PathBuf> {
    let dir = tempdir().map_err(|e| Error::Io(e))?;
    let test_file = dir.path().join("invalid_file");
    
    let mut file = File::create(&test_file).map_err(|e| Error::Io(e))?;
    file.write_all(b"This is not an ELF file").map_err(|e| Error::Io(e))?;
    
    Ok(test_file)
}

#[test]
fn test_config_builder() {
    let config = LibtreeConfigBuilder::new()
        .verbosity(2)
        .show_path(true)
        .color(false)
        .max_depth(16)
        .ld_conf_file("/custom/path.conf")
        .build();
    
    assert_eq!(config.verbosity, 2);
    assert_eq!(config.show_path, true);
    assert_eq!(config.color, false);
    assert