I need to complete the integration test file for the libtree_rs library. This file will test the main functionality of parsing ELF files and analyzing dependencies. The test was cut off in the middle, so I'll complete it and add more comprehensive tests.

```rust
//! Integration tests for the libtree library.
//!
//! These tests verify that the library correctly analyzes ELF files and their dependencies.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use libtree_rs::{LibtreeConfig, LibtreeConfigBuilder, LibtreeState, OutputFormat};

// Helper function to find a system library
fn find_system_library() -> PathBuf {
    // Try to find common system libraries
    let common_libs = [
        "/lib/x86_64-linux-gnu/libc.so.6",
        "/lib/libc.so.6",
        "/usr/lib/libc.so.6",
        "/lib64/libc.so.6",
        "/usr/lib64/libc.so.6",
    ];

    for lib in &common_libs {
        let path = Path::new(lib);
        if path.exists() {
            return path.to_path_buf();
        }
    }

    // If we can't find a specific library, try to find any .so file
    let output = Command::new("find")
        .args(["/usr/lib", "-name", "*.so", "-type", "f", "-print", "-quit"])
        .output()
        .expect("Failed to execute find command");

    let lib_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !lib_path.is_empty() {
        return PathBuf::from(lib_path);
    }

    panic!("Could not find any system library for testing");
}

// Helper function to create a simple test executable
fn create_test_executable() -> PathBuf {
    let temp_dir = env::temp_dir();
    let test_file = temp_dir.join("libtree_test_executable");
    
    // Create a simple C program that links to libc
    let c_file = temp_dir.join("test.c");
    let mut file = File::create(&c_file).expect("Failed to create test C file");
    file.write_all(b"
        #include <stdio.h>
        int main() {
            printf(\"Hello, World!\\n\");
            return 0;
        }
    ").expect("Failed to write to test C file");
    
    // Compile the program
    let status = Command::new("gcc")
        .args([c_file.to_str().unwrap(), "-o", test_file.to_str().unwrap()])
        .status()
        .expect("Failed to execute gcc");
    
    if !status.success() {
        panic!("Failed to compile test executable");
    }
    
    test_file
}

#[test]
fn test_analyze_system_library() {
    // Find a system library to analyze
    let lib_path = find_system_library();
    println!("Testing with system library: {}", lib_path.display());
    
    // Create and initialize the libtree state
    let config = LibtreeConfigBuilder::new()
        .verbosity(1)
        .color(false)
        .build();
    
    let mut state = LibtreeState::from_config(&config);
    state.initialize().expect("Failed to initialize libtree state");
    
    // Analyze the library
    let result = state.analyze(&lib_path, OutputFormat::Tree);
    assert!(result.is_ok(), "Failed to analyze library: {:?}", result.err());
}

#[test]
fn test_analyze_executable() {
    // Create a test executable
    let executable_path = create_test_executable();
    println!("Testing with executable: {}", executable_path.display());
    
    // Create and initialize the libtree state
    let config = LibtreeConfigBuilder::new()
        .verbosity(1)
        .color