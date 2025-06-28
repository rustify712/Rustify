我需要创建一个测试配置相关功能的单元测试文件。根据之前的分析，我们已经实现了一个 libtree_rs 库，它是 libtree.c 的 Rust 版本。现在我需要为配置相关功能编写测试。

```rust
//! Tests for the configuration functionality of libtree_rs.
//!
//! This file contains unit tests for the configuration module, including
//! the LibtreeConfig and LibtreeConfigBuilder structs.

use libtree_rs::{LibtreeConfig, LibtreeConfigBuilder};

#[test]
fn test_default_config() {
    let config = LibtreeConfig::default();
    
    // Check default values
    assert_eq!(config.verbosity, 0);
    assert_eq!(config.show_path, false);
    assert_eq!(config.color, true);
    assert_eq!(config.max_depth, 32);
    assert_eq!(config.ld_conf_file, "/etc/ld.so.conf".to_string());
}

#[test]
fn test_config_builder() {
    // Test building a config with custom values
    let config = LibtreeConfigBuilder::new()
        .verbosity(2)
        .show_path(true)
        .color(false)
        .max_depth(16)
        .ld_conf_file("/custom/path.conf")
        .build();
    
    // Verify all values were set correctly
    assert_eq!(config.verbosity, 2);
    assert_eq!(config.show_path, true);
    assert_eq!(config.color, false);
    assert_eq!(config.max_depth, 16);
    assert_eq!(config.ld_conf_file, "/custom/path.conf".to_string());
}

#[test]
fn test_config_builder_chaining() {
    // Test that builder methods can be chained in any order
    let config1 = LibtreeConfigBuilder::new()
        .verbosity(1)
        .color(false)
        .show_path(true)
        .build();
    
    let config2 = LibtreeConfigBuilder::new()
        .color(false)
        .show_path(true)
        .verbosity(1)
        .build();
    
    // Both configs should have the same values
    assert_eq!(config1.verbosity, config2.verbosity);
    assert_eq!(config1.show_path, config2.show_path);
    assert_eq!(config1.color, config2.color);
    assert_eq!(config1.max_depth, config2.max_depth);
    assert_eq!(config1.ld_conf_file, config2.ld_conf_file);
}

#[test]
fn test_verbosity_levels() {
    // Test different verbosity levels
    for level in 0..4 {
        let config = LibtreeConfigBuilder::new()
            .verbosity(level)
            .build();
        
        assert_eq!(config.verbosity, level);
    }
}

#[test]
fn test_from_config() {
    // Test creating a new config from an existing one
    let original = LibtreeConfigBuilder::new()
        .verbosity(3)
        .show_path(true)
        .build();
    
    let derived = LibtreeConfig {
        verbosity: original.verbosity,
        show_path: original.show_path,
        color: original.color,
        max_depth: original.max_depth,
        ld_conf_file: original.ld_conf_file.clone(),
    };
    
    // Values should match
    assert_eq!(original.verbosity, derived.verbosity);
    assert_eq!(original.show_path, derived.show_path);
    assert_eq!(original.color, derived.color);
    assert_eq!(original.max_depth, derived.max_depth);
    assert_eq!(original.ld_conf_file, derived.ld_conf_file);
}

#[test]
fn test_