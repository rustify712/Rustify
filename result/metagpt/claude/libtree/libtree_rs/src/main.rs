//! Command-line interface for libtree.
//!
//! This module provides the entry point for the libtree program,
//! parsing command line arguments and calling the library functions.

use std::env;
use std::process;

use libtree_rs::{LibtreeConfig, LibtreeConfigBuilder, LibtreeState, OutputFormat};

fn print_usage() {
    println!("libtree v{} - Display library dependencies as a tree", env!("CARGO_PKG_VERSION"));
    println!("Usage: libtree [options] <path-to-binary>");
    println!("Options:");
    println!("  -h, --help            Display this help message");
    println!("  -v, --version         Display version information");
    println!("  -V, --verbose         Increase verbosity (can be used multiple times)");
    println!("  -p, --path            Show full paths");
    println!("  -n, --nocolor         Disable colored output");
    println!("  -d, --max-depth=N     Set maximum recursion depth (default: 32)");
    println!("  -c, --config=FILE     Use FILE as ld.so.conf instead of default");
}

fn print_version() {
    println!("libtree v{}", env!("CARGO_PKG_VERSION"));
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }
    
    let mut config_builder = LibtreeConfigBuilder::new();
    let mut binary_path = None;
    
    // Parse command line arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            "-v" | "--version" => {
                print_version();
                process::exit(0);
            }
            "-V" | "--verbose" => {
                config_builder = config_builder.verbosity(1);
            }
            "-VV" => {
                config_builder = config_builder.verbosity(2);
            }
            "-VVV" => {
                config_builder = config_builder.verbosity(3);
            }
            "-p" | "--path" => {
                config_builder = config_builder.show_path(true);
            }
            "-n" | "--nocolor" => {
                config_builder = config_builder.color(false);
            }
            arg if arg.starts_with("-d=") || arg.starts_with("--max-depth=") => {
                let parts: Vec<&str> = arg.splitn(2, '=').collect();
                if parts.len() == 2 {
                    if let Ok(depth) = parts[1].parse::<usize>() {
                        config_builder = config_builder.max_depth(depth);
                    } else {
                        eprintln!("Error: Invalid depth value: {}", parts[1]);
                        process::exit(1);
                    }
                }
            }
            arg if arg.starts_with("-c=") || arg.starts_with("--config=") => {
                let parts: Vec<&str> = arg.splitn(2, '=').collect();
                if parts.len() == 2 {
                    config_builder = config_builder.ld_conf_file(parts[1]);
                }
            }
            arg if arg.starts_with('-') => {
                eprintln!("Error: Unknown option: {}", arg);
                print_usage();
                process::exit(1);
            }
            _ => {
                binary_path = Some(args[i].clone());
                break;
            }
        }
        i += 1;
    }
    
    // Check if we have a binary path
    let binary_path = match binary_path {
        Some(path) => path,
        None => {
            eprintln!("Error: No binary path specified");
            print_usage();
            process::exit(1);
        }
    };

    // Create and initialize the libtree state
    let config = config_builder.build();
    let mut state = LibtreeState::from_config(&config);

    // Initialize the state with system information
    if let Err(e) = state.initialize() {
        eprintln!("Error initializing libtree: {}", e);
        process::exit(1);
    }

    // Set LD_LIBRARY_PATH if present in environment
    if let Ok(ld_library_path) = env::var("LD_LIBRARY_PATH") {
        state.set_ld_library_path(&ld_library_path);
    }

    // Analyze the binary and its dependencies
    match state.analyze(&binary_path, OutputFormat::Tree) {
        Ok(_) => process::exit(0),
        Err(e) => {
            eprintln!("Error analyzing {}: {}", binary_path, e);
            process::exit(1);
        }
    }
}