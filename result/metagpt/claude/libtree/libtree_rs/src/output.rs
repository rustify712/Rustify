//! Output formatting for dependency tree display.
//!
//! This module provides functionality for displaying the dependency tree
//! and formatting error messages.

use std::fmt::Write;
use std::path::Path;

use crate::LibtreeState;
use crate::search::SearchMethod;
use crate::error::LibtreeError;

/// Unicode box drawing characters for tree display
const LIGHT_HORIZONTAL: &str = "─";
const LIGHT_QUADRUPLE_DASH_VERTICAL: &str = "┊";
const LIGHT_UP_AND_RIGHT: &str = "└";
const LIGHT_VERTICAL: &str = "│";
const LIGHT_VERTICAL_AND_RIGHT: &str = "├";

/// Indentation constants
const JUST_INDENT: &str = "    ";
const LIGHT_VERTICAL_WITH_INDENT: &str = "│   ";

/// ANSI color codes
const REGULAR_RED: &str = "\x1b[0;31m";
const BOLD_RED: &str = "\x1b[1;31m";
const CLEAR: &str = "\x1b[0m";
const BOLD_YELLOW: &str = "\x1b[33m";
const BOLD_CYAN: &str = "\x1b[1;36m";
const REGULAR_CYAN: &str = "\x1b[0;36m";
const REGULAR_MAGENTA: &str = "\x1b[0;35m";
const REGULAR_BLUE: &str = "\x1b[0;34m";
const BRIGHT_BLACK: &str = "\x1b[0;90m";
const REGULAR: &str = "\x1b[0m";

/// Output format options
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    /// Tree-like structure with indentation
    Tree,
    /// Flat list of dependencies
    List,
    /// JSON output
    Json,
}

/// Print the tree preamble (indentation and branch characters)
fn tree_preamble(state: &LibtreeState, depth: usize) -> String {
    let mut result = String::new();
    
    if depth == 0 {
        return result;
    }
    
    for i in 0..depth - 1 {
        let indent = if state.internal.found_all_needed[i] {
            JUST_INDENT
        } else {
            LIGHT_VERTICAL_WITH_INDENT
        };
        result.push_str(indent);
    }
    
    let branch = if state.internal.found_all_needed[depth - 1] {
        format!("{}{}{} ", LIGHT_UP_AND_RIGHT, LIGHT_HORIZONTAL, LIGHT_HORIZONTAL)
    } else {
        format!("{}{}{} ", LIGHT_VERTICAL_AND_RIGHT, LIGHT_HORIZONTAL, LIGHT_HORIZONTAL)
    };
    
    result.push_str(&branch);
    result
}

/// Format search method for display
fn format_search_method(method: SearchMethod, state: &LibtreeState) -> String {
    match method {
        SearchMethod::Input => String::new(),
        SearchMethod::Direct => "[direct]".to_string(),
        SearchMethod::Rpath { depth } => {
            if depth + 1 >= state.max_depth {
                "[rpath]".to_string()
            } else {
                format!("[rpath of {}]", depth + 1)
            }
        }
        SearchMethod::LdLibraryPath => "[LD_LIBRARY_PATH]".to_string(),
        SearchMethod::Runpath => "[runpath]".to_string(),
        SearchMethod::LdSoConf => {
            let conf_name = Path::new(&state.ld_conf_file)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("ld.so.conf");
            format!("[{}]", conf_name)
        }
        SearchMethod::Default => "[default path]".to_string(),
    }
}

/// Format a library path for display
fn format_library_path(path: &str, method: SearchMethod, state: &LibtreeState) -> String {
    let mut result = String::new();
    
    if state.config.show_search_method {
        let method_str = format_search_method(method, state);
        if !method_str.is_empty() {
            result.push_str(&format!("{}{}{} ", BRIGHT_BLACK, method_str, CLEAR));
        }
    }
    
    result.push_str(path);
    result
}

/// Format error messages with appropriate colors
pub fn format_error(err: &LibtreeError) -> String {
    match err {
        LibtreeError::NotFound(path) => {
            format!("{}{}{}: not found", BOLD_RED, path, CLEAR)
        }
        LibtreeError::InvalidElf(path) => {
            format!("{}{}{}: not a valid ELF file", BOLD_RED, path, CLEAR)
        }
        LibtreeError::IoError(path, err) => {
            format!("{}{}{}: {}", BOLD_RED, path, CLEAR, err)
        }
        LibtreeError::ConfigError(msg) => {
            format!("{}Configuration error{}: {}", BOLD_RED, CLEAR, msg)
        }
    }
}

/// Format and display the dependency tree
pub fn print_tree(state: &LibtreeState, format: OutputFormat) -> Result<String, std::fmt::Error> {
    let mut output = String::new();
    
    match format {
        OutputFormat::Tree => {
            for (depth, entry) in state.dependency_tree.iter().enumerate() {
                let preamble = tree_preamble(state, depth);
                let path_str = format_library_path(&entry.path, entry.search_method, state);
                
                writeln!(&mut output, "{}{}", preamble, path_str)?;
                
                if let Some(ref error) = entry.error {
                    writeln!(
                        &mut output,
                        "{}{}",
                        tree_preamble(state, depth + 1),
                        format_error(error)
                    )?;
                }
            }
        }
        OutputFormat::List => {
            for entry in state.dependency_tree.iter() {
                let path_str = format_library_path(&entry.path, entry.search_method, state);
                writeln!(&mut output, "{}", path_str)?;
                
                if let Some(ref error) = entry.error {
                    writeln!(&mut output, "  {}", format_error(error))?;
                }
            }
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&state.dependency_tree)
                .map_err(|_| std::fmt::Error)?;
            write!(&mut output, "{}", json)?;
        }
    }
    
    Ok(output)
}

/// Print warning messages
pub fn print_warning(msg: &str) -> String {
    format!("{}Warning{}: {}", BOLD_YELLOW, CLEAR, msg)
}

/// Print debug information
pub fn print_debug(msg: &str) -> String {
    format!("{}Debug{}: {}", REGULAR_BLUE, CLEAR, msg)
}