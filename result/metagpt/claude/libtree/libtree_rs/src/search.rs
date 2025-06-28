//! Library search and dependency analysis functionality.
//!
//! This module handles searching for library dependencies and analyzing ELF files
//! recursively to build the dependency tree.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use crate::elf::{CompatType, ElfClass, ElfParser, DynamicInfo};
use crate::error::{Error, Result};
use crate::output::{print_error, print_line, print_search_paths, OutputFormat};
use crate::LibtreeState;

/// How a library was found
#[derive(Debug, Clone, Copy)]
pub enum SearchMethod {
    /// Input file specified by user
    Input,
    /// Direct path specified in DT_NEEDED
    Direct,
    /// Found via RPATH
    Rpath { depth: usize },
    /// Found via LD_LIBRARY_PATH
    LdLibraryPath,
    /// Found via RUNPATH
    Runpath,
    /// Found via ld.so.conf
    LdSoConf,
    /// Found via default paths
    Default,
}

/// Check if a library should be excluded from detailed analysis
fn is_excluded(soname: &str) -> bool {
    const EXCLUDE_LIST: &[&str] = &[
        "ld-linux-aarch64.so",
        "ld-linux-armhf.so",
        "ld-linux-x86-64.so",
        "ld-linux.so",
        "ld64.so",
        "libc.musl-aarch64.so",
        "libc.musl-armhf.so",
        "libc.musl-i386.so",
        "libc.musl-x86_64.so",
        "libc.so",
        "libdl.so",
        "libgcc_s.so",
        "libm.so",
        "libstdc++.so",
    ];

    let base_name = soname.trim_end_matches(|c: char| c.is_ascii_digit() || c == '.');
    EXCLUDE_LIST.iter().any(|&excluded| base_name.starts_with(excluded))
}

/// Search for a library in a colon-separated list of paths
fn search_in_paths(
    needed: &str,
    paths: &str,
    depth: usize,
    state: &mut LibtreeState,
    compat: &CompatType,
    method: SearchMethod,
) -> Result<bool> {
    for path in paths.split(':') {
        if path.is_empty() {
            continue;
        }

        let mut full_path = PathBuf::from(path);
        full_path.push(needed);
        
        let full_path_str = full_path.to_string_lossy().to_string();
        
        match analyze_file(&full_path_str, depth + 1, state, Some(compat), method) {
            Ok(_) => return Ok(true),
            Err(Error::DependencyNotFound(_)) => continue,
            Err(Error::CouldNotOpenFile(_, _)) => continue,
            Err(e) => return Err(e),
        }
    }
    
    Ok(false)
}

/// Analyze an ELF file and its dependencies
pub fn analyze_file(
    path: &str,
    depth: usize,
    state: &mut LibtreeState,
    compat: Option<&CompatType>,
    method: SearchMethod,
) -> Result<()> {
    // Open and parse the ELF file
    let file = File::open(path).map_err(|e| Error::CouldNotOpenFile(PathBuf::from(path), e))?;
    let metadata = file.metadata()?;
    
    // Check if we've seen this file before
    let dev = metadata.dev();
    let ino = metadata.ino();
    let seen_before = is_visited(state, dev, ino);
    
    if !seen_before {
        mark_visited(state, dev, ino);
    }

    let mut parser = ElfParser::new(file)?;
    let dynamic_info = parser.parse_dynamic_section()?;
    
    // Determine if we should exclude this library from detailed analysis
    let is_excluded = dynamic_info.soname.as_ref().map_or(false, |s| is_excluded(s));

    // Determine if we should recurse deeper
    let should_recurse = depth < state.max_depth && (
        (!seen_before && !is_excluded) ||
        (!seen_before && is_excluded && state.verbosity >= 2) ||
        state.verbosity >= 3
    );

    // Print the current library
    let print_name = if state.show_path {
        path
    } else {
        dynamic_info.soname.as_deref().unwrap_or(path)
    };

    print_line(depth, print_name, seen_before, is_excluded, method, state);

    if !should_recurse {
        return Ok(());
    }

    // Process needed libraries
    let mut exit_code = Ok(());
    let mut found_count = 0;
    let total_needed = dynamic_info.needed.len();

    for needed in dynamic_info.needed {
        let found = if needed.contains('/') {
            // Absolute or relative path
            analyze_file(&needed, depth + 1, state, compat, SearchMethod::Direct).is_ok()
        } else {
            // Try all search paths in order
            let mut found = false;

            // Search in RPATH if no RUNPATH
            if dynamic_info.runpath.is_none() {
                for (i, rpath) in dynamic_info.rpath.iter().enumerate().rev() {
                    if search_in_paths(
                        &needed,
                        rpath,
                        depth,
                        state,
                        compat.unwrap(),
                        SearchMethod::Rpath { depth: i },
                    )? {
                        found = true;
                        break;
                    }
                }
            }

            // Search in LD_LIBRARY_PATH
            if !found {
                if let Some(ld_path) = state.internal.get_ld_library_path() {
                    found = search_in_paths(
                        &needed,
                        ld_path,
                        depth,
                        state,
                        compat.unwrap(),
                        SearchMethod::LdLibraryPath,
                    )?;
                }
            }

            // Search in RUNPATH
            if !found {
                if let Some(runpath) = &dynamic_info.runpath {
                    found = search_in_paths(
                        &needed,
                        runpath,
                        depth,
                        state,
                        compat.unwrap(),
                        SearchMethod::Runpath,
                    )?;
                }
            }

            // Search in ld.so.conf paths
            if !found {
                if let Some(conf_paths) = state.internal.get_ld_so_conf_paths() {
                    found = search_in_paths(
                        &needed,
                        conf_paths,
                        depth,
                        state,
                        compat.unwrap(),
                        SearchMethod::LdSoConf,
                    )?;
                }
            }

            // Search in default paths
            if !found {
                if let Some(default_paths) = state.internal.get_default_paths() {
                    found = search_in_paths(
                        &needed,
                        default_paths,
                        depth,
                        state,
                        compat.unwrap(),
                        SearchMethod::Default,
                    )?;
                }
            }

            found
        };

        if found {
            found_count += 1;
        } else {
            print_error(depth + 1, &needed, state);
            exit_code = Err(Error::DependencyNotFound(PathBuf::from(&needed)));
        }
    }

    // Update found_all_needed status
    state.internal.found_all_needed[depth] = found_count == total_needed;

    // Print search paths if any dependencies were not found
    if found_count < total_needed {
        print_search_paths(
            depth,
            state,
            dynamic_info.runpath.as_deref(),
            dynamic_info.no_default_lib,
        );
    }

    exit_code
}

/// Check if a file has been visited before
fn is_visited(state: &LibtreeState, dev: u64, ino: u64) -> bool {
    state.internal.visited_files.contains(&(dev, ino))
}

/// Mark a file as visited
fn mark_visited(state: &mut LibtreeState, dev: u64, ino: u64) {
    state.internal.visited_files.insert((dev, ino));
}