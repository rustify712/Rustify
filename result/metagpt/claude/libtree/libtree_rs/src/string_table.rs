//! String table management for libtree.
//!
//! This module provides functionality for storing and managing strings,
//! including string table operations and variable interpolation.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::collections::HashSet;
use std::os::unix::fs::MetadataExt;

use crate::error::{Error, Result};

/// Maximum path length for string operations
const MAX_PATH_LENGTH: usize = 4096;
/// Maximum recursion depth for dependency resolution
const MAX_RECURSION_DEPTH: usize = 32;

/// Internal state for string table and platform-specific variables
#[derive(Debug)]
pub(crate) struct InternalState {
    /// String buffer for storing all strings
    string_buffer: Vec<u8>,
    /// Platform-specific variable values
    platform: String,
    lib: String,
    osname: String,
    osrel: String,
    /// Offset to LD_LIBRARY_PATH in string buffer
    ld_library_path_offset: Option<usize>,
    /// Offset to default paths in string buffer
    default_paths_offset: Option<usize>,
    /// Offset to ld.so.conf paths in string buffer
    ld_so_conf_offset: Option<usize>,
    /// Stack of RPATH offsets for dependency resolution
    rpath_offsets: Vec<Option<usize>>,
    /// Set of visited files to prevent circular dependencies
    pub(crate) visited_files: HashSet<(u64, u64)>, // (dev, ino) pairs
    /// Track whether all needed libraries were found at each depth
    pub(crate) found_all_needed: Vec<bool>,
}

impl InternalState {
    /// Create a new empty internal state
    pub(crate) fn new() -> Self {
        Self {
            string_buffer: Vec::with_capacity(4096),
            platform: String::new(),
            lib: String::new(),
            osname: String::new(),
            osrel: String::new(),
            ld_library_path_offset: None,
            default_paths_offset: None,
            ld_so_conf_offset: None,
            rpath_offsets: vec![None; MAX_RECURSION_DEPTH],
            visited_files: HashSet::new(),
            found_all_needed: vec![false; MAX_RECURSION_DEPTH],
        }
    }

    /// Initialize platform-specific variables
    pub(crate) fn initialize_platform_vars(&mut self) -> Result<()> {
        // Set default library paths
        self.default_paths_offset = Some(self.string_buffer.len());
        self.store_string("/lib:/lib64:/usr/lib:/usr/lib64");

        // Determine platform-specific values
        #[cfg(target_arch = "x86_64")]
        {
            self.platform = "x86_64".to_string();
            self.lib = "lib64".to_string();
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.platform = "aarch64".to_string();
            self.lib = "lib64".to_string();
        }
        #[cfg(target_arch = "x86")]
        {
            self.platform = "i386".to_string();
            self.lib = "lib".to_string();
        }
        #[cfg(target_os = "linux")]
        {
            self.osname = "linux".to_string();
        }

        // Get kernel release for OSREL
        if let Ok(uname) = nix::sys::utsname::uname() {
            self.osrel = uname.release().to_string_lossy().into_owned();
        }

        Ok(())
    }

    /// Store a string in the buffer and return its offset
    pub(crate) fn store_string(&mut self, s: &str) -> usize {
        let offset = self.string_buffer.len();
        self.string_buffer.extend_from_slice(s.as_bytes());
        self.string_buffer.push(0);
        offset
    }

    /// Copy a string from a file into the buffer
    pub(crate) fn copy_from_file(&mut self, file: &mut File) -> io::Result<usize> {
        let offset = self.string_buffer.len();
        let mut buf = [0u8; 1];
        
        loop {
            file.read_exact(&mut buf)?;
            if buf[0] == 0 {
                self.string_buffer.push(0);
                break;
            }
            self.string_buffer.push(buf[0]);
        }
        
        Ok(offset)
    }

    /// Get a string from the buffer by offset
    pub(crate) fn get_string(&self, offset: usize) -> Option<&str> {
        if offset >= self.string_buffer.len() {
            return None;
        }

        let bytes = &self.string_buffer[offset..];
        let end = bytes.iter().position(|&b| b == 0)?;
        std::str::from_utf8(&bytes[..end]).ok()
    }

    /// Set the LD_LIBRARY_PATH
    pub(crate) fn set_ld_library_path(&mut self, path: &str) {
        self.ld_library_path_offset = Some(self.string_buffer.len());
        self.store_string(path);
    }

    /// Get the LD_LIBRARY_PATH
    pub(crate) fn get_ld_library_path(&self) -> Option<&str> {
        self.ld_library_path_offset.and_then(|offset| self.get_string(offset))
    }

    /// Initialize ld.so.conf paths
    pub(crate) fn init_ld_conf_paths(&mut self) {
        self.ld_so_conf_offset = Some(self.string_buffer.len());
    }

    /// Append a path to ld.so.conf paths
    pub(crate) fn append_ld_conf_path(&mut self, path: &str) {
        if let Some(offset) = self.ld_so_conf_offset {
            if offset < self.string_buffer.len() {
                self.string_buffer.push(b':');
            }
            self.string_buffer.extend_from_slice(path.as_bytes());
        }
    }

    /// Get the ld.so.conf paths
    pub(crate) fn get_ld_so_conf_paths(&self) -> Option<&str> {
        self.ld_so_conf_offset.and_then(|offset| self.get_string(offset))
    }

    /// Get the default library paths
    pub(crate) fn get_default_paths(&self) -> Option<&str> {
        self.default_paths_offset.and_then(|offset| self.get_string(offset))
    }

    /// Get all RPATH entries
    pub(crate) fn get_rpaths(&self) -> &[Option<usize>] {
        &self.rpath_offsets
    }

    /// Set RPATH at specific depth
    pub(crate) fn set_rpath(&mut self, depth: usize, path: &str) {
        if depth < MAX_RECURSION_DEPTH {
            self.rpath_offsets[depth] = Some(self.store_string(path));
        }
    }

    /// Interpolate variables in a path string
    pub(crate) fn interpolate_variables(&mut self, src_offset: usize, origin: &Path) -> Option<usize> {
        let src = self.get_string(src_offset)?;
        let mut result = String::with_capacity(src.len() * 2);
        let mut modified = false;
        let mut chars = src.chars().peekable();

        while let Some(c) = chars.next() {
            if c != '$' {
                result.push(c);
                continue;
            }

            let var_name = if chars.peek() == Some(&'{') {
                chars.next(); // skip '{'
                let mut name = String::new();
                while let Some(c) = chars.next() {
                    if c == '}' {
                        break;
                    }
                    name.push(c);
                }
                name
            } else {
                let mut name = String::new();
                while let Some(&c) = chars.peek() {
                    if !c.is_ascii_alphabetic() {
                        break;
                    }
                    name.push(chars.next().unwrap());
                }
                name
            };

            let var_value = match var_name.as_str() {
                "ORIGIN" => Some(origin.to_string_lossy()),
                "LIB" => Some(self.lib.as_str().into()),
                "PLATFORM" => Some(self.platform.as_str().into()),
                "OSNAME" => Some(self.osname.as_str().into()),
                "OSREL" => Some(self.osrel.as_str().into()),
                _ => None,
            };

            if let Some(value) = var_value {
                result.push_str(&value);
                modified = true;
            }
        }

        if modified {
            Some(self.store_string(&result))
        } else {
            None
        }
    }
}