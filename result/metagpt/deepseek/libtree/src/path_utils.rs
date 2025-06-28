//! 路径处理工具模块

use std::path::{Path, PathBuf};
use std::ffi::OsStr;
use std::fs;
use std::io;

use crate::elf::ElfError;

/// 检查路径是否为绝对路径
pub fn is_absolute_path(path: &str) -> bool {
    Path::new(path).is_absolute()
}

/// 拼接路径和库名
pub fn join_path_with_lib(path: &str, libname: &str) -> PathBuf {
    let mut path_buf = PathBuf::from(path);
    if !path_buf.to_string_lossy().ends_with('/') {
        path_buf.push("/");
    }
    path_buf.push(libname);
    path_buf
}

/// 在搜索路径中查找库文件
pub fn find_library_in_paths(
    libname: &str,
    search_paths: &[String],
) -> Result<PathBuf, ElfError> {
    for path in search_paths {
        let full_path = join_path_with_lib(path, libname);
        if let Ok(metadata) = fs::metadata(&full_path) {
            if metadata.is_file() {
                return Ok(full_path);
            }
        }
    }
    Err(ElfError::DependencyNotFound)
}

/// 解析rpath/rpath中的特殊变量
pub fn expand_rpath_tokens(rpath: &str, platform: &str, lib: &str) -> String {
    rpath.replace("$ORIGIN", ".")
        .replace("${ORIGIN}", ".")
        .replace("$PLATFORM", platform)
        .replace("${PLATFORM}", platform)
        .replace("$LIB", lib)
        .replace("${LIB}", lib)
}

/// 规范化路径分隔符
pub fn normalize_path_separators(path: &str) -> String {
    if cfg!(windows) {
        path.replace("/", "\\")
    } else {
        path.replace("\\", "/")
    }
}

/// 读取符号链接并解析为绝对路径
pub fn resolve_symlink(path: &Path) -> io::Result<PathBuf> {
    let mut current = path.to_path_buf();
    let mut visited = Vec::new();

    while fs::symlink_metadata(&current)?.file_type().is_symlink() {
        if visited.contains(&current) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Symlink loop detected",
            ));
        }
        visited.push(current.clone());
        current = fs::read_link(&current)?;
    }

    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_absolute_path() {
        assert!(is_absolute_path("/usr/lib"));
        assert!(!is_absolute_path("lib"));
    }

    #[test]
    fn test_join_path_with_lib() {
        let path = join_path_with_lib("/usr/lib", "libc.so");
        assert_eq!(path, PathBuf::from("/usr/lib/libc.so"));
    }

    #[test]
    fn test_expand_rpath_tokens() {
        let rpath = "$ORIGIN:/usr/${LIB}:$PLATFORM";
        let expanded = expand_rpath_tokens(rpath, "x86_64", "lib64");
        assert_eq!(expanded, ".:/usr/lib64:x86_64");
    }
}