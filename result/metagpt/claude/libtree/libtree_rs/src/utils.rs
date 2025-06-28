use std::ffi::{CStr, CString, OsStr};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use crate::error::{Error, Result};

/// Convert a Path to a CString
///
/// This function converts a Path to a CString, which is useful when interacting
/// with C APIs. Returns an error if the path contains invalid UTF-8 characters
/// or internal null bytes.
pub fn path_to_cstring(path: &Path) -> Result<CString> {
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| Error::Other("Path contains null bytes".to_string()))
}

/// Convert a CStr to a PathBuf
///
/// This function safely converts a CStr to a PathBuf. It's particularly useful
/// when working with paths received from C APIs.
pub fn cstr_to_path_buf(cstr: &CStr) -> PathBuf {
    let bytes = cstr.to_bytes();
    let os_str = OsStr::from_bytes(bytes);
    PathBuf::from(os_str)
}

/// Normalize a file path by removing redundant separators and resolving . and ..
///
/// This function takes a path and returns a normalized version of it:
/// - Removes duplicate path separators
/// - Resolves . and .. components
/// - Maintains absolute/relative path status
pub fn normalize_path(path: &Path) -> Result<PathBuf> {
    let mut components = Vec::new();
    let is_absolute = path.is_absolute();

    for component in path.components() {
        match component {
            std::path::Component::Prefix(p) => components.push(p.as_os_str().to_owned()),
            std::path::Component::RootDir => {
                if components.is_empty() {
                    components.push(OsStr::new("/").to_owned());
                }
            }
            std::path::Component::CurDir => (), // Skip .
            std::path::Component::ParentDir => {
                if !components.is_empty() && components.last().unwrap() != ".." {
                    components.pop();
                } else if !is_absolute {
                    components.push(OsStr::new("..").to_owned());
                }
            }
            std::path::Component::Normal(x) => components.push(x.to_owned()),
        }
    }

    let mut result = PathBuf::new();
    if is_absolute {
        result.push("/");
    }
    for component in components {
        result.push(component);
    }
    Ok(result)
}

/// Join multiple path components together
///
/// This function joins multiple path components together, handling both
/// absolute and relative paths correctly.
pub fn join_paths<P: AsRef<Path>>(paths: &[P]) -> Result<PathBuf> {
    let mut result = PathBuf::new();
    for path in paths {
        if path.as_ref().is_absolute() {
            result = path.as_ref().to_path_buf();
        } else {
            result.push(path);
        }
    }
    normalize_path(&result)
}

/// Check if a path exists and is a file
pub fn is_file(path: &Path) -> bool {
    path.is_file()
}

/// Check if a path exists and is a directory
pub fn is_directory(path: &Path) -> bool {
    path.is_dir()
}

/// Check if a path exists and is readable
pub fn is_readable(path: &Path) -> bool {
    use std::fs::File;
    File::open(path).is_ok()
}

/// Check if a path has execute permissions
#[cfg(unix)]
pub fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(metadata) = path.metadata() {
        return metadata.permissions().mode() & 0o111 != 0;
    }
    false
}

#[cfg(not(unix))]
pub fn is_executable(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile::tempdir;

    #[test]
    fn test_path_conversion() {
        let path = Path::new("/usr/lib/test");
        let cstring = path_to_cstring(path).unwrap();
        let path_buf = cstr_to_path_buf(&cstring);
        assert_eq!(path, path_buf);
    }

    #[test]
    fn test_normalize_path() {
        let test_cases = vec![
            ("/usr/./lib/../bin", "/usr/bin"),
            ("./test/../foo", "foo"),
            ("/usr//lib//test", "/usr/lib/test"),
            ("../test/./foo", "../test/foo"),
        ];

        for (input, expected) in test_cases {
            let normalized = normalize_path(Path::new(input)).unwrap();
            assert_eq!(normalized, PathBuf::from(expected));
        }
    }

    #[test]
    fn test_join_paths() {
        let paths = vec![
            "usr",
            "local",
            "bin",
        ];
        let joined = join_paths(&paths).unwrap();
        assert_eq!(joined, PathBuf::from("usr/local/bin"));

        let paths = vec![
            "/usr",
            "local",
            "bin",
        ];
        let joined = join_paths(&paths).unwrap();
        assert_eq!(joined, PathBuf::from("/usr/local/bin"));
    }

    #[test]
    fn test_file_checks() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        File::create(&file_path).unwrap();

        assert!(is_file(&file_path));
        assert!(is_directory(dir.path()));
        assert!(is_readable(&file_path));
    }
}