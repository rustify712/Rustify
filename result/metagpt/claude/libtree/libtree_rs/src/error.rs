use std::fmt;
use std::io;
use std::path::PathBuf;

/// Custom error type for the libtree library
#[derive(Debug)]
pub enum Error {
    /// I/O operation error
    Io(io::Error),
    /// File not found error with path
    FileNotFound(PathBuf),
    /// Invalid ELF file format
    InvalidElf(String),
    /// Error parsing ELF file
    ElfParse(String),
    /// Maximum recursion depth exceeded
    MaxDepthExceeded(usize),
    /// Library dependency not found
    DependencyNotFound(String),
    /// Error reading configuration file
    ConfigError(String),
    /// Invalid search path
    InvalidSearchPath(String),
    /// Platform not supported
    UnsupportedPlatform(String),
    /// Generic error with message
    Other(String),
}

/// Custom result type for the libtree library
pub type Result<T> = std::result::Result<T, Error>;

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(err) => write!(f, "I/O error: {}", err),
            Error::FileNotFound(path) => write!(
                f,
                "File not found: {}",
                path.to_string_lossy()
            ),
            Error::InvalidElf(msg) => write!(f, "Invalid ELF file: {}", msg),
            Error::ElfParse(msg) => write!(f, "ELF parsing error: {}", msg),
            Error::MaxDepthExceeded(depth) => write!(
                f,
                "Maximum recursion depth exceeded: {}",
                depth
            ),
            Error::DependencyNotFound(lib) => write!(
                f,
                "Library dependency not found: {}",
                lib
            ),
            Error::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            Error::InvalidSearchPath(path) => write!(
                f,
                "Invalid search path: {}",
                path
            ),
            Error::UnsupportedPlatform(msg) => write!(
                f,
                "Unsupported platform: {}",
                msg
            ),
            Error::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<String> for Error {
    fn from(err: String) -> Self {
        Error::Other(err)
    }
}

impl From<&str> for Error {
    fn from(err: &str) -> Self {
        Error::Other(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as StdError;

    #[test]
    fn test_error_display() {
        let err = Error::FileNotFound(PathBuf::from("/nonexistent"));
        assert_eq!(
            err.to_string(),
            "File not found: /nonexistent"
        );

        let err = Error::InvalidElf("Bad magic number".to_string());
        assert_eq!(
            err.to_string(),
            "Invalid ELF file: Bad magic number"
        );

        let err = Error::MaxDepthExceeded(32);
        assert_eq!(
            err.to_string(),
            "Maximum recursion depth exceeded: 32"
        );
    }

    #[test]
    fn test_error_conversion() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        match err {
            Error::Io(_) => (),
            _ => panic!("Expected Io error variant"),
        }

        let str_err: Error = "test error".into();
        match str_err {
            Error::Other(s) => assert_eq!(s, "test error"),
            _ => panic!("Expected Other error variant"),
        }
    }

    #[test]
    fn test_error_source() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err = Error::Io(io_err);
        assert!(err.source().is_some());

        let err = Error::InvalidElf("test".to_string());
        assert!(err.source().is_none());
    }
}