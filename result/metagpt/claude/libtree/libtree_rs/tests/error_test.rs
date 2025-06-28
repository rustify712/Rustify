use libtree_rs::{Error, FileTree, TreeOptions};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_nonexistent_path() {
    let path = PathBuf::from("/path/that/does/not/exist");
    let options = TreeOptions::default();
    
    match FileTree::new(&path, &options) {
        Err(Error::IoError(e)) => {
            assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
        }
        other => panic!("Expected IoError for nonexistent path, got {:?}", other),
    }
}

#[test]
fn test_invalid_pattern() {
    let temp_dir = TempDir::new().unwrap();
    let options = TreeOptions {
        pattern: Some("[invalid regex".to_string()),
        ..Default::default()
    };
    
    match FileTree::new(temp_dir.path(), &options) {
        Err(Error::PatternError(e)) => {
            assert!(e.to_string().contains("bracket"), "Error message should mention the invalid bracket");
        }
        other => panic!("Expected PatternError for invalid regex pattern, got {:?}", other),
    }
}

#[test]
fn test_permission_denied() {
    let temp_dir = TempDir::new().unwrap();
    let restricted_dir = temp_dir.path().join("restricted");
    fs::create_dir(&restricted_dir).unwrap();
    
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = fs::metadata(&restricted_dir).unwrap();
        let mut perms = metadata.permissions();
        perms.set_mode(0o000);
        fs::set_permissions(&restricted_dir, perms).unwrap();
        
        let options = TreeOptions::default();
        match FileTree::new(&restricted_dir, &options) {
            Err(Error::IoError(e)) => {
                assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied);
            }
            other => panic!("Expected IoError for permission denied, got {:?}", other),
        }
        
        // Cleanup: restore permissions to allow directory removal
        perms.set_mode(0o755);
        fs::set_permissions(&restricted_dir, perms).unwrap();
    }
}

#[test]
fn test_max_depth_exceeded() {
    let temp_dir = TempDir::new().unwrap();
    let deep_dir = temp_dir.path();
    
    // Create a deeply nested directory structure
    let mut current_dir = deep_dir.to_path_buf();
    for i in 0..5 {
        current_dir = current_dir.join(format!("level_{}", i));
        fs::create_dir(&current_dir).unwrap();
        // Add a file in each directory to make the tree more realistic
        fs::write(current_dir.join("file.txt"), "content").unwrap();
    }
    
    let options = TreeOptions {
        max_depth: Some(3),
        ..Default::default()
    };
    
    let tree = FileTree::new(deep_dir, &options).unwrap();
    let output = tree.display();
    
    // Verify that only 3 levels are displayed
    let level_count = output.matches("level_").count();
    assert_eq!(level_count, 3);
    assert!(!output.contains("level_3"), "Should not contain level_3");
    assert!(!output.contains("level_4"), "Should not contain level_4");
}

#[test]
fn test_symlink_cycle() {
    let temp_dir = TempDir::new().unwrap();
    let dir_a = temp_dir.path().join("dir_a");
    let dir_b = temp_dir.path().join("dir_a/dir_b");
    
    fs::create_dir(&dir_a).unwrap();
    fs::create_dir(&dir_b).unwrap();
    
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&dir_a, dir_b.join("cycle")).unwrap();
        
        let options = TreeOptions {
            follow_links: true,
            ..Default::default()
        };
        
        match FileTree::new(temp_dir.path(), &options) {
            Err(Error::CyclicLink(path)) => {
                assert!(path.to_string_lossy().contains("cycle"), 
                       "Error should contain the cyclic link path");
            }
            other => panic!("Expected CyclicLink error for symlink cycle, got {:?}", other),
        }
    }
}

#[test]
fn test_empty_directory() {
    let temp_dir = TempDir::new().unwrap();
    let options = TreeOptions::default();
    
    let tree = FileTree::new(temp_dir.path(), &options).unwrap();
    let output = tree.display();
    
    assert!(output.contains("0 directories"), "Should report zero directories");
    assert!(output.contains("0 files"), "Should report zero files");
}

#[test]
fn test_invalid_depth_value() {
    let temp_dir = TempDir::new().unwrap();
    
    // Test with zero depth
    let options = TreeOptions {
        max_depth: Some(0),
        ..Default::default()
    };
    
    match FileTree::new(temp_dir.path(), &options) {
        Err(Error::InvalidDepth) => (),
        other => panic!("Expected InvalidDepth error for depth = 0, got {:?}", other),
    }
    
    // Test with negative depth (using i32::MIN as an example)
    let options = TreeOptions {
        max_depth: Some(-1),
        ..Default::default()
    };
    
    match FileTree::new(temp_dir.path(), &options) {
        Err(Error::InvalidDepth) => (),
        other => panic!("Expected InvalidDepth error for negative depth, got {:?}", other),
    }
}

#[test]
fn test_invalid_file_type() {
    let temp_dir = TempDir::new().unwrap();
    let special_file = temp_dir.path().join("special");
    
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileTypeExt;
        // Try to read a character device (like /dev/null)
        let options = TreeOptions::default();
        if let Ok(metadata) = fs::metadata("/dev/null") {
            if metadata.file_type().is_char_device() {
                match FileTree::new(std::path::Path::new("/dev/null"), &options) {
                    Err(Error::IoError(e)) => {
                        assert!(e.kind() == std::io::ErrorKind::Other || 
                               e.kind() == std::io::ErrorKind::InvalidInput,
                               "Expected error for special file");
                    }
                    other => panic!("Expected IoError for special file, got {:?}", other),
                }
            }
        }
    }
}