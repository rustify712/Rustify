use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

use crate::{
    Config, FileInfo, FileType, TreeBuilder, TreeError,
    pattern::{Pattern, PatternType},
};

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_directory() -> TempDir {
        let temp_dir = TempDir::new().unwrap();
        
        // Create a test directory structure
        fs::create_dir_all(temp_dir.path().join("dir1")).unwrap();
        fs::create_dir_all(temp_dir.path().join("dir1/subdir1")).unwrap();
        fs::create_dir_all(temp_dir.path().join("dir2")).unwrap();
        
        fs::write(temp_dir.path().join("file1.txt"), "content1").unwrap();
        fs::write(temp_dir.path().join("dir1/file2.txt"), "content2").unwrap();
        fs::write(temp_dir.path().join("dir1/subdir1/file3.txt"), "content3").unwrap();
        fs::write(temp_dir.path().join("dir2/file4.rs"), "content4").unwrap();
        
        temp_dir
    }

    #[test]
    fn test_basic_tree_traversal() {
        let temp_dir = create_test_directory();
        let config = Config::default();
        let mut builder = TreeBuilder::new(config);
        
        let result = builder.build(temp_dir.path());
        assert!(result.is_ok());
        
        let tree = result.unwrap();
        assert_eq!(tree.total_dirs(), 3); // root + dir1 + dir1/subdir1 + dir2
        assert_eq!(tree.total_files(), 4); // file1.txt + file2.txt + file3.txt + file4.rs
    }

    #[test]
    fn test_pattern_matching() {
        let temp_dir = create_test_directory();
        let mut config = Config::default();
        
        // Only match .txt files
        config.add_pattern(Pattern::new("*.txt", PatternType::Include));
        
        let mut builder = TreeBuilder::new(config);
        let result = builder.build(temp_dir.path());
        assert!(result.is_ok());
        
        let tree = result.unwrap();
        assert_eq!(tree.total_files(), 3); // Only .txt files should be counted
    }

    #[test]
    fn test_max_depth_limit() {
        let temp_dir = create_test_directory();
        let mut config = Config::default();
        config.max_depth = Some(1);
        
        let mut builder = TreeBuilder::new(config);
        let result = builder.build(temp_dir.path());
        assert!(result.is_ok());
        
        let tree = result.unwrap();
        assert_eq!(tree.total_dirs(), 2); // Only root + immediate subdirs
        assert_eq!(tree.total_files(), 1); // Only file1.txt in root
    }

    #[test]
    fn test_hidden_files() {
        let temp_dir = create_test_directory();
        fs::write(temp_dir.path().join(".hidden"), "hidden content").unwrap();
        
        let mut config = Config::default();
        config.show_hidden = false;
        
        let mut builder = TreeBuilder::new(config);
        let result = builder.build(temp_dir.path());
        assert!(result.is_ok());
        
        let tree = result.unwrap();
        assert!(!tree.files().iter().any(|f| f.name == ".hidden"));
    }

    #[test]
    fn test_file_info() {
        let temp_dir = create_test_directory();
        let test_file = temp_dir.path().join("file1.txt");
        
        let metadata = fs::metadata(&test_file).unwrap();
        let file_info = FileInfo::from_path(&test_file, &metadata).unwrap();
        
        assert_eq!(file_info.name, "file1.txt");
        assert_eq!(file_info.file_type, FileType::File);
        assert_eq!(file_info.size, 8); // "content1" is 8 bytes
    }

    #[test]
    fn test_invalid_path() {
        let config = Config::default();
        let mut builder = TreeBuilder::new(config);
        
        let result = builder.build(PathBuf::from("/nonexistent/path"));
        assert!(matches!(result, Err(TreeError::IoError(_))));
    }

    #[test]
    fn test_symlink_handling() {
        let temp_dir = create_test_directory();
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let link_path = temp_dir.path().join("symlink");
            symlink(temp_dir.path().join("file1.txt"), &link_path).unwrap();
            
            let config = Config::default();
            let mut builder = TreeBuilder::new(config);
            let result = builder.build(temp_dir.path());
            assert!(result.is_ok());
            
            let tree = result.unwrap();
            assert!(tree.files().iter().any(|f| f.file_type == FileType::Symlink));
        }
    }

    #[test]
    fn test_directory_only() {
        let temp_dir = create_test_directory();
        let mut config = Config::default();
        config.dirs_only = true;
        
        let mut builder = TreeBuilder::new(config);
        let result = builder.build(temp_dir.path());
        assert!(result.is_ok());
        
        let tree = result.unwrap();
        assert_eq!(tree.total_files(), 0);
        assert_eq!(tree.total_dirs(), 3);
    }
}