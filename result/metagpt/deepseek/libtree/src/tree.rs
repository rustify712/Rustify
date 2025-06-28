//! 依赖树生成和显示模块

use std::path::{Path, PathBuf};
use std::collections::HashSet;
use colored::Colorize;

/// 依赖树节点
#[derive(Debug, PartialEq)]
struct TreeNode {
    path: PathBuf,
    children: Vec<TreeNode>,
}

/// 依赖树结构
pub struct DependencyTree {
    root: TreeNode,
    visited: HashSet<PathBuf>,
    max_depth: usize,
}

impl DependencyTree {
    /// 创建新的依赖树
    pub fn new(max_depth: usize) -> Self {
        DependencyTree {
            root: TreeNode {
                path: PathBuf::new(),
                children: Vec::new(),
            },
            visited: HashSet::new(),
            max_depth,
        }
    }

    /// 添加依赖到树中
    pub fn add_dependency(&mut self, path: &Path, depth: usize) {
        if depth > self.max_depth || self.visited.contains(path) {
            return;
        }

        self.visited.insert(path.to_path_buf());
        let mut current = &mut self.root;
        
        for component in path.components() {
            let component_path = current.path.join(component);
            let mut found = None;
            
            for (i, child) in current.children.iter_mut().enumerate() {
                if child.path == component_path {
                    found = Some(i);
                    break;
                }
            }
            
            if let Some(i) = found {
                current = &mut current.children[i];
            } else {
                current.children.push(TreeNode {
                    path: component_path.clone(),
                    children: Vec::new(),
                });
                current = current.children.last_mut().unwrap();
            }
        }
    }

    /// 打印依赖树
    pub fn print(&self, full_path: bool) {
        self.print_node(&self.root, 0, full_path, true, Vec::new());
    }

    /// 递归打印节点
    fn print_node(
        &self,
        node: &TreeNode,
        depth: usize,
        full_path: bool,
        is_last: bool,
        mut continuation_lines: Vec<bool>,
    ) {
        if node.path == Path::new("") {
            for (i, child) in node.children.iter().enumerate() {
                let is_last_child = i == node.children.len() - 1;
                self.print_node(
                    child,
                    depth,
                    full_path,
                    is_last_child,
                    continuation_lines.clone(),
                );
            }
            return;
        }

        // 打印前缀
        for &cont in &continuation_lines {
            if cont {
                print!("    ");
            } else {
                print!("│   ");
            }
        }

        // 打印连接线
        if depth > 0 {
            if is_last {
                print!("└── ");
            } else {
                print!("├── ");
            }
        }

        // 打印路径
        let display_path = if full_path {
            node.path.display().to_string()
        } else {
            node.path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        };

        println!("{}", display_path.cyan());

        // 递归打印子节点
        continuation_lines.push(is_last);
        for (i, child) in node.children.iter().enumerate() {
            let is_last_child = i == node.children.len() - 1;
            self.print_node(
                child,
                depth + 1,
                full_path,
                is_last_child,
                continuation_lines.clone(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_add_dependency() {
        let mut tree = DependencyTree::new(10);
        let path1 = Path::new("/usr/lib/libc.so");
        let path2 = Path::new("/usr/lib/libm.so");

        tree.add_dependency(path1, 0);
        tree.add_dependency(path2, 0);

        assert_eq!(tree.visited.len(), 2);
        assert!(tree.visited.contains(path1));
        assert!(tree.visited.contains(path2));
    }

    #[test]
    fn test_max_depth() {
        let mut tree = DependencyTree::new(1);
        let path = Path::new("/usr/lib/libc.so");

        tree.add_dependency(path, 2); // 超过最大深度
        assert!(tree.visited.is_empty());
    }
}