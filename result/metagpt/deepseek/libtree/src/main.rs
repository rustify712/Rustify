//! libtree 主程序 - ELF依赖分析工具

use std::path::PathBuf;
use std::process;
use structopt::StructOpt;
use colored::Colorize;

mod elf;
mod path_utils;
mod tree;

use crate::elf::ElfFile;
use crate::path_utils::{is_absolute_path, find_library_in_paths};

/// 命令行参数定义
#[derive(Debug, StructOpt)]
#[structopt(
    name = "libtree",
    about = "Display ELF dependency tree",
    version = "3.1.1"
)]
struct Opt {
    /// 输入文件路径
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    /// 显示完整路径
    #[structopt(short, long)]
    full_path: bool,

    /// 禁用颜色输出
    #[structopt(long)]
    no_color: bool,

    /// 显示所有依赖，包括系统库
    #[structopt(short, long)]
    all: bool,

    /// 最大递归深度
    #[structopt(short, long, default_value = "32")]
    depth: usize,
}

fn main() {
    let opt = Opt::from_args();

    // 初始化颜色配置
    if opt.no_color {
        colored::control::set_override(false);
    }

    // 解析ELF文件
    let mut elf = match ElfFile::open(&opt.input) {
        Ok(elf) => elf,
        Err(e) => {
            eprintln!("{}: {}", "Error".red(), e);
            process::exit(1);
        }
    };

    // 获取依赖库列表
    let needed_libs = match elf.get_needed_libs() {
        Ok(libs) => libs,
        Err(e) => {
            eprintln!("{}: {}", "Error".red(), e);
            process::exit(1);
        }
    };

    // 构建依赖树
    let mut tree = tree::DependencyTree::new(opt.depth);
    for lib in needed_libs {
        if !opt.all && is_system_lib(&lib) {
            continue;
        }

        let path = if is_absolute_path(&lib) {
            PathBuf::from(&lib)
        } else {
            match find_library_in_paths(&lib, &get_search_paths()) {
                Ok(path) => path,
                Err(_) => {
                    eprintln!("{}: Library not found: {}", "Warning".yellow(), lib);
                    continue;
                }
            }
        };

        tree.add_dependency(&path, 0);
    }

    // 打印依赖树
    tree.print(opt.full_path);
}

/// 判断是否为系统库
fn is_system_lib(lib: &str) -> bool {
    let system_libs = [
        "ld-linux", "libc.", "libdl.", "libm.", "libgcc_s.", "libstdc++."
    ];
    
    system_libs.iter().any(|&prefix| lib.starts_with(prefix))
}

/// 获取系统库搜索路径
fn get_search_paths() -> Vec<String> {
    let mut paths = Vec::new();
    
    // 添加默认系统路径
    paths.push("/usr/lib".to_string());
    paths.push("/usr/local/lib".to_string());
    
    // 添加LD_LIBRARY_PATH
    if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
        paths.extend(ld_path.split(':').map(|s| s.to_string()));
    }
    
    paths
}