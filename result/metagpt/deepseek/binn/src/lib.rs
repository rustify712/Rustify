//! Binn Rust 实现
//! 从C项目转换而来的二进制数据序列化库

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

mod types;
mod storage;
mod container;
mod value;
mod iter;

pub use types::*;
pub use storage::*;
pub use container::*;
pub use value::*;
pub use iter::*;

/// 版本信息
pub fn binn_version() -> &'static str {
    "3.0.0"
}