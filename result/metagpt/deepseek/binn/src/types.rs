//! 类型定义模块

/// 布尔类型
pub type BOOL = bool;

/// 64位有符号整数
type int64 = i64;
/// 64位无符号整数
type uint64 = u64;

/// 存储类型常量
pub const BINN_STORAGE_NOBYTES: u32 = 0x00;
pub const BINN_STORAGE_BYTE: u32 = 0x20;
pub const BINN_STORAGE_WORD: u32 = 0x40;
pub const BINN_STORAGE_DWORD: u32 = 0x60;
pub const BINN_STORAGE_QWORD: u32 = 0x80;
pub const BINN_STORAGE_STRING: u32 = 0xA0;
pub const BINN_STORAGE_BLOB: u32 = 0xC0;
pub const BINN_STORAGE_CONTAINER: u32 = 0xE0;

/// 数据格式常量
pub const BINN_LIST: u32 = 0xE0;
pub const BINN_MAP: u32 = 0xE1;
pub const BINN_OBJECT: u32 = 0xE2;

pub const BINN_NULL: u32 = 0x00;
pub const BINN_TRUE: u32 = 0x01;
pub const BINN_FALSE: u32 = 0x02;

pub const BINN_UINT8: u32 = 0x20;
pub const BINN_INT8: u32 = 0x21;
pub const BINN_UINT16: u32 = 0x40;
pub const BINN_INT16: u32 = 0x41;
pub const BINN_UINT32: u32 = 0x60;
pub const BINN_INT32: u32 = 0x61;
pub const BINN_UINT64: u32 = 0x80;
pub const BINN_INT64: u32 = 0x81;

pub const BINN_STRING: u32 = 0xA0;
pub const BINN_BLOB: u32 = 0xC0;