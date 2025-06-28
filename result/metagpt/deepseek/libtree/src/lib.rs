//! ELF文件解析库

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use std::os::raw::c_char;
use std::ffi::CStr;
use std::path::Path;
use std::fs::File;
use std::io::Read;
use std::mem;

// 系统相关类型定义
#[cfg(target_os = "linux")]
mod sys {
    use std::os::raw::c_char;
    
    pub type dev_t = u64;
    pub type ino_t = u64;
    pub type mode_t = u32;
    pub type nlink_t = u64;
    pub type uid_t = u32;
    pub type gid_t = u32;
    pub type off_t = i64;
    pub type blksize_t = i64;
    pub type blkcnt_t = i64;
    pub type time_t = i64;

    #[repr(C)]
    pub struct Stat {
        pub st_dev: dev_t,
        pub st_ino: ino_t,
        pub st_mode: mode_t,
        pub st_nlink: nlink_t,
        pub st_uid: uid_t,
        pub st_gid: gid_t,
        pub st_rdev: dev_t,
        pub st_size: off_t,
        pub st_blksize: blksize_t,
        pub st_blocks: blkcnt_t,
        pub st_atime: time_t,
        pub st_mtime: time_t,
        pub st_ctime: time_t,
    }

    #[repr(C)]
    pub struct UtsName {
        pub sysname: [c_char; 65],
        pub nodename: [c_char; 65],
        pub release: [c_char; 65],
        pub version: [c_char; 65],
        pub machine: [c_char; 65],
        pub domainname: [c_char; 65],
    }
}

// ELF相关常量定义
pub const ET_EXEC: u16 = 2;
pub const ET_DYN: u16 = 3;
pub const PT_NULL: u32 = 0;
pub const PT_LOAD: u32 = 1;
pub const PT_DYNAMIC: u32 = 2;
pub const DT_NULL: u64 = 0;
pub const DT_NEEDED: u64 = 1;
pub const DT_STRTAB: u64 = 5;
pub const DT_SONAME: u64 = 14;
pub const DT_RPATH: u64 = 15;
pub const DT_RUNPATH: u64 = 29;
pub const DT_STRSZ: u64 = 10;
pub const BITS32: u8 = 1;
pub const BITS64: u8 = 2;

// ELF文件头结构体定义
#[repr(C)]
#[derive(Default)]
pub struct Header64 {
    pub e_type: u16,
    pub e_machine: u16,
    pub e_version: u32,
    pub e_entry: u64,
    pub e_phoff: u64,
    pub e_shoff: u64,
    pub e_flags: u32,
    pub e_ehsize: u16,
    pub e_phentsize: u16,
    pub e_phnum: u16,
    pub e_shentsize: u16,
    pub e_shnum: u16,
    pub e_shstrndx: u16,
}

#[repr(C)]
#[derive(Default)]
pub struct Header32 {
    pub e_type: u16,
    pub e_machine: u16,
    pub e_version: u32,
    pub e_entry: u32,
    pub e_phoff: u32,
    pub e_shoff: u32,
    pub e_flags: u32,
    pub e_ehsize: u16,
    pub e_phentsize: u16,
    pub e_phnum: u16,
    pub e_shentsize: u16,
    pub e_shnum: u16,
    pub e_shstrndx: u16,
}

// 程序头结构体定义
#[repr(C)]
#[derive(Default)]
pub struct Prog64 {
    pub p_type: u32,
    pub p_flags: u32,
    pub p_offset: u64,
    pub p_vaddr: u64,
    pub p_paddr: u64,
    pub p_filesz: u64,
    pub p_memsz: u64,
    pub p_align: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct Prog32 {
    pub p_type: u32,
    pub p_offset: u32,
    pub p_vaddr: u32,
    pub p_paddr: u32,
    pub p_filesz: u32,
    pub p_memsz: u32,
    pub p_flags: u32,
    pub p_align: u32,
}

// 动态段结构体定义
#[repr(C)]
#[derive(Default)]
pub struct Dyn64 {
    pub d_tag: i64,
    pub d_val: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct Dyn32 {
    pub d_tag: i32,
    pub d_val: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_struct_sizes() {
        assert_eq!(mem::size_of::<Header64>(), 64);
        assert_eq!(mem::size_of::<Header32>(), 52);
        assert_eq!(mem::size_of::<Prog64>(), 56);
        assert_eq!(mem::size_of::<Prog32>(), 32);
    }

    #[test]
    fn test_struct_alignment() {
        assert_eq!(mem::align_of::<Header64>(), 8);
        assert_eq!(mem::align_of::<Header32>(), 4);
        assert_eq!(mem::align_of::<Prog64>(), 8);
        assert_eq!(mem::align_of::<Prog32>(), 4);
    }
}