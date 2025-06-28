// lib.rs

use std::ffi::CString;
use std::mem;
use std::ptr;
use std::slice;

const VERSION: &str = "3.1.1";

const ET_EXEC: u16 = 2;
const ET_DYN: u16 = 3;

const PT_NULL: u32 = 0;
const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;

const DT_NULL: u32 = 0;
const DT_NEEDED: u32 = 1;
const DT_STRTAB: u32 = 5;
const DT_SONAME: u32 = 14;
const DT_RPATH: u32 = 15;
const DT_RUNPATH: u32 = 29;

const BITS32: u8 = 1;
const BITS64: u8 = 2;

const ERR_INVALID_MAGIC: u32 = 11;
const ERR_INVALID_CLASS: u32 = 12;
const ERR_INVALID_DATA: u32 = 13;
const ERR_INVALID_HEADER: u32 = 14;
const ERR_INVALID_BITS: u32 = 15;
const ERR_INVALID_ENDIANNESS: u32 = 16;
const ERR_NO_EXEC_OR_DYN: u32 = 17;
const ERR_INVALID_PHOFF: u32 = 18;
const ERR_INVALID_PROG_HEADER: u32 = 19;
const ERR_CANT_STAT: u32 = 20;
const ERR_INVALID_DYNAMIC_SECTION: u32 = 21;
const ERR_INVALID_DYNAMIC_ARRAY_ENTRY: u32 = 22;
const ERR_NO_STRTAB: u32 = 23;
const ERR_INVALID_SONAME: u32 = 24;
const ERR_INVALID_RPATH: u32 = 25;
const ERR_INVALID_RUNPATH: u32 = 26;
const ERR_INVALID_NEEDED: u32 = 27;
const ERR_DEPENDENCY_NOT_FOUND: u32 = 28;
const ERR_NO_PT_LOAD: u32 = 29;
const ERR_VADDRS_NOT_ORDERED: u32 = 30;
const ERR_COULD_NOT_OPEN_FILE: u32 = 31;
const ERR_INCOMPATIBLE_ISA: u32 = 32;

const DT_FLAGS_1: u32 = 0x6ffffffb;
const DT_1_NODEFLIB: u32 = 0x800;

const MAX_OFFSET_T: u64 = 0xFFFFFFFFFFFFFFFF;

const REGULAR_RED: &str = "\x1b[0;31m";
const BOLD_RED: &str = "\x1b[1;31m";
const CLEAR: &str = "\x1b[0m";
const BOLD_YELLOW: &str = "\x1b[33m";
const BOLD_CYAN: &str = "\x1b[1;36m";
const REGULAR_CYAN: &str = "\x1b[0;36m";
const REGULAR_MAGENTA: &str = "\x1b[0;35m";
const REGULAR_BLUE: &str = "\x1b[0;34m";
const BRIGHT_BLACK: &str = "\x1b[0;90m";
const REGULAR: &str = "\x1b[0m";

const LIGHT_HORIZONTAL: &str = "\xe2\x94\x80";
const LIGHT_QUADRUPLE_DASH_VERTICAL: &str = "\xe2\x94\x8a";
const LIGHT_UP_AND_RIGHT: &str = "\xe2\x94\x94";
const LIGHT_VERTICAL: &str = "\xe2\x94\x82";
const LIGHT_VERTICAL_AND_RIGHT: &str = "\xe2\x94\x9c";

const JUST_INDENT: &str = "    ";
const LIGHT_VERTICAL_WITH_INDENT: &str = "\xe2\x94\x82   ";

const SMALL_VEC_SIZE: usize = 16;
const MAX_RECURSION_DEPTH: usize = 32;
const MAX_PATH_LENGTH: usize = 4096;

const EXCLUDE_LIST: [&str; 15] = [
    "ld-linux-aarch64.so",
    "ld-linux-armhf.so",
    "ld-linux-x86-64.so",
    "ld-linux.so",
    "ld64.so",
    "libc.musl-aarch64.so",
    "libc.musl-armhf.so",
    "libc.musl-i386.so",
    "libc.musl-x86_64.so",
    "libc.so",
    "libdl.so",
    "libgcc_s.so",
    "libm.so",
    "libstdc++.so",
];

#[repr(C)]
struct Header64 {
    e_type: u16,
    e_machine: u16,
    e_version: u32,
    e_entry: u64,
    e_phoff: u64,
    e_shoff: u64,
    e_flags: u32,
    e_ehsize: u16,
    e_phentsize: u16,
    e_phnum: u16,
    e_shentsize: u16,
    e_shnum: u16,
    e_shstrndx: u16,
}

#[repr(C)]
struct Header32 {
    e_type: u16,
    e_machine: u16,
    e_version: u32,
    e_entry: u32,
    e_phoff: u32,
    e_shoff: u32,
    e_flags: u32,
    e_ehsize: u16,
    e_phentsize: u16,
    e_phnum: u16,
    e_shentsize: u16,
    e_shnum: u16,
    e_shstrndx: u16,
}

#[repr(C)]
struct Prog64 {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_paddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
}

#[repr(C)]
struct Prog32 {
    p_type: u32,
    p_offset: u32,
    p_vaddr: u32,
    p_paddr: u32,
    p_filesz: u32,
    p_memsz: u32,
    p_flags: u32,
    p_align: u32,
}

#[repr(C)]
struct Dyn64 {
    d_tag: i64,
    d_val: u64,
}

#[repr(C)]
struct Dyn32 {
    d_tag: i32,
    d_val: u32,
}

#[repr(C)]
struct Compat {
    any: bool, // 1 iff we don't look for libs matching a certain architecture
    class: u8, // 32 or 64 bits?
    machine: u16, // instruction set
}

#[derive(Debug, Clone, Copy)]
enum How {
    Input,
    Direct,
    Rpath,
    LdLibraryPath,
    Runpath,
    LdSoConf,
    Default,
}

#[repr(C)]
struct Found {
    how: How,
    depth: usize,
}

#[repr(C)]
struct StringTable {
    arr: *mut u8,
    n: usize,
    capacity: usize,
}

#[repr(C)]
struct VisitedFile {
    st_dev: u64,
    st_ino: u64,
}

#[repr(C)]
struct VisitedFileArray {
    arr: *mut VisitedFile,
    n: usize,
    capacity: usize,
}

#[repr(C)]
struct LibtreeState {
    verbosity: i32,
    path: i32,
    color: i32,
    ld_conf_file: *mut u8,
    max_depth: u64,
    string_table: StringTable,
    visited: VisitedFileArray,
    platform: *mut u8,
    lib: *mut u8,
    osname: *mut u8,
    osrel: *mut u8,
    rpath_offsets: [usize; MAX_RECURSION_DEPTH],
    ld_library_path_offset: usize,
    default_paths_offset: usize,
    ld_so_conf_offset: usize,
    found_all_needed: [bool; MAX_RECURSION_DEPTH],
}

#[repr(C)]
struct SmallVecU64 {
    buf: [u64; SMALL_VEC_SIZE],
    p: *mut u64,
    n: usize,
    capacity: usize,
}

impl SmallVecU64 {
    fn new() -> Self {
        let mut v = SmallVecU64 {
            buf: [0; SMALL_VEC_SIZE],
            p: ptr::null_mut(),
            n: 0,
            capacity: 0,
        };
        v.p = v.buf.as_mut_ptr();
        v
    }

    fn append(&mut self, val: u64) {
        if self.n < SMALL_VEC_SIZE {
            unsafe {
                *self.p.add(self.n) = val;
            }
            self.n += 1;
            return;
        }

        if self.n == SMALL_VEC_SIZE {
            self.capacity = 2 * SMALL_VEC_SIZE;
            self.p = unsafe { libc::malloc(self.capacity * mem::size_of::<u64>()) as *mut u64 };
            if self.p.is_null() {
                panic!("Memory allocation failed");
            }
            unsafe {
                ptr::copy_nonoverlapping(self.buf.as_ptr(), self.p, SMALL_VEC_SIZE);
            }
        } else if self.n == self.capacity {
            self.capacity *= 2;
            self.p = unsafe { libc::realloc(self.p as *mut libc::c_void, self.capacity * mem::size_of::<u64>()) as *mut u64 };
            if self.p.is_null() {
                panic!("Memory allocation failed");
            }
        }

        unsafe {
            *self.p.add(self.n) = val;
        }
        self.n += 1;
    }

    fn free(&mut self) {
        if self.n > SMALL_VEC_SIZE {
            unsafe {
                libc::free(self.p as *mut libc::c_void);
            }
        }
    }
}

fn utoa(str: &mut [u8], mut v: usize) {
    let mut p = 0;
    while v > 0 {
        str[p] = b'0' + (v % 10) as u8;
        v /= 10;
        p += 1;
    }
    str[..p].reverse();
    str[p] = 0;
}