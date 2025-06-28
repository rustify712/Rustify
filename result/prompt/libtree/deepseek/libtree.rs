use std::collections::HashMap;
use std::ffi::CString;
use std::fs::File;
use std::io::{self, Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;
use std::str;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use libc::{c_char, c_int, c_void, dev_t, ino_t, size_t, stat, S_IFMT, S_IFREG};
use md5;
use rand;
use regex;

const VERSION: &str = "3.1.1";

const ET_EXEC: u16 = 2;
const ET_DYN: u16 = 3;

const PT_NULL: u32 = 0;
const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;

const DT_NULL: i64 = 0;
const DT_NEEDED: i64 = 1;
const DT_STRTAB: i64 = 5;
const DT_SONAME: i64 = 14;
const DT_RPATH: i64 = 15;
const DT_RUNPATH: i64 = 29;

const BITS32: u8 = 1;
const BITS64: u8 = 2;

const ERR_INVALID_MAGIC: i32 = 11;
const ERR_INVALID_CLASS: i32 = 12;
const ERR_INVALID_DATA: i32 = 13;
const ERR_INVALID_HEADER: i32 = 14;
const ERR_INVALID_BITS: i32 = 15;
const ERR_INVALID_ENDIANNESS: i32 = 16;
const ERR_NO_EXEC_OR_DYN: i32 = 17;
const ERR_INVALID_PHOFF: i32 = 18;
const ERR_INVALID_PROG_HEADER: i32 = 19;
const ERR_CANT_STAT: i32 = 20;
const ERR_INVALID_DYNAMIC_SECTION: i32 = 21;
const ERR_INVALID_DYNAMIC_ARRAY_ENTRY: i32 = 22;
const ERR_NO_STRTAB: i32 = 23;
const ERR_INVALID_SONAME: i32 = 24;
const ERR_INVALID_RPATH: i32 = 25;
const ERR_INVALID_RUNPATH: i32 = 26;
const ERR_INVALID_NEEDED: i32 = 27;
const ERR_DEPENDENCY_NOT_FOUND: i32 = 28;
const ERR_NO_PT_LOAD: i32 = 29;
const ERR_VADDRS_NOT_ORDERED: i32 = 30;
const ERR_COULD_NOT_OPEN_FILE: i32 = 31;
const ERR_INCOMPATIBLE_ISA: i32 = 32;

const DT_FLAGS_1: i64 = 0x6ffffffb;
const DT_1_NODEFLIB: i64 = 0x800;

const MAX_OFFSET_T: u64 = 0xFFFFFFFFFFFFFFFF;

const REGULAR_RED: &str = "\033[0;31m";
const BOLD_RED: &str = "\033[1;31m";
const CLEAR: &str = "\033[0m";
const BOLD_YELLOW: &str = "\033[33m";
const BOLD_CYAN: &str = "\033[1;36m";
const REGULAR_CYAN: &str = "\033[0;36m";
const REGULAR_MAGENTA: &str = "\033[0;35m";
const REGULAR_BLUE: &str = "\033[0;34m";
const BRIGHT_BLACK: &str = "\033[0;90m";
const REGULAR: &str = "\033[0m";

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

static EXCLUDE_LIST: [&str; 14] = [
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

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
struct Dyn64 {
    d_tag: i64,
    d_val: u64,
}

#[derive(Debug)]
struct Dyn32 {
    d_tag: i32,
    d_val: u32,
}

#[derive(Debug)]
struct Compat {
    any: bool,
    class: u8,
    machine: u16,
}

#[derive(Debug)]
enum How {
    Input,
    Direct,
    Rpath,
    LdLibraryPath,
    Runpath,
    LdSoConf,
    Default,
}

#[derive(Debug)]
struct Found {
    how: How,
    depth: usize,
}

#[derive(Debug)]
struct StringTable {
    arr: Vec<u8>,
    n: usize,
    capacity: usize,
}

#[derive(Debug)]
struct VisitedFile {
    st_dev: dev_t,
    st_ino: ino_t,
}

#[derive(Debug)]
struct VisitedFileArray {
    arr: Vec<VisitedFile>,
    n: usize,
    capacity: usize,
}

#[derive(Debug)]
struct LibtreeState {
    verbosity: i32,
    path: bool,
    color: bool,
    ld_conf_file: String,
    max_depth: usize,

    string_table: StringTable,
    visited: VisitedFileArray,

    platform: String,
    lib: String,
    osname: String,
    osrel: String,

    rpath_offsets: [usize; MAX_RECURSION_DEPTH],
    ld_library_path_offset: usize,
    default_paths_offset: usize,
    ld_so_conf_offset: usize,

    found_all_needed: [bool; MAX_RECURSION_DEPTH],
}

#[derive(Debug)]
struct SmallVecU64 {
    buf: [u64; SMALL_VEC_SIZE],
    p: Vec<u64>,
    n: usize,
    capacity: usize,
}

impl SmallVecU64 {
    fn new() -> Self {
        SmallVecU64 {
            buf: [0; SMALL_VEC_SIZE],
            p: Vec::new(),
            n: 0,
            capacity: SMALL_VEC_SIZE,
        }
    }

    fn append(&mut self, val: u64) {
        if self.n < SMALL_VEC_SIZE {
            self.buf[self.n] = val;
            self.n += 1;
            return;
        }

        if self.n == SMALL_VEC_SIZE {
            self.capacity = 2 * SMALL_VEC_SIZE;
            self.p = self.buf.to_vec();
        } else if self.n == self.capacity {
            self.capacity *= 2;
            self.p.resize(self.capacity, 0);
        }

        self.p[self.n] = val;
        self.n += 1;
    }

    fn free(&mut self) {
        if self.n > SMALL_VEC_SIZE {
            self.p.clear();
        }
    }
}

fn host_is_little_endian() -> bool {
    let test: i32 = 1;
    let bytes = test.to_ne_bytes();
    bytes[0] == 1
}

fn is_ascending_order(v: &[u64], n: usize) -> bool {
    for j in 1..n {
        if v[j - 1] >= v[j] {
            return false;
        }
    }
    true
}

fn string_table_maybe_grow(t: &mut StringTable, n: usize) {
    if t.n + n <= t.capacity {
        return;
    }

    t.capacity = 2 * (t.n + n);
    t.arr.resize(t.capacity, 0);
}

fn string_table_store(t: &mut StringTable, str: &str) {
    let n = str.len() + 1;
    string_table_maybe_grow(t, n);
    t.arr[t.n..t.n + n].copy_from_slice(str.as_bytes());
    t.n += n;
}

fn string_table_copy_from_file(t: &mut StringTable, fptr: &mut File) -> io::Result<()> {
    let mut buf = [0; 1];
    while fptr.read(&mut buf)? > 0 {
        if buf[0] == b'\0' {
            break;
        }
        string_table_maybe_grow(t, 1);
        t.arr[t.n] = buf[0];
        t.n += 1;
    }
    string_table_maybe_grow(t, 1);
    t.arr[t.n] = b'\0';
    t.n += 1;
    Ok(())
}

fn is_in_exclude_list(soname: &str) -> bool {
    let start = soname;
    let end = start.rfind('\0').unwrap_or(start.len());

    if start == end {
        return false;
    }

    for excluded in EXCLUDE_LIST.iter() {
        if start.starts_with(excluded) {
            return true;
        }
    }
    false
}

fn tree_preamble(s: &LibtreeState, depth: usize) {
    if depth == 0 {
        return;
    }

    for i in 0..depth - 1 {
        if s.found_all_needed[i] {
            print!("{}", JUST_INDENT);
        } else {
            print!("{}", LIGHT_VERTICAL_WITH_INDENT);
        }
    }

    if s.found_all_needed[depth - 1] {
        print!("{}", LIGHT_UP_AND_RIGHT);
    } else {
        print!("{}", LIGHT_VERTICAL_AND_RIGHT);
    }
}

fn recurse(
    current_file: &str,
    depth: usize,
    state: &mut LibtreeState,
    compat: Compat,
    reason: Found,
) -> i32 {
    let mut fptr = match File::open(current_file) {
        Ok(file) => file,
        Err(_) => return ERR_COULD_NOT_OPEN_FILE,
    };

    let old_buf_size = state.string_table.n;

    let mut e_ident = [0; 16];
    if fptr.read_exact(&mut e_ident).is_err() {
        return ERR_INVALID_MAGIC;
    }

    if e_ident[0] != 0x7f || e_ident[1] != b'E' || e_ident[2] != b'L' || e_ident[3] != b'F' {
        return ERR_INVALID_MAGIC;
    }

    if e_ident[4] != BITS32 && e_ident[4] != BITS64 {
        return ERR_INVALID_CLASS;
    }

    if e_ident[5] != 1 && e_ident[5] != 2 {
        return ERR_INVALID_DATA;
    }

    let curr_type = Compat {
        any: false,
        class: e_ident[4],
    };
    let is_little_endian = e_ident[5] == 1;

    if !compat.any && compat.class != curr_type.class {
        return ERR_INVALID_BITS;
    }

    if is_little_endian != host_is_little_endian() {
        return ERR_INVALID_ENDIANNESS;
    }

    let mut header = if curr_type.class == BITS64 {
        let mut hdr = Header64 {
            e_type: 0,
            e_machine: 0,
            e_version: 0,
            e_entry: 0,
            e_phoff: 0,
            e_shoff: 0,
            e_flags: 0,
            e_ehsize: 0,
            e_phentsize: 0,
            e_phnum: 0,
            e_shentsize: 0,
            e_shnum: 0,
            e_shstrndx: 0,
        };
        if fptr.read_exact(unsafe {
            slice::from_raw_parts_mut(&mut hdr as *mut _ as *mut u8, std::mem::size_of::<Header64>())
        }).is_err() {
            return ERR_INVALID_HEADER;
        }
        hdr
    } else {
        let mut hdr = Header32 {
            e_type: 0,
            e_machine: 0,
            e_version: 0,
            e_entry: 0,
            e_phoff: 0,
            e_shoff: 0,
            e_flags: 0,
            e_ehsize: 0,
            e_phentsize: 0,
            e_phnum: 0,
            e_shentsize: 0,
            e_shnum: 0,
            e_shstrndx: 0,
        };
        if fptr.read_exact(unsafe {
            slice::from_raw_parts_mut(&mut hdr as *mut _ as *mut u8, std::mem::size_of::<Header32>())
        }).is_err() {
            return ERR_INVALID_HEADER;
        }
        hdr
    };

    if (curr_type.class == BITS64 && (header.e_type != ET_EXEC && header.e_type != ET_DYN))
        || (curr_type.class == BITS32 && (header.e_type != ET_EXEC && header.e_type != ET_DYN))
    {
        return ERR_NO_EXEC_OR_DYN;
    }

    let mut pt_load_offset = SmallVecU64::new();
    let mut pt_load_vaddr = SmallVecU64::new();

    let mut p_offset = MAX_OFFSET_T;

    if curr_type.class == BITS64 {
        for _ in 0..header.e_phnum {
            let mut prog = Prog64 {
                p_type: 0,
                p_flags: 0,
                p_offset: 0,
                p_vaddr: 0,
                p_paddr: 0,
                p_filesz: 0,
                p_memsz: 0,
                p_align: 0,
            };
            if fptr.read_exact(unsafe {
                slice::from_raw_parts_mut(&mut prog as *mut _ as *mut u8, std::mem::size_of::<Prog64>())
            }).is_err() {
                return ERR_INVALID_PROG_HEADER;
            }

            if prog.p_type == PT_LOAD {
                pt_load_offset.append(prog.p_offset);
                pt_load_vaddr.append(prog.p_vaddr);
            } else if prog.p_type == PT_DYNAMIC {
                p_offset = prog.p_offset;
            }
        }
    } else {
        for _ in 0..header.e_phnum {
            let mut prog = Prog32 {
                p_type: 0,
                p_offset: 0,
                p_vaddr: 0,
                p_paddr: 0,
                p_filesz: 0,
                p_memsz: 0,
                p_flags: 0,
                p_align: 0,
            };
            if fptr.read_exact(unsafe {
                slice::from_raw_parts_mut(&mut prog as *mut _ as *mut u8, std::mem::size_of::<Prog32>())
            }).is_err() {
                return ERR_INVALID_PROG_HEADER;
            }

            if prog.p_type == PT_LOAD {
                pt_load_offset.append(prog.p_offset);
                pt_load_vaddr.append(prog.p_vaddr);
            } else if prog.p_type == PT_DYNAMIC {
                p_offset = prog.p_offset;
            }
        }
    }

    let mut finfo = stat {
        st_dev: 0,
        st_ino: 0,
        st_mode: 0,
        st_nlink: 0,
        st_uid: 0,
        st_gid: 0,
        st_rdev: 0,
        st_size: 0,
        st_blksize: 0,
        st_blocks: 0,
        st_atime: 0,
        st_atime_nsec: 0,
        st_mtime: 0,
        st_mtime_nsec: 0,
        st_ctime: 0,
        st_ctime_nsec: 0,
    };

    if unsafe { libc::stat(current_file.as_ptr() as *const c_char, &mut finfo) } != 0 {
        return ERR_CANT_STAT;
    }

    let seen_before = state.visited.arr.iter().any(|f| f.st_dev == finfo.st_dev && f.st_ino == finfo.st_ino);

    if !seen_before {
        state.visited.arr.push(VisitedFile {
            st_dev: finfo.st_dev,
            st_ino: finfo.st_ino,
        });
    }

    if p_offset == MAX_OFFSET_T {
        print_line(depth, current_file, BOLD_CYAN, REGULAR_CYAN, true, reason, state);
        return 0;
    }

    if pt_load_offset.n == 0 {
        return ERR_NO_PT_LOAD;
    }

    if fptr.seek(io::SeekFrom::Start(p_offset)).is_err() {
        return ERR_INVALID_DYNAMIC_SECTION;
    }

    let mut no_def_lib = false;

    let mut strtab = MAX_OFFSET_T;
    let mut rpath = MAX_OFFSET_T;
    let mut runpath = MAX_OFFSET_T;
    let mut soname = MAX_OFFSET_T;

    let mut needed = SmallVecU64::new();

    loop {
        let d_tag;
        let d_val;

        if curr_type.class == BITS64 {
            let mut dyn = Dyn64 { d_tag: 0, d_val: 0 };
            if fptr.read_exact(unsafe {
                slice::from_raw_parts_mut(&mut dyn as *mut _ as *mut u8, std::mem::size_of::<Dyn64>())
            }).is_err() {
                return ERR_INVALID_DYNAMIC_ARRAY_ENTRY;
            }
            d_tag = dyn.d_tag;
            d_val = dyn.d_val;
        } else {
            let mut dyn = Dyn32 { d_tag: 0, d_val: 0 };
            if fptr.read_exact(unsafe {
                slice::from_raw_parts_mut(&mut dyn as *mut _ as *mut u8, std::mem::size_of::<Dyn32>())
            }).is_err() {
                return ERR_INVALID_DYNAMIC_ARRAY_ENTRY;
            }
            d_tag = dyn.d_tag as i64;
            d_val = dyn.d_val as u64;
        }

        match d_tag {
            DT_NULL => break,
            DT_STRTAB => strtab = d_val,
            DT_RPATH => rpath = d_val,
            DT_RUNPATH => runpath = d_val,
            DT_NEEDED => needed.append(d_val),
            DT_SONAME => soname = d_val,
            DT_FLAGS_1 => no_def_lib |= (DT_1_NODEFLIB & d_val) == DT_1_NODEFLIB,
            _ => (),
        }
    }

    if strtab == MAX_OFFSET_T {
        return ERR_NO_STRTAB;
    }

    if !is_ascending_order(&pt_load_vaddr.p, pt_load_vaddr.n) {
        return ERR_VADDRS_NOT_ORDERED;
    }

    let mut vaddr_idx = 0;
    while vaddr_idx + 1 != pt_load_vaddr.n && strtab >= pt_load_vaddr.p[vaddr_idx + 1] {
        vaddr_idx += 1;
    }

    let strtab_offset = pt_load_offset.p[vaddr_idx] + strtab - pt_load_vaddr.p[vaddr_idx];

    pt_load_vaddr.free();
    pt_load_offset.free();

    let soname_buf_offset = state.string_table.n;
    if soname != MAX_OFFSET_T {
        if fptr.seek(io::SeekFrom::Start(strtab_offset + soname)).is_err() {
            state.string_table.n = old_buf_size;
            return ERR_INVALID_SONAME;
        }
        if string_table_copy_from_file(&mut state.string_table, &mut fptr).is_err() {
            return ERR_INVALID_SONAME;
        }
    }

    let in_exclude_list = soname != MAX_OFFSET_T && is_in_exclude_list(unsafe {
        str::from_utf8_unchecked(&state.string_table.arr[soname_buf_offset..])
    });

    let should_recurse = depth < state.max_depth
        && ((!seen_before && !in_exclude_list)
            || (!seen_before && in_exclude_list && state.verbosity >= 2)
            || state.verbosity >= 3);

    if !should_recurse {
        let print_name = if soname == MAX_OFFSET_T || state.path {
            current_file
        } else {
            unsafe { str::from_utf8_unchecked(&state.string_table.arr[soname_buf_offset..]) }
        };

        let bold_color = if in_exclude_list {
            REGULAR_MAGENTA
        } else if seen_before {
            REGULAR_BLUE
        } else {
            BOLD_CYAN
        };

        let regular_color = if in_exclude_list {
            REGULAR_MAGENTA
        } else if seen_before {
            REGULAR_BLUE
        } else {
            REGULAR_CYAN
        };

        let highlight = !seen_before && !in_exclude_list;
        print_line(depth, print_name, bold_color, regular_color, highlight, reason, state);

        state.string_table.n = old_buf_size;
        return 0;
    }

    let mut origin = [0; MAX_PATH_LENGTH];
    let last_slash = current_file.rfind('/').unwrap_or(0);
    if last_slash != 0 {
        origin[..last_slash].copy_from_slice(&current_file.as_bytes()[..last_slash]);
        origin[last_slash] = b'\0';
    } else {
        origin[..2].copy_from_slice(b"./");
        origin[2] = b'\0';
    }

    if rpath == MAX_OFFSET_T {
        state.rpath_offsets[depth] = usize::MAX;
    } else {
        state.rpath_offsets[depth] = state.string_table.n;
        if fptr.seek(io::SeekFrom::Start(strtab_offset + rpath)).is_err() {
            state.string_table.n = old_buf_size;
            return ERR_INVALID_RPATH;
        }

        if string_table_copy_from_file(&mut state.string_table, &mut fptr).is_err() {
            return ERR_INVALID_RPATH;
        }

        let curr_buf_size = state.string_table.n;
        if interpolate_variables(state, state.rpath_offsets[depth], unsafe {
            str::from_utf8_unchecked(&origin)
        }) {
            state.rpath_offsets[depth] = curr_buf_size;
        }
    }

    let runpath_buf_offset = state.string_table.n;
    if runpath != MAX_OFFSET_T {
        if fptr.seek(io::SeekFrom::Start(strtab_offset + runpath)).is_err() {
            state.string_table.n = old_buf_size;
            return ERR_INVALID_RUNPATH;
        }

        if string_table_copy_from_file(&mut state.string_table, &mut fptr).is_err() {
            return ERR_INVALID_RUNPATH;
        }

        let curr_buf_size = state.string_table.n;
        if interpolate_variables(state, runpath_buf_offset, unsafe {
            str::from_utf8_unchecked(&origin)
        }) {
            runpath_buf_offset = curr_buf_size;
        }
    }

    let mut needed_buf_offsets = SmallVecU64::new();

    for i in 0..needed.n {
        needed_buf_offsets.append(state.string_table.n);
        if fptr.seek(io::SeekFrom::Start(strtab_offset + needed.p[i])).is_err() {
            state.string_table.n = old_buf_size;
            return ERR_INVALID_NEEDED;
        }
        if string_table_copy_from_file(&mut state.string_table, &mut fptr).is_err() {
            return ERR_INVALID_NEEDED;
        }
    }

    let print_name = if soname == MAX_OFFSET_T || state.path {
        current_file
    } else {
        unsafe { str::from_utf8_unchecked(&state.string_table.arr[soname_buf_offset..]) }
    };

    let bold_color = if in_exclude_list {
        REGULAR_MAGENTA
    } else if seen_before {
        REGULAR_BLUE
    } else {
        BOLD_CYAN
    };

    let regular_color = if in_exclude_list {
        REGULAR_MAGENTA
    } else if seen_before {
        REGULAR_BLUE
    } else {
        REGULAR_CYAN
    };

    let highlight = !seen_before && !in_exclude_list;
    print_line(depth, print_name, bold_color, regular_color, highlight, reason, state);

    let mut exit_code = 0;

    let mut needed_not_found = needed_buf_offsets.n;

    if needed_not_found > 0 && state.verbosity == 0 {
        apply_exclude_list(&mut needed_not_found, &mut needed_buf_offsets, state);
    }

    if needed_not_found > 0 {
        exit_code |= check_absolute_paths(
            &mut needed_not_found,
            &mut needed_buf_offsets,
            depth,
            state,
            curr_type,
        );
    }

    if runpath == MAX_OFFSET_T {
        for j in (0..=depth).rev() {
            if state.rpath_offsets[j] == usize::MAX {
                continue;
            }

            exit_code |= check_search_paths(
                Found {
                    how: How::Rpath,
                    depth: j,
                },
                state.rpath_offsets[j],
                &mut needed_not_found,
                &mut needed_buf_offsets,
                depth,
                state,
                curr_type,
            );
        }
    }

    if needed_not_found > 0 && state.ld_library_path_offset != usize::MAX {
        exit_code |= check_search_paths(
            Found {
                how: How::LdLibraryPath,
            },
            state.ld_library_path_offset,
            &mut needed_not_found,
            &mut needed_buf_offsets,
            depth,
            state,
            curr_type,
        );
    }

    if needed_not_found > 0 && runpath != MAX_OFFSET_T {
        exit_code |= check_search_paths(
            Found {
                how: How::Runpath,
            },
            runpath_buf_offset,
            &mut needed_not_found,
            &mut needed_buf_offsets,
            depth,
            state,
            curr_type,
        );
    }

    if needed_not_found > 0 && !no_def_lib {
        exit_code |= check_search_paths(
            Found {
                how: How::LdSoConf,
            },
            state.ld_so_conf_offset,
            &mut needed_not_found,
            &mut needed_buf_offsets,
            depth,
            state,
            curr_type,
        );
    }

    if needed_not_found > 0 && !no_def_lib {
        exit_code |= check_search_paths(
            Found {
                how: How::Default,
            },
            state.default_paths_offset,
            &mut needed_not_found,
            &mut needed_buf_offsets,
            depth,
            state,
            curr_type,
        );
    }

    if needed_not_found > 0 {
        print_error(
            depth,
            needed_not_found,
            &mut needed_buf_offsets,
            if runpath == MAX_OFFSET_T {
                None
            } else {
                Some(unsafe {
                    str::from_utf8_unchecked(&state.string_table.arr[runpath_buf_offset..])
                })
            },
            state,
            no_def_lib,
        );
        state.string_table.n = old_buf_size;
        return ERR_DEPENDENCY_NOT_FOUND;
    }

    state.string_table.n = old_buf_size;
    needed_buf_offsets.free();
    needed.free();
    exit_code
}

fn parse_ld_config_file(st: &mut StringTable, path: &str) -> i32 {
    let mut fptr = match File::open(path) {
        Ok(file) => file,
        Err(_) => return 1,
    };

    let mut c = 0;
    let mut line = [0; MAX_PATH_LENGTH];
    let mut tmp = [0; MAX_PATH_LENGTH];

    while c != b'\0' {
        let mut line_len = 0;
        while c != b'\n' && c != b'\0' {
            if line_len < MAX_PATH_LENGTH - 1 {
                line[line_len] = c;
                line_len += 1;
            }
            c = fptr.read(&mut [0; 1]).unwrap_or(0)[0];
        }

        line[line_len] = b'\0';

        let mut begin = 0;
        let mut end = line_len;

        while begin < end && line[begin].is_ascii_whitespace() {
            begin += 1;
        }

        let comment = line[begin..end].iter().position(|&x| x == b'#');
        if let Some(comment) = comment {
            end = begin + comment;
        }

        while end > begin && line[end - 1].is_ascii_whitespace() {
            end -= 1;
        }

        if begin == end {
            continue;
        }

        line[end] = b'\0';

        if line[begin..].starts_with(b"include") && line[begin + 7].is_ascii_whitespace() {
            begin += 8;
            while begin < end && line[begin].is_ascii_whitespace() {
                begin += 1;
            }

            if line[begin] != b'/' {
                let wd = path.rfind('/').unwrap_or(0);
                let wd_len = wd;
                let include_len = end - begin;

                if wd_len + 1 + include_len >= MAX_PATH_LENGTH {
                    continue;
                }

                tmp[..wd_len].copy_from_slice(&path.as_bytes()[..wd_len]);
                tmp[wd_len] = b'/';
                tmp[wd_len + 1..wd_len + 1 + include_len]
                    .copy_from_slice(&line[begin..begin + include_len]);
                tmp[wd_len + 1 + include_len] = b'\0';
                begin = 0;
            }

            ld_conf_globbing(st, unsafe { str::from_utf8_unchecked(&tmp[begin..]) });
        } else {
            string_table_store(st, unsafe { str::from_utf8_unchecked(&line[begin..end]) });
            st.arr[st.n - 1] = b':';
        }
    }

    0
}

fn ld_conf_globbing(st: &mut StringTable, pattern: &str) -> i32 {
    let mut result = Vec::new();
    let glob_pattern = pattern.to_string() + "/*";
    for entry in glob::glob(&glob_pattern).unwrap_or_else(|_| glob::glob("").unwrap()) {
        if let Ok(path) = entry {
            result.push(path);
        }
    }

    let mut code = 0;
    for path in result {
        code |= parse_ld_config_file(st, path.to_str().unwrap());
    }
    code
}

fn parse_ld_so_conf(s: &mut LibtreeState) {
    s.ld_so_conf_offset = s.string_table.n;

    parse_ld_config_file(&mut s.string_table, &s.ld_conf_file);

    if s.string_table.n > s.ld_so_conf_offset {
        s.string_table.arr[s.string_table.n - 1] = b'\0';
    } else {
        string_table_store(&mut s.string_table, "");
    }
}

fn parse_ld_library_path(s: &mut LibtreeState) {
    s.ld_library_path_offset = usize::MAX;
    let val = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();

    if val.is_empty() {
        return;
    }

    s.ld_library_path_offset = s.string_table.n;

    string_table_store(&mut s.string_table, &val);

    let search = s.string_table.arr[s.ld_library_path_offset..]
        .iter_mut()
        .position(|&mut x| x == b';');
    if let Some(search) = search {
        s.string_table.arr[s.ld_library_path_offset + search] = b':';
    }
}

fn set_default_paths(s: &mut LibtreeState) {
    s.default_paths_offset = s.string_table.n;
    string_table_store(&mut s.string_table, "/lib:/lib64:/usr/lib:/usr/lib64");
}

fn libtree_state_init(s: &mut LibtreeState) {
    s.string_table.n = 0;
    s.string_table.capacity = 1024;
    s.string_table.arr = vec![0; s.string_table.capacity];
    s.visited.n = 0;
    s.visited.capacity = 256;
    s.visited.arr = Vec::with_capacity(s.visited.capacity);
}

fn libtree_state_free(s: &mut LibtreeState) {
    s.string_table.arr.clear();
    s.visited.arr.clear();
}

fn print_tree(pathc: i32, pathv: &[String], s: &mut LibtreeState) -> i32 {
    libtree_state_init(s);

    parse_ld_so_conf(s);
    parse_ld_library_path(s);
    set_default_paths(s);

    let mut exit_code = 0;

    for i in 0..pathc {
        let code = recurse(
            &pathv[i as usize],
            0,
            s,
            Compat {
                any: true,
                class: 0,
                machine: 0,
            },
            Found {
                how: How::Input,
                depth: 0,
            },
        );
        io::stdout().flush().unwrap();
        if code != 0 {
            exit_code = code;
            eprint!("Error [{}]: ", pathv[i as usize]);
        }
        let msg = match code {
            ERR_INVALID_MAGIC => "Invalid ELF magic bytes\n",
            ERR_INVALID_CLASS => "Invalid ELF class\n",
            ERR_INVALID_DATA => "Invalid ELF data\n",
            ERR_INVALID_HEADER => "Invalid ELF header\n",
            ERR_INVALID_BITS => "Invalid bits\n",
            ERR_INVALID_ENDIANNESS => "Invalid endianness\n",
            ERR_NO_EXEC_OR_DYN => "Not an ET_EXEC or ET_DYN ELF file\n",
            ERR_INVALID_PHOFF => "Invalid ELF program header offset\n",
            ERR_INVALID_PROG_HEADER => "Invalid ELF program header\n",
            ERR_CANT_STAT => "Can't stat file\n",
            ERR_INVALID_DYNAMIC_SECTION => "Invalid ELF dynamic section\n",
            ERR_INVALID_DYNAMIC_ARRAY_ENTRY => "Invalid ELF dynamic array entry\n",
            ERR_NO_STRTAB => "No ELF string table found\n",
            ERR_INVALID_SONAME => "Can't read DT_SONAME\n",
            ERR_INVALID_RPATH => "Can't read DT_RPATH\n",
            ERR_INVALID_RUNPATH => "Can't read DT_RUNPATH\n",
            ERR_INVALID_NEEDED => "Can't read DT_NEEDED\n",
            ERR_DEPENDENCY_NOT_FOUND => "Not all dependencies were found\n",
            ERR_NO_PT_LOAD => "No PT_LOAD found in ELF file\n",
            ERR_VADDRS_NOT_ORDERED => "Virtual addresses are not ordered\n",
            ERR_COULD_NOT_OPEN_FILE => "Could not open file\n",
            ERR_INCOMPATIBLE_ISA => "Incompatible ISA\n",
            _ => "",
        };
        eprint!("{}", msg);
        io::stderr().flush().unwrap();
    }

    libtree_state_free(s);
    exit_code
}

fn main() {
    let mut s = LibtreeState {
        verbosity: 0,
        path: false,
        color: std::env::var("NO_COLOR").is_err() && unsafe { libc::isatty(libc::STDOUT_FILENO) } != 0,
        ld_conf_file: String::new(),
        max_depth: MAX_RECURSION_DEPTH,
        string_table: StringTable {
            arr: Vec::new(),
            n: 0,
            capacity: 0,
        },
        visited: VisitedFileArray {
            arr: Vec::new(),
            n: 0,
            capacity: 0,
        },
        platform: String::new(),
        lib: String::new(),
        osname: String::new(),
        osrel: String::new(),
        rpath_offsets: [0; MAX_RECURSION_DEPTH],
        ld_library_path_offset: 0,
        default_paths_offset: 0,
        ld_so_conf_offset: 0,
        found_all_needed: [false; MAX_RECURSION_DEPTH],
    };

    let mut uname_val = libc::utsname {
        sysname: [0; 65],
        nodename: [0; 65],
        release: [0; 65],
        version: [0; 65],
        machine: [0; 65],
        domainname: [0; 65],
    };

    if unsafe { libc::uname(&mut uname_val) } != 0 {
        std::process::exit(1);
    }

    s.platform = unsafe { CString::from_vec_unchecked(uname_val.machine.to_vec()) }
        .into_string()
        .unwrap();
    s.osname = unsafe { CString::from_vec_unchecked(uname_val.sysname.to_vec()) }
        .into_string()
        .unwrap();
    s.osrel = unsafe { CString::from_vec_unchecked(uname_val.release.to_vec()) }
        .into_string()
        .unwrap();
    s.ld_conf_file = if s.osname == "FreeBSD" {
        "/etc/ld-elf.so.conf".to_string()
    } else {
        "/etc/ld.so.conf".to_string()
    };

    s.lib = "lib".to_string();

    let mut opt_help = false;
    let mut opt_version = false;
    let mut opt_raw = false;

    let mut positional = 1;

    for i in 1..std::env::args().len() {
        let arg = std::env::args().nth(i).unwrap();

        if opt_raw || !arg.starts_with('-') || arg.len() == 1 {
            std::env::args().nth(positional).unwrap().clone();
            positional += 1;
            continue;
        }

        if arg.starts_with("--") {
            if arg.len() == 2 {
                opt_raw = true;
                continue;
            }

            match &arg[2..] {
                "version" => opt_version = true,
                "path" => s.path = true,
                "verbose" => s.verbosity += 1,
                "help" => opt_help = true,
                "ldconf" => {
                    if i + 1 >= std::env::args().len() {
                        eprintln!("Expected value after `--ldconf`");
                        std::process::exit(1);
                    }
                    s.ld_conf_file = std::env::args().nth(i + 1).unwrap().clone();
                }
                "max-depth" => {
                    if i + 1 >= std::env::args().len() {
                        eprintln!("Expected value after `--max-depth`");
                        std::process::exit(1);
                    }
                    s.max_depth = std::env::args().nth(i + 1).unwrap().parse().unwrap();
                    if s.max_depth > MAX_RECURSION_DEPTH {
                        s.max_depth = MAX_RECURSION_DEPTH;
                    }
                }
                _ => {
                    eprintln!("Unrecognized flag `{}`", arg);
                    std::process::exit(1);
                }
            }
            continue;
        }

        for c in arg.chars().skip(1) {
            match c {
                'h' => opt_help = true,
                'p' => s.path = true,
                'v' => s.verbosity += 1,
                _ => {
                    eprintln!("Unrecognized flag `-{}`", c);
                    std::process::exit(1);
                }
            }
        }
    }

    if opt_help || (!opt_version && positional == 1) {
        println!(
            "Show the dynamic dependency tree of ELF files\n\
             Usage: libtree [OPTION]... [--] FILE [FILES]...\n\
             \n\
             -h, --help     Print help info\n\
             --version      Print version info\n\
             \n\
             File names starting with '-', for example '-.so', can be specified as follows:\n\
             libtree -- -.so\n\
             \n\
             Locating libs options:\n\
             -p, --path       Show the path of libraries instead of the soname\n\
             -v               Show libraries skipped by default*\n\
             -vv              Show dependencies of libraries skipped by default*\n\
             -vvv             Show dependencies of already encountered libraries\n\
             --ldconf <path>  Config file for extra search paths [{}]\n\
             --max-depth <n>  Limit library traversal to at most n levels of depth\n\
             \n\
             * For brevity, the following libraries are not shown by default:\n\
             {}",
            s.ld_conf_file,
            EXCLUDE_LIST.join(", ")
        );
        std::process::exit(!opt_help as i32);
    }

    if opt_version {
        println!("{}", VERSION);
        std::process::exit(0);
    }

    let paths: Vec<String> = std::env::args().skip(1).take(positional - 1).collect();
    std::process::exit(print_tree(positional as i32 - 1, &paths, &mut s));
}