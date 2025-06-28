use std::cmp::Ordering;
use std::env;
use std::fs::File;
use std::io::{self, Write, Read, Seek, SeekFrom, stderr, BufReader, BufRead};
use std::path::Path;
use std::process::exit;
use std::str;
use uname::uname;
use byteorder::ReadBytesExt;
use glob::glob;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
use atty::Stream;

// Constants
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

const REGULAR_RED: &str = "\x1b[0;31m";
const BOLD_RED: &str = "\x1b[1;31m";
const CLEAR: &str = "\x1b[0m";
const BOLD_YELLOW: &str = "\x1b[33m";
const BOLD_CYAN: &str = "\x1b[1;36m";
const REGULAR_CYAN: &str = "\x1b[0;36m";
const REGULAR_MAGENTA: &str = "\x1b[0;35m";
const REGULAR_BLUE: &str = "\x1b[0;34m";
const REGULAR: &str = "\x1b[0m";

// Drawing characters
const LIGHT_HORIZONTAL: &str = "─";
const LIGHT_QUADRUPLE_DASH_VERTICAL: &str = "╊";
const LIGHT_UP_AND_RIGHT: &str = "└";
const LIGHT_VERTICAL: &str = "│";
const LIGHT_VERTICAL_AND_RIGHT: &str = "├";

const JUST_INDENT: &str = "    ";
const LIGHT_VERTICAL_WITH_INDENT: &str = "│   ";

const MAX_RECURSION_DEPTH: usize = 32;
const MAX_PATH_LENGTH: usize = 4096;

// Libraries we do not show by default
const EXCLUDE_LIST: [&str; 14] = [
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

// Structs
#[repr(C)]
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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

enum Header {
    Header64(Header64),
    Header32(Header32),
}

#[repr(C)]
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
struct Dyn64 {
    d_tag: i64,
    d_val: u64,
}

#[repr(C)]
#[derive(Debug, Clone)]
struct Dyn32 {
    d_tag: i32,
    d_val: u32,
}

#[derive(Debug, Clone, Copy)]
struct Compat {
    any: bool,    // true iff we don't look for libs matching a certain architecture
    class: u8,    // 32 or 64 bits?
    machine: u16, // instruction set
}

#[derive(Debug, Clone, Copy)]
enum How {
    Input,
    Direct,
    Rpath,
    LDLibraryPath,
    Runpath,
    LDSoConf,
    Default,
}

#[derive(Debug, Clone, Copy)]
struct Found {
    how: How,
    depth: usize,
}

#[derive(Debug)]
struct StringTable {
    arr: Vec<u8>,
}

impl StringTable {
    fn new() -> Self {
        StringTable { arr: Vec::with_capacity(1024) }
    }

    fn store(&mut self, s: &str) {
        self.arr.extend_from_slice(s.as_bytes());
        self.arr.push(0);
    }

    fn copy_from_file(&mut self, f: &mut File) -> io::Result<()> {
        loop {
            let byte = f.read_u8()?;
            if byte == 0 {
                self.arr.push(0);
                break;
            }
            self.arr.push(byte);
        }
        Ok(())
    }

    fn get_str(&self, offset: usize) -> &str {
        let end = self.arr[offset..]
            .iter()
            .position(|&c| c == 0)
            .map(|p| offset + p)
            .unwrap_or(self.arr.len());
        str::from_utf8(&self.arr[offset..end]).unwrap_or("")
    }
}

#[derive(Debug)]
struct VisitedFile {
    st_dev: u64,
    st_ino: u64,
}

#[derive(Debug)]
struct VisitedFileArray {
    arr: Vec<VisitedFile>,
}

impl VisitedFileArray {
    fn new() -> Self {
        VisitedFileArray {
            arr: Vec::with_capacity(256),
        }
    }

    fn contains(&self, needle: &VisitedFile) -> bool {
        self.arr.iter().any(|f| f.st_dev == needle.st_dev && f.st_ino == needle.st_ino)
    }

    fn append(&mut self, new: VisitedFile) {
        self.arr.push(new);
    }
}

#[derive(Debug)]
struct LibTreeState {
    verbosity: usize,
    path: bool,
    color: bool,
    ld_conf_file: String,
    max_depth: usize,

    string_table: StringTable,
    visited: VisitedFileArray,

    // rpath substitutions values
    PLATFORM: String,
    LIB: String,
    OSNAME: String,
    OSREL: String,

    // rpath stack: list of offsets into the string buffer where rpaths start
    rpath_offsets: [usize; MAX_RECURSION_DEPTH],
    ld_library_path_offset: usize,
    default_paths_offset: usize,
    ld_so_conf_offset: usize,

    found_all_needed: [bool; MAX_RECURSION_DEPTH],
}

impl LibTreeState {
    fn new() -> Self {
        LibTreeState {
            verbosity: 0,
            path: false,
            color: false,
            ld_conf_file: "/etc/ld.so.conf".to_string(),
            max_depth: MAX_RECURSION_DEPTH,

            string_table: StringTable::new(),
            visited: VisitedFileArray::new(),

            PLATFORM: String::new(),
            LIB: "lib".to_string(),
            OSNAME: String::new(),
            OSREL: String::new(),

            rpath_offsets: [usize::MAX; MAX_RECURSION_DEPTH],
            ld_library_path_offset: usize::MAX,
            default_paths_offset: usize::MAX,
            ld_so_conf_offset: usize::MAX,

            found_all_needed: [false; MAX_RECURSION_DEPTH],
        }
    }
}

#[derive(Debug)]
#[repr(i32)]
enum ErrorCode {
    InvalidMagic = ERR_INVALID_MAGIC,
    InvalidClass = ERR_INVALID_CLASS,
    InvalidData = ERR_INVALID_DATA,
    InvalidHeader = ERR_INVALID_HEADER,
    InvalidBits = ERR_INVALID_BITS,
    InvalidEndianness = ERR_INVALID_ENDIANNESS,
    NoExecOrDyn = ERR_NO_EXEC_OR_DYN,
    InvalidPhoff = ERR_INVALID_PHOFF,
    InvalidProgHeader = ERR_INVALID_PROG_HEADER,
    CantStat = ERR_CANT_STAT,
    InvalidDynamicSection = ERR_INVALID_DYNAMIC_SECTION,
    InvalidDynamicArrayEntry = ERR_INVALID_DYNAMIC_ARRAY_ENTRY,
    NoStrtab = ERR_NO_STRTAB,
    InvalidSoname = ERR_INVALID_SONAME,
    InvalidRpath = ERR_INVALID_RPATH,
    InvalidRunpath = ERR_INVALID_RUNPATH,
    InvalidNeeded = ERR_INVALID_NEEDED,
    DependencyNotFound = ERR_DEPENDENCY_NOT_FOUND,
    NoPtLoad = ERR_NO_PT_LOAD,
    VaddrsNotOrdered = ERR_VADDRS_NOT_ORDERED,
    CouldNotOpenFile = ERR_COULD_NOT_OPEN_FILE,
    IncompatibleIsa = ERR_INCOMPATIBLE_ISA,
    Unknown = 100, // 默认情况
}

// Helper Functions

fn host_is_little_endian() -> bool {
    let test: u32 = 1;
    let bytes = test.to_le_bytes();
    bytes[0] == 1
}

fn is_ascending_order(v: &[u64]) -> bool {
    v.windows(2).all(|w| w[0] < w[1])
}

fn is_in_exclude_list(soname: &str) -> bool {
    for &exclude in EXCLUDE_LIST.iter() {
        if soname.starts_with(exclude) {
            return true;
        }
    }
    false
}

fn print_colon_delimited_paths(start: &str, indent: &str, stdout: &mut StandardStream) {
    for path in start.split(':').filter(|s| !s.is_empty()) {
        stdout
            .set_color(ColorSpec::new().set_fg(Some(Color::White)))
            .unwrap();
        writeln!(stdout, "{}{}", indent, path).unwrap();
    }
}

fn utoa(mut v: usize) -> String {
    if v == 0 {
        return "0".to_string();
    }
    let mut s = String::new();
    while v > 0 {
        s.push((b'0' + (v % 10) as u8) as char);
        v /= 10;
    }
    s.chars().rev().collect()
}

fn print_line(
    depth: usize,
    name: &str,
    color_bold: ColorSpec,
    color_regular: ColorSpec,
    highlight: bool,
    reason: &Found,
    state: &LibTreeState,
    stdout: &mut StandardStream,
) {
    if depth != 0 {
        for i in 0..depth - 1 {
            if state.found_all_needed[i] {
                write!(stdout, "{}", JUST_INDENT).unwrap();
            } else {
                write!(stdout, "{}", LIGHT_VERTICAL_WITH_INDENT).unwrap();
            }
        }

        if state.found_all_needed[depth - 1] {
            write!(
                stdout,
                "{}{} ",
                LIGHT_UP_AND_RIGHT,
                LIGHT_HORIZONTAL.repeat(2)
            )
                .unwrap();
        } else {
            write!(
                stdout,
                "{}{} ",
                LIGHT_VERTICAL_AND_RIGHT,
                LIGHT_HORIZONTAL.repeat(2)
            )
                .unwrap();
        }
    }

    // Color the filename differently than the path name, if we have a path.
    if highlight {
        if let Some(slash) = name.rfind('/') {
            stdout.set_color(&color_regular).unwrap();
            write!(stdout, "{}", &name[..slash + 1]).unwrap();
            stdout.set_color(&color_bold).unwrap();
            write!(stdout, "{}", &name[slash + 1..]).unwrap();
        } else {
            stdout.set_color(&color_bold).unwrap();
            write!(stdout, "{}", name).unwrap();
        }
    } else {
        stdout.set_color(&color_bold).unwrap();
        write!(stdout, "{}", name).unwrap();
    }

    if highlight {
        stdout
            .set_color(ColorSpec::new().set_fg(Some(Color::Yellow)).set_bold(true))
            .unwrap();
    } else {
        stdout
            .set_color(ColorSpec::new().set_fg(Some(Color::White)))
            .unwrap();
    }

    match reason.how {
        How::Rpath => {
            if reason.depth + 1 >= depth {
                write!(stdout, "[rpath]").unwrap();
            } else {
                let num = utoa(reason.depth + 1);
                write!(stdout, "[rpath of {}]", num).unwrap();
            }
        }
        How::LDLibraryPath => {
            write!(stdout, "[LD_LIBRARY_PATH]").unwrap();
        }
        How::Runpath => {
            write!(stdout, "[runpath]").unwrap();
        }
        How::LDSoConf => {
            if let Some(filename) = Path::new(&state.ld_conf_file)
                .file_name()
                .and_then(|f| f.to_str())
            {
                write!(stdout, "[{}]", filename).unwrap();
            }
        }
        How::Direct => {
            write!(stdout, "[direct]").unwrap();
        }
        How::Default => {
            write!(stdout, "[default path]").unwrap();
        }
        How::Input => {}
    }

    stdout.reset().unwrap();
    writeln!(stdout).unwrap();
}

fn print_error(
    depth: usize,
    needed_not_found: usize,
    needed_buf_offsets: &[usize],
    runpath: Option<&str>,
    state: &mut LibTreeState,
    no_def_lib: bool,
    stdout: &mut StandardStream,
) {
    for i in 0..needed_not_found {
        state.found_all_needed[depth] = i + 1 >= needed_not_found;
        for d in 0..depth + 1 {
            if d < depth {
                if state.found_all_needed[d] {
                    write!(stdout, "{}", JUST_INDENT).unwrap();
                } else {
                    write!(stdout, "{}", LIGHT_VERTICAL_WITH_INDENT).unwrap();
                }
            }
        }
        stdout
            .set_color(ColorSpec::new().set_fg(Some(Color::Red)).set_bold(true))
            .unwrap();
        write!(
            stdout,
            "{} not found",
            state.string_table.get_str(needed_buf_offsets[i])
        )
            .unwrap();
        stdout.reset().unwrap();
        writeln!(stdout).unwrap();
    }

    // If anything was not found, we print the search paths in order they are considered.
    let box_vertical = LIGHT_QUADRUPLE_DASH_VERTICAL.to_string();

    let mut indent = String::new();
    for i in 0..depth {
        if state.found_all_needed[i] {
            indent.push_str(JUST_INDENT);
        } else {
            indent.push_str(LIGHT_VERTICAL_WITH_INDENT);
        }
    }
    indent.push_str(&box_vertical);

    stdout
        .set_color(ColorSpec::new().set_fg(Some(Color::Black)))
        .unwrap();
    writeln!(stdout, "{} Paths considered in this order:", indent).unwrap();
    stdout.reset().unwrap();

    // Consider rpaths only when runpath is empty
    if runpath.is_none() {
        for j in (0..=depth).rev() {
            if state.rpath_offsets[j] != usize::MAX && needed_not_found > 0 {
                let num = j + 1;
                write!(stdout, "{}    depth {}\n", indent, num).unwrap();
                print_colon_delimited_paths(
                    &state.string_table.get_str(state.rpath_offsets[j]),
                    &indent,
                    stdout,
                );
            }
        }
    } else {
        write!(
            stdout,
            "{} 1. rpath is skipped because runpath was set\n",
            indent
        )
            .unwrap();
    }

    // Environment variables
    if state.ld_library_path_offset == usize::MAX {
        write!(
            stdout,
            "{} 2. LD_LIBRARY_PATH was not set\n",
            indent
        )
            .unwrap();
    } else {
        writeln!(stdout, "{} 2. LD_LIBRARY_PATH:", indent).unwrap();
        print_colon_delimited_paths(
            &state.string_table.get_str(state.ld_library_path_offset),
            &format!("{}    ", indent),
            stdout,
        );
    }

    // runpath
    if let Some(runpath_str) = runpath {
        writeln!(stdout, "{} 3. runpath:", indent).unwrap();
        print_colon_delimited_paths(runpath_str, &format!("{}    ", indent), stdout);
    } else {
        write!(stdout, "{} 3. runpath was not set\n", indent).unwrap();
    }

    // ld.so.conf paths
    if no_def_lib {
        write!(
            stdout,
            "{} 4. ld config files not considered due to NODEFLIB flag\n",
            indent
        )
            .unwrap();
    } else {
        writeln!(stdout, "{} 4. ld config files:", indent).unwrap();
        print_colon_delimited_paths(
            &state.string_table.get_str(state.ld_so_conf_offset),
            &format!("{}    ", indent),
            stdout,
        );
    }

    // Standard paths
    if no_def_lib {
        write!(
            stdout,
            "{} 5. Standard paths not considered due to NODEFLIB flag\n",
            indent
        )
            .unwrap();
    } else {
        writeln!(stdout, "{} 5. Standard paths:", indent).unwrap();
        print_colon_delimited_paths(
            &state.string_table.get_str(state.default_paths_offset),
            &format!("{}    ", indent),
            stdout,
        );
    }
}

fn parse_ld_config_file(st: &mut StringTable, path: &str) -> Result<(), ()> {
    let file = File::open(path).map_err(|_| ())?;
    let reader = BufReader::new(file);

    for line_res in reader.lines() {
        let line = line_res.map_err(|_| ())?;
        let trimmed = line.split('#').next().unwrap_or("").trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("include ") {
            let pattern = trimmed.trim_start_matches("include ").trim();
            for entry in glob(pattern).map_err(|_| ())? {
                match entry {
                    Ok(p) => {
                        parse_ld_config_file(st, p.to_str().unwrap()).unwrap_or(());
                    }
                    Err(_) => continue,
                }
            }
        } else {
            st.store(trimmed);
        }
    }

    Ok(())
}

fn parse_ld_so_conf(state: &mut LibTreeState) {
    state.ld_so_conf_offset = state.string_table.arr.len();
    parse_ld_config_file(&mut state.string_table, &state.ld_conf_file).unwrap_or(());
    if state.string_table.arr.len() > state.ld_so_conf_offset {
        if let Some(last) = state.string_table.arr.get_mut(state.string_table.arr.len() - 1) {
            *last = 0;
        }
    } else {
        state.string_table.store("");
    }
}

fn parse_ld_library_path(state: &mut LibTreeState) {
    state.ld_library_path_offset = usize::MAX;
    if let Ok(val) = env::var("LD_LIBRARY_PATH") {
        state.ld_library_path_offset = state.string_table.arr.len();
        state.string_table.store(&val.replace(';', ":"));
    }
}

fn set_default_paths(state: &mut LibTreeState) {
    state.default_paths_offset = state.string_table.arr.len();
    state.string_table.store("/lib:/lib64:/usr/lib:/usr/lib64");
}

fn visited_files_contains(
    files: &VisitedFileArray,
    needle: &VisitedFile,
) -> bool {
    files.contains(needle)
}

fn visited_files_append(files: &mut VisitedFileArray, new: VisitedFile) {
    files.append(new);
}

// The main recurse function
fn recurse(
    current_file: &str,
    depth: usize,
    state: &mut LibTreeState,
    compat: Compat,
    reason: Found,
    stdout: &mut StandardStream,
) -> Result<(), ErrorCode> {
    let mut fptr = File::open(current_file).map_err(|_| ErrorCode::CouldNotOpenFile)?;

    // When we're done recursing, we should give back the memory we've claimed.
    let old_buf_size = state.string_table.arr.len();

    // Parse the header
    let mut e_ident = [0u8; 16];
    fptr.read_exact(&mut e_ident).map_err(|_| ErrorCode::InvalidMagic)?;

    // Find magic elfs
    if e_ident[0] != 0x7f || e_ident[1] != b'E' || e_ident[2] != b'L' || e_ident[3] != b'F' {
        return Err(ErrorCode::InvalidMagic);
    }

    // Do at least *some* header validation
    if e_ident[4] != BITS32 && e_ident[4] != BITS64 {
        return Err(ErrorCode::InvalidClass);
    }

    if e_ident[5] != 0x01 && e_ident[5] != 0x02 {
        return Err(ErrorCode::InvalidData);
    }

    let curr_type = Compat {
        any: false,
        class: e_ident[4],
        machine: 0, // Will be set later
    };
    let is_little_endian = e_ident[5] == 0x01;

    // Make sure that we have matching bits with parent
    if !compat.any && compat.class != curr_type.class {
        return Err(ErrorCode::InvalidBits);
    }

    // Make sure that the elf file has the host's endianness
    if is_little_endian != host_is_little_endian() {
        return Err(ErrorCode::InvalidEndianness);
    }

    // And get the type
    let header = if curr_type.class == BITS64 {
        let mut h64 = Header64 {
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
        let mut buffer = [0u8; std::mem::size_of::<Header64>()];
        fptr.read_exact(&mut buffer).map_err(|_| ErrorCode::InvalidHeader)?;
        if h64.e_type != ET_EXEC && h64.e_type != ET_DYN {
            return Err(ErrorCode::NoExecOrDyn);
        }
        Compat {
            any: false,
            class: curr_type.class,
            machine: h64.e_machine,
        };
        Header::Header64(h64)
    } else {
        let mut h32 = Header32 {
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
        let mut buffer = [0u8; std::mem::size_of::<Header32>()];
        fptr.read_exact(&mut buffer).map_err(|_| ErrorCode::InvalidHeader)?;
        if h32.e_type != ET_EXEC && h32.e_type != ET_DYN {
            return Err(ErrorCode::NoExecOrDyn);
        }
        Compat {
            any: false,
            class: curr_type.class,
            machine: h32.e_machine,
        };
        Header::Header32(h32)
    };

    // Read the program header.
    let mut pt_load_offset = Vec::new();
    let mut pt_load_vaddr = Vec::new();
    let mut p_offset = u64::MAX;

    match header {
        Header::Header64(ref h64) => {
            for _ in 0..h64.e_phnum {
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
                let mut buffer = [0u8; std::mem::size_of::<Header32>()];
                fptr.read_exact(&mut buffer).map_err(|_| ErrorCode::InvalidProgHeader)?;
                if prog.p_type == PT_LOAD {
                    pt_load_offset.push(prog.p_offset);
                    pt_load_vaddr.push(prog.p_vaddr);
                } else if prog.p_type == PT_DYNAMIC {
                    p_offset = prog.p_offset;
                }
            }
        }
        Header::Header32(ref h32) => {
            for _ in 0..h32.e_phnum {
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
                let mut buffer = [0u8; std::mem::size_of::<Header32>()];
                fptr.read_exact(&mut buffer).map_err(|_| ErrorCode::InvalidProgHeader)?;
                if prog.p_type == PT_LOAD {
                    pt_load_offset.push(prog.p_offset as u64);
                    pt_load_vaddr.push(prog.p_vaddr as u64);
                } else if prog.p_type == PT_DYNAMIC {
                    p_offset = prog.p_offset as u64;
                }
            }
        }
    }

    // At this point we're going to store the file as "success"
    // In Rust, we'll get file metadata
    let metadata = std::fs::metadata(current_file).map_err(|_| ErrorCode::CantStat)?;
    #[cfg(unix)]
    let finfo = VisitedFile {
        st_dev: metadata.dev(),
        st_ino: metadata.ino(),
    };
    #[cfg(not(unix))]
    let finfo = VisitedFile {
        st_dev: 0,
        st_ino: 0,
    };

    let seen_before = visited_files_contains(&state.visited, &finfo);
    if !seen_before {
        visited_files_append(&mut state.visited, finfo);
    }

    // No dynamic section?
    if p_offset == u64::MAX {
        // Print the library and return
        let print_name = if state.path {
            current_file
        } else {
            // 假设在字符串表中有 SONAME
            current_file
        };

        let in_exclude_list = is_in_exclude_list(print_name);
        let bold_color = if in_exclude_list {
            ColorSpec::new().set_fg(Some(Color::Magenta)).clone()
        } else if seen_before {
            ColorSpec::new().set_fg(Some(Color::Blue)).clone()
        } else {
            ColorSpec::new().set_fg(Some(Color::Cyan)).set_bold(true).clone()
        };

        let regular_color = if in_exclude_list {
            ColorSpec::new().set_fg(Some(Color::Magenta)).clone()
        } else if seen_before {
            ColorSpec::new().set_fg(Some(Color::Blue)).clone()
        } else {
            ColorSpec::new().set_fg(Some(Color::Cyan)).clone()
        };

        let highlight = !seen_before && !in_exclude_list;
        print_line(
            depth,
            print_name,
            bold_color,
            regular_color,
            highlight,
            &reason,
            state,
            stdout,
        );

        state.string_table.arr.truncate(old_buf_size);
        return Ok(());
    }

    // I guess you always have to load at least a string
    // table, so if there are no PT_LOAD sections, then
    // it is an error.
    if pt_load_offset.is_empty() {
        return Err(ErrorCode::NoPtLoad);
    }

    // Go to the dynamic section
    fptr.seek(SeekFrom::Start(p_offset)).map_err(|_| ErrorCode::InvalidDynamicSection)?;

    // Shared libraries can disable searching in
    // "default" search paths, aka ld.so.conf and
    // /usr/lib etc. At least glibc respects this.
    let mut no_def_lib = false;

    let mut strtab: Option<u64> = None;
    let mut rpath: Option<u64> = None;
    let mut runpath: Option<u64> = None;
    let mut soname: Option<u64> = None;

    // Offsets in strtab
    let mut needed = Vec::new();

    loop {
        let (d_tag, d_val) = match header {
            Header::Header64(_) => {
                let mut dyn_entry = Dyn64 { d_tag: 0, d_val: 0 };
                let mut buffer = [0u8; std::mem::size_of::<Header32>()];
                fptr.read_exact(&mut buffer).map_err(|_| ErrorCode::InvalidDynamicArrayEntry)?;
                (dyn_entry.d_tag, dyn_entry.d_val)
            }
            Header::Header32(_) => {
                let mut dyn_entry = Dyn32 { d_tag: 0, d_val: 0 };
                let mut buffer = [0u8; std::mem::size_of::<Header32>()];
                fptr.read_exact(&mut buffer).map_err(|_| ErrorCode::InvalidDynamicArrayEntry)?;
                (dyn_entry.d_tag as i64, dyn_entry.d_val as u64)
            }
        };

        match d_tag {
            DT_NULL => break,
            DT_STRTAB => strtab = Some(d_val),
            DT_RPATH => rpath = Some(d_val),
            DT_RUNPATH => runpath = Some(d_val),
            DT_NEEDED => needed.push(d_val),
            DT_SONAME => soname = Some(d_val),
            DT_FLAGS_1 => {
                if (d_val & DT_1_NODEFLIB as u64) == DT_1_NODEFLIB as u64 {
                    no_def_lib = true;
                }
            }
            _ => {}
        }
    }

    if strtab.is_none() {
        return Err(ErrorCode::NoStrtab);
    }

    // Let's verify just to be sure that the offsets are ordered.
    if !is_ascending_order(&pt_load_vaddr) {
        return Err(ErrorCode::VaddrsNotOrdered);
    }

    // Find the file offset corresponding to the strtab virtual address
    let strtab_val = strtab.unwrap();
    let mut vaddr_idx = 0;
    while vaddr_idx + 1 < pt_load_vaddr.len() && strtab_val >= pt_load_vaddr[vaddr_idx + 1] {
        vaddr_idx += 1;
    }

    let strtab_offset = pt_load_offset[vaddr_idx] + (strtab_val - pt_load_vaddr[vaddr_idx]);

    // From this point on we actually copy strings from the ELF file into our own string buffer.

    // Copy the current soname
    let soname_buf_offset = state.string_table.arr.len();
    if let Some(soname_val) = soname {
        fptr.seek(SeekFrom::Start(strtab_offset + soname_val))
            .map_err(|_| ErrorCode::InvalidSoname)?;
        state.string_table.copy_from_file(&mut fptr).map_err(|_| ErrorCode::InvalidSoname)?;
    }

    let in_exclude_list = if let Some(soname_val) = soname {
        is_in_exclude_list(&state.string_table.get_str(soname_val as usize))
    } else {
        false
    };

    // No need to recurse deeper when we aren't in very verbose mode.
    let should_recurse = depth < state.max_depth
        && ((!seen_before && !in_exclude_list)
        || (!seen_before && in_exclude_list && state.verbosity >= 2)
        || state.verbosity >= 3);

    if !should_recurse {
        let print_name = if state.path {
            current_file
        } else if let Some(soname_val) = soname {
            state.string_table.get_str(soname_val as usize)
        } else {
            current_file
        };

        let bold_color = if in_exclude_list {
            ColorSpec::new().set_fg(Some(Color::Magenta)).clone()
        } else if seen_before {
            ColorSpec::new().set_fg(Some(Color::Blue)).clone()
        } else {
            ColorSpec::new().set_fg(Some(Color::Cyan)).set_bold(true).clone()
        };

        let regular_color = if in_exclude_list {
            ColorSpec::new().set_fg(Some(Color::Magenta)).clone()
        } else if seen_before {
            ColorSpec::new().set_fg(Some(Color::Blue)).clone()
        } else {
            ColorSpec::new().set_fg(Some(Color::Cyan)).clone()
        };

        let highlight = !seen_before && !in_exclude_list;
        print_line(
            depth,
            &print_name,
            bold_color,
            regular_color,
            highlight,
            &reason,
            state,
            stdout,
        );

        state.string_table.arr.truncate(old_buf_size);
        return Ok(());
    }

    // Store the ORIGIN string.
    let origin = if let Some(last_slash) = current_file.rfind('/') {
        &current_file[..last_slash]
    } else {
        "./"
    };

    // Copy DT_RPATH
    if let Some(rpath_val) = rpath {
        state.rpath_offsets[depth] = state.string_table.arr.len();
        fptr.seek(SeekFrom::Start(strtab_offset + rpath_val))
            .map_err(|_| ErrorCode::InvalidRpath)?;
        state.string_table.copy_from_file(&mut fptr).map_err(|_| ErrorCode::InvalidRpath)?;
        // TODO: Interpolate variables if necessary
    } else {
        state.rpath_offsets[depth] = usize::MAX;
    }

    // Copy DT_RUNPATH
    let runpath_str = if let Some(runpath_val) = runpath {
        let offset = state.string_table.arr.len();
        fptr.seek(SeekFrom::Start(strtab_offset + runpath_val))
            .map_err(|_| ErrorCode::InvalidRunpath)?;
        state.string_table.copy_from_file(&mut fptr).map_err(|_| ErrorCode::InvalidRunpath)?;
        Some(state.string_table.get_str(offset))
    } else {
        None
    };

    // Copy needed libraries.
    let mut needed_buf_offsets = Vec::with_capacity(needed.len());
    for &needed_offset in &needed {
        needed_buf_offsets.push(state.string_table.arr.len());
        fptr.seek(SeekFrom::Start(strtab_offset + needed_offset))
            .map_err(|_| ErrorCode::InvalidNeeded)?;
        state.string_table.copy_from_file(&mut fptr).map_err(|_| ErrorCode::InvalidNeeded)?;
    }

    // Print the library
    let print_name = if state.path {
        current_file
    } else if let Some(soname_val) = soname {
        state.string_table.get_str(soname_val as usize)
    } else {
        current_file
    };

    let bold_color = if in_exclude_list {
        ColorSpec::new().set_fg(Some(Color::Magenta)).clone()
    } else if seen_before {
        ColorSpec::new().set_fg(Some(Color::Blue)).clone()
    } else {
        ColorSpec::new().set_fg(Some(Color::Cyan)).set_bold(true).clone()
    };

    let regular_color = if in_exclude_list {
        ColorSpec::new().set_fg(Some(Color::Magenta)).clone()
    } else if seen_before {
        ColorSpec::new().set_fg(Some(Color::Blue)).clone()
    } else {
        ColorSpec::new().set_fg(Some(Color::Cyan)).clone()
    };

    let highlight = !seen_before && !in_exclude_list;
    print_line(
        depth,
        &print_name,
        bold_color,
        regular_color,
        highlight,
        &reason,
        state,
        stdout,
    );

    // Finally start searching.

    let mut exit_code = Ok(());

    let mut needed_not_found = needed_buf_offsets.len();

    // Skip common libraries if not verbose
    if needed_not_found > 0 && state.verbosity == 0 {
        needed_not_found = needed_buf_offsets
            .iter()
            .filter(|&&offset| !is_in_exclude_list(state.string_table.get_str(offset)))
            .count();
    }

    // Check absolute paths first
    if needed_not_found > 0 {
        for i in 0..needed_not_found {
            let soname = state.string_table.get_str(needed_buf_offsets[i]);
            if soname.contains('/') {
                let path = soname;
                if !path.starts_with('/') {
                    // Not absolute path
                    continue;
                }
                let code = recurse(
                    path,
                    depth + 1,
                    state,
                    Compat {
                        any: true,
                        class: compat.class,
                        machine: compat.machine,
                    },
                    Found {
                        how: How::Direct,
                        depth,
                    },
                    stdout,
                );
                if code.is_err() {
                    exit_code = code;
                    eprintln!("Error [{}]: Not all dependencies were found", current_file);
                }
            }
        }
    }

    // Consider rpaths only when runpath is empty
    if runpath_str.is_none() {
        for j in (0..=depth).rev() {
            if state.rpath_offsets[j] != usize::MAX && needed_not_found > 0 {
                // Implement search paths based on rpath_offsets[j]
                // Placeholder for actual search logic
                // 您需要根据 rpath_offsets[j] 的值搜索所需的库并递归调用 `recurse`
            }
        }
    }

    // Then try LD_LIBRARY_PATH, if we have it.
    if needed_not_found > 0 && state.ld_library_path_offset != usize::MAX {
        // Implement search paths based on ld_library_path_offset
        // Placeholder for actual search logic
    }

    // Then consider runpaths
    if needed_not_found > 0 && runpath_str.is_some() {
        // Implement search paths based on runpath
        // Placeholder for actual search logic
    }

    // Check ld.so.conf paths
    if needed_not_found > 0 && !no_def_lib {
        // Implement search paths based on ld_so_conf_offset
        // Placeholder for actual search logic
    }

    // Then consider standard paths
    if needed_not_found > 0 && !no_def_lib {
        // Implement search paths based on default_paths_offset
        // Placeholder for actual search logic
    }

    // Finally summarize those that could not be found.
    if needed_not_found > 0 {
        print_error(
            depth,
            needed_not_found,
            &needed_buf_offsets,
            runpath_str,
            state,
            no_def_lib,
            stdout,
        );
        state.string_table.arr.truncate(old_buf_size);
        return Err(ErrorCode::DependencyNotFound);
    }

    // Free memory in our string table
    state.string_table.arr.truncate(old_buf_size);
    Ok(())
}

fn main() {
    // Initialize LibTreeState
    let mut state = LibTreeState::new();

    // Enable or disable colors (no-color.com)
    state.color = env::var("NO_COLOR").is_err() && atty::is(Stream::Stdout);

    // Parse uname
    let uname_val = match uname() {
        Ok(u) => u,
        Err(_) => {
            eprintln!("Failed to get system information");
            exit(1);
        }
    };

    state.PLATFORM = uname_val.machine;
    state.OSNAME = uname_val.sysname.clone();
    state.OSREL = uname_val.release;
    state.ld_conf_file = "/etc/ld.so.conf".to_string();

    if uname_val.sysname == "FreeBSD" {
        state.ld_conf_file = "/etc/ld-elf.so.conf".to_string();
    }

    // 'LIB' is set to "lib"
    state.LIB = "lib".to_string();

    // Command-line argument parsing
    let args: Vec<String> = env::args().collect();
    let mut positional_args = Vec::new();
    let mut opt_help = false;
    let mut opt_version = false;
    let mut opt_raw = false;

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if opt_raw || !arg.starts_with('-') || arg == "-" {
            positional_args.push(arg.clone());
            i += 1;
            continue;
        }

        if arg.starts_with("--") {
            let flag = &arg[2..];
            if flag == "version" {
                opt_version = true;
            } else if flag == "path" {
                state.path = true;
            } else if flag == "verbose" {
                state.verbosity += 1;
            } else if flag == "help" {
                opt_help = true;
            } else if flag == "ldconf" {
                if i + 1 >= args.len() {
                    eprintln!("Expected value after `--ldconf`");
                    exit(1);
                }
                state.ld_conf_file = args[i + 1].clone();
                i += 1;
            } else if flag == "max-depth" {
                if i + 1 >= args.len() {
                    eprintln!("Expected value after `--max-depth`");
                    exit(1);
                }
                state.max_depth = args[i + 1].parse().unwrap_or(MAX_RECURSION_DEPTH);
                if state.max_depth > MAX_RECURSION_DEPTH {
                    state.max_depth = MAX_RECURSION_DEPTH;
                }
                i += 1;
            } else {
                eprintln!("Unrecognized flag `--{}`", flag);
                exit(1);
            }
        } else {
            // Short flags
            for ch in arg[1..].chars() {
                match ch {
                    'h' => opt_help = true,
                    'p' => state.path = true,
                    'v' => state.verbosity += 1,
                    _ => {
                        eprintln!("Unrecognized flag `-{}'", ch);
                        exit(1);
                    }
                }
            }
        }
        i += 1;
    }

    // Print help message
    if opt_help || (!opt_version && positional_args.is_empty()) {
        // clang-format off
        println!("Show the dynamic dependency tree of ELF files");
        println!("Usage: libtree [OPTION]... [--] FILE [FILES]...");
        println!();
        println!("  -h, --help     Print help info");
        println!("      --version  Print version info");
        println!();
        println!("File names starting with '-', for example '-.so', can be specified as follows:");
        println!("  libtree -- -.so");
        println!();
        println!("Locating libs options:");
        println!("  -p, --path       Show the path of libraries instead of the soname");
        println!("  -v               Show libraries skipped by default*");
        println!("  -vv              Show dependencies of libraries skipped by default*");
        println!("  -vvv             Show dependencies of already encountered libraries");
        println!("  --ldconf <path>  Config file for extra search paths [{}]", state.ld_conf_file);
        println!("  --max-depth <n>  Limit library traversal to at most n levels of depth");
        println!();
        println!("* For brevity, the following libraries are not shown by default:");
        // Print a comma separated list of skipped libraries,
        // with some new lines every now and then to make it readable.
        for (j, &exclude) in EXCLUDE_LIST.iter().enumerate() {
            print!("{}", exclude);
            if j + 1 != EXCLUDE_LIST.len() {
                print!(", ");
            }
            if (j + 1) % 3 == 0 && j + 1 != EXCLUDE_LIST.len() {
                println!();
                print!("  ");
            }
        }
        println!();
        println!();
        println!("The following rpath/runpath substitutions are used:");
        println!("  PLATFORM       {}", state.PLATFORM);
        println!("  LIB            {}", state.LIB);
        println!("  OSNAME         {}", state.OSNAME);
        println!("  OSREL          {}", state.OSREL);
        println!();
        exit(!opt_help as i32);
    }

    if opt_version {
        println!("{}", VERSION);
        exit(0);
    }

    // Initialize LibTreeState
    parse_ld_so_conf(&mut state);
    parse_ld_library_path(&mut state);
    set_default_paths(&mut state);

    let mut exit_code = Ok(());

    let mut stdout = StandardStream::stdout(ColorChoice::Auto);

    for path in positional_args {
        let result = recurse(
            &path,
            0,
            &mut state,
            Compat {
                any: true,
                class: 0,
                machine: 0,
            },
            Found {
                how: How::Input,
                depth: 0,
            },
            &mut stdout,
        );

        if let Err(code) = result {
            // exit_code = Err(code);
            let mut stderr = stderr();
            writeln!(stderr, "Error [{}]: ", path).unwrap();
            match code {
                ErrorCode::InvalidMagic => {
                    writeln!(stderr, "Invalid ELF magic bytes").unwrap();
                }
                ErrorCode::InvalidClass => {
                    writeln!(stderr, "Invalid ELF class").unwrap();
                }
                ErrorCode::InvalidData => {
                    writeln!(stderr, "Invalid ELF data").unwrap();
                }
                ErrorCode::InvalidHeader => {
                    writeln!(stderr, "Invalid ELF header").unwrap();
                }
                ErrorCode::InvalidBits => {
                    writeln!(stderr, "Invalid bits").unwrap();
                }
                ErrorCode::InvalidEndianness => {
                    writeln!(stderr, "Invalid endianness").unwrap();
                }
                ErrorCode::NoExecOrDyn => {
                    writeln!(stderr, "Not an ET_EXEC or ET_DYN ELF file").unwrap();
                }
                ErrorCode::InvalidPhoff => {
                    writeln!(stderr, "Invalid ELF program header offset").unwrap();
                }
                ErrorCode::InvalidProgHeader => {
                    writeln!(stderr, "Invalid ELF program header").unwrap();
                }
                ErrorCode::CantStat => {
                    writeln!(stderr, "Can't stat file").unwrap();
                }
                ErrorCode::InvalidDynamicSection => {
                    writeln!(stderr, "Invalid ELF dynamic section").unwrap();
                }
                ErrorCode::InvalidDynamicArrayEntry => {
                    writeln!(stderr, "Invalid ELF dynamic array entry").unwrap();
                }
                ErrorCode::NoStrtab => {
                    writeln!(stderr, "No ELF string table found").unwrap();
                }
                ErrorCode::InvalidSoname => {
                    writeln!(stderr, "Can't read DT_SONAME").unwrap();
                }
                ErrorCode::InvalidRpath => {
                    writeln!(stderr, "Can't read DT_RPATH").unwrap();
                }
                ErrorCode::InvalidRunpath => {
                    writeln!(stderr, "Can't read DT_RUNPATH").unwrap();
                }
                ErrorCode::InvalidNeeded => {
                    writeln!(stderr, "Can't read DT_NEEDED").unwrap();
                }
                ErrorCode::DependencyNotFound => {
                    writeln!(stderr, "Not all dependencies were found").unwrap();
                }
                ErrorCode::NoPtLoad => {
                    writeln!(stderr, "No PT_LOAD found in ELF file").unwrap();
                }
                ErrorCode::VaddrsNotOrdered => {
                    writeln!(stderr, "Virtual addresses are not ordered").unwrap();
                }
                ErrorCode::CouldNotOpenFile => {
                    writeln!(stderr, "Could not open file").unwrap();
                }
                ErrorCode::IncompatibleIsa => {
                    writeln!(stderr, "Incompatible ISA").unwrap();
                }
                ErrorCode::Unknown => {
                    writeln!(stderr, "Unknown error").unwrap();
                }
            }
        }
    }

    // Determine exit code
    match exit_code {
        Ok(_) => exit(0),
        Err(code) => exit(code as i32),
    }
}
