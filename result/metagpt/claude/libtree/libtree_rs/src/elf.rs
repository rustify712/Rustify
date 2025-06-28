//! ELF file format handling module.
//!
//! This module provides types and functions for parsing ELF (Executable and Linkable Format)
//! files, including headers, program headers, and dynamic sections.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::mem;
use std::path::Path;

use crate::error::{Error, Result};
use crate::utils::{host_is_little_endian, is_ascending_order};

/// ELF file class (32-bit or 64-bit)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElfClass {
    Bits32 = 1,
    Bits64 = 2,
}

/// ELF file type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElfType {
    None = 0,
    Rel = 1,
    Exec = 2,
    Dyn = 3,
    Core = 4,
}

/// Compatibility information for ELF files
#[derive(Debug, Clone)]
pub struct CompatType {
    /// Whether to accept any architecture
    pub any: bool,
    /// 32-bit or 64-bit
    pub class: ElfClass,
    /// Instruction set architecture
    pub machine: u16,
}

// ELF header constants
const EI_NIDENT: usize = 16;
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
const DT_FLAGS_1: i64 = 0x6ffffffb;
const DT_1_NODEFLIB: u64 = 0x800;

/// 64-bit ELF header
#[repr(C)]
#[derive(Debug, Clone, Copy)]
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

/// 32-bit ELF header
#[repr(C)]
#[derive(Debug, Clone, Copy)]
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

/// 64-bit program header
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ProgramHeader64 {
    pub p_type: u32,
    pub p_flags: u32,
    pub p_offset: u64,
    pub p_vaddr: u64,
    pub p_paddr: u64,
    pub p_filesz: u64,
    pub p_memsz: u64,
    pub p_align: u64,
}

/// 32-bit program header
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ProgramHeader32 {
    pub p_type: u32,
    pub p_offset: u32,
    pub p_vaddr: u32,
    pub p_paddr: u32,
    pub p_filesz: u32,
    pub p_memsz: u32,
    pub p_flags: u32,
    pub p_align: u32,
}

/// 64-bit dynamic section entry
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Dynamic64 {
    pub d_tag: i64,
    pub d_val: u64,
}

/// 32-bit dynamic section entry
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Dynamic32 {
    pub d_tag: i32,
    pub d_val: u32,
}

/// Results from parsing the dynamic section
#[derive(Debug, Default)]
pub struct DynamicInfo {
    pub soname: Option<String>,
    pub needed: Vec<String>,
    pub rpath: Vec<String>,
    pub runpath: Option<String>,
    pub no_default_lib: bool,
}

/// ELF file parser
pub struct ElfParser {
    file: File,
    class: ElfClass,
    machine: u16,
    is_little_endian: bool,
}

impl ElfParser {
    /// Create a new ELF parser from a file
    pub fn new(mut file: File) -> Result<Self> {
        let mut ident = [0u8; EI_NIDENT];
        file.read_exact(&mut ident)?;

        // Check ELF magic number
        if ident[0..4] != [0x7f, b'E', b'L', b'F'] {
            return Err(Error::InvalidMagic);
        }

        // Check class
        let class = match ident[4] {
            1 => ElfClass::Bits32,
            2 => ElfClass::Bits64,
            _ => return Err(Error::InvalidClass),
        };

        // Check data format (endianness)
        let is_little_endian = match ident[5] {
            1 => true,
            2 => false,
            _ => return Err(Error::InvalidData),
        };

        // Check host endianness compatibility
        if is_little_endian != host_is_little_endian() {
            return Err(Error::InvalidEndianness);
        }

        Ok(Self {
            file,
            class,
            machine: 0, // Will be set after reading the header
            is_little_endian,
        })
    }

    /// Get the ELF class
    pub fn class(&self) -> ElfClass {
        self.class
    }

    /// Read the ELF header
    pub fn read_header(&mut self) -> Result<(ElfType, u16)> {
        match self.class {
            ElfClass::Bits64 => {
                let mut header: Header64 = unsafe { mem::zeroed() };
                self.file.read_exact(unsafe {
                    std::slice::from_raw_parts_mut(
                        &mut header as *mut _ as *mut u8,
                        mem::size_of::<Header64>(),
                    )
                })?;

                self.machine = header.e_machine;

                match header.e_type {
                    ET_EXEC => Ok((ElfType::Exec, header.e_machine)),
                    ET_DYN => Ok((ElfType::Dyn, header.e_machine)),
                    _ => Err(Error::NotExecOrDyn),
                }
            }
            ElfClass::Bits32 => {
                let mut header: Header32 = unsafe { mem::zeroed() };
                self.file.read_exact(unsafe {
                    std::slice::from_raw_parts_mut(
                        &mut header as *mut _ as *mut u8,
                        mem::size_of::<Header32>(),
                    )
                })?;

                self.machine = header.e_machine;

                match header.e_type {
                    ET_EXEC => Ok((ElfType::Exec, header.e_machine)),
                    ET_DYN => Ok((ElfType::Dyn, header.e_machine)),
                    _ => Err(Error::NotExecOrDyn),
                }
            }
        }
    }

    /// Parse the dynamic section and extract needed information
    pub fn parse_dynamic_section(&mut self) -> Result<DynamicInfo> {
        let mut info = DynamicInfo::default();
        let mut strtab_addr = None;
        let mut dynamic_strings = Vec::new();

        // Read program headers to find PT_DYNAMIC and PT_LOAD segments
        let mut pt_load_offsets = Vec::new();
        let mut pt_load_vaddrs = Vec::new();
        let mut dynamic_offset = None;

        match self.class {
            ElfClass::Bits64 => self.parse_program_headers_64(
                &mut pt_load_offsets,
                &mut pt_load_vaddrs,
                &mut dynamic_offset,
            )?,
            ElfClass::Bits32 => self.parse_program_headers_32(
                &mut pt_load_offsets,
                &mut pt_load_vaddrs,
                &mut dynamic_offset,
            )?,
        }

        // Verify PT_LOAD segments
        if pt_load_offsets.is_empty() {
            return Err(Error::NoPtLoad);
        }
        if !is_ascending_order(&pt_load_vaddrs) {
            return Err(Error::VaddrsNotOrdered);
        }

        // Parse dynamic section
        if let Some(offset) = dynamic_offset {
            self.file.seek(SeekFrom::Start(offset))?;
            match self.class {
                ElfClass::Bits64 => {
                    self.parse_dynamic_entries_64(&mut info, &mut strtab_addr, &mut dynamic_strings)?
                }
                ElfClass::Bits32 => {
                    self.parse_dynamic_entries_32(&mut info, &mut strtab_addr, &mut dynamic_strings)?
                }
            }
        }

        // Process string table
        if let Some(addr) = strtab_addr {
            self.process_string_table(addr, &pt_load_offsets, &pt_load_vaddrs, &dynamic_strings, &mut info)?;
        }

        Ok(info)
    }

    // Private helper methods
    fn parse_program_headers_64(
        &mut self,
        pt_load_offsets: &mut Vec<u64>,
        pt_load_vaddrs: &mut Vec<u64>,
        dynamic_offset: &mut Option<u64>,
    ) -> Result<()> {
        let mut header: Header64 = unsafe { mem::zeroed() };
        self.file.read_exact(unsafe {
            std::slice::from_raw_parts_mut(
                &mut header as *mut _ as *mut u8,
                mem::size_of::<Header64>(),
            )
        })?;

        for _ in 0..header.e_phnum {
            let mut prog: ProgramHeader64 = unsafe { mem::zeroed() };
            self.file.read_exact(unsafe {
                std::slice::from_raw_parts_mut(
                    &mut prog as *mut _ as *mut u8,
                    mem::size_of::<ProgramHeader64>(),
                )
            })?;

            match prog.p_type {
                PT_LOAD => {
                    pt_load_offsets.push(prog.p_offset);
                    pt_load_vaddrs.push(prog.p_vaddr);
                }
                PT_DYNAMIC => {
                    *dynamic_offset = Some(prog.p_offset);
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn parse_program_headers_32(
        &mut self,
        pt_load_offsets: &mut Vec<u64>,
        pt_load_vaddrs: &mut Vec<u64>,
        dynamic_offset: &mut Option<u64>,
    ) -> Result<()> {
        let mut header: Header32 = unsafe { mem::zeroed() };
        self.file.read_exact(unsafe {
            std::slice::from_raw_parts_mut(
                &mut header as *mut _ as *mut u8,
                mem::size_of::<Header32>(),
            )
        })?;

        for _ in 0..header.e_phnum {
            let mut prog: ProgramHeader32 = unsafe { mem::zeroed() };
            self.file.read_exact(unsafe {
                std::slice::from_raw_parts_mut(
                    &mut prog as *mut _ as *mut u8,
                    mem::size_of::<ProgramHeader32>(),
                )
            })?;

            match prog.p_type {
                PT_LOAD => {
                    pt_load_offsets.push(prog.p_offset as u64);
                    pt_load_vaddrs.push(prog.p_vaddr as u64);
                }
                PT_DYNAMIC => {
                    *dynamic_offset = Some(prog.p_offset as u64);
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn parse_dynamic_entries_64(
        &mut self,
        info: &mut DynamicInfo,
        strtab_addr: &mut Option<u64>,
        dynamic_strings: &mut Vec<u64>,
    ) -> Result<()> {
        loop {
            let mut entry: Dynamic64 = unsafe { mem::zeroed() };
            self.file.read_exact(unsafe {
                std::slice::from_raw_parts_mut(
                    &mut entry as *mut _ as *mut u8,
                    mem::size_of::<Dynamic64>(),
                )
            })?;

            match entry.d_tag {
                DT_NULL => break,
                DT_STRTAB => *strtab_addr = Some(entry.d_val),
                DT_NEEDED => dynamic_strings.push(entry.d_val),
                DT_SONAME => dynamic_strings.push(entry.d_val),
                DT_RPATH => dynamic_strings.push(entry.d_val),
                DT_RUNPATH => dynamic_strings.push(entry.d_val),
                DT_FLAGS_1 if (entry.d_val & DT_1_NODEFLIB as u64) != 0 => info.no_default_lib = true,
                _ => {}
            }
        }
        Ok(())
    }

    fn parse_dynamic_entries_32(
        &mut self,
        info: &mut DynamicInfo,
        strtab_addr: &mut Option<u64>,
        dynamic_strings: &mut Vec<u64>,
    ) -> Result<()> {
        loop {
            let mut entry: Dynamic32 = unsafe { mem::zeroed() };
            self.file.read_exact(unsafe {
                std::slice::from_raw_parts_mut(
                    &mut entry as *mut _ as *mut u8,
                    mem::size_of::<Dynamic32>(),
                )
            })?;

            match entry.d_tag {
                DT_NULL => break,
                DT_STRTAB => *strtab_addr = Some(entry.d_val as u64),
                DT_NEEDED => dynamic_strings.push(entry.d_val as u64),
                DT_SONAME => dynamic_strings.push(entry.d_val as u64),
                DT_RPATH => dynamic_strings.push(entry.d_val as u64),
                DT_RUNPATH => dynamic_strings.push(entry.d_val as u64),
                DT_FLAGS_1 if (entry.d_val as u64 & DT_1_NODEFLIB as u64) != 0 => {
                    info.no_default_lib = true
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn process_string_table(
        &mut self,
        strtab_addr: u64,
        pt_load_offsets: &[u64],
        pt_load_vaddrs: &[u64],
        dynamic_strings: &[u64],
        info: &mut DynamicInfo,
    ) -> Result<()> {
        // Find the file offset corresponding to the string table virtual address
        let mut idx = 0;
        while idx + 1 < pt_load_vaddrs.len() && strtab_addr >= pt_load_vaddrs[idx + 1] {
            idx += 1;
        }

        let strtab_offset = pt_load_offsets[idx] + (strtab_addr - pt_load_vaddrs[idx]);

        // Read strings from the string table
        for &offset in dynamic_strings {
            self.file.seek(SeekFrom::Start(strtab_offset + offset))?;
            let mut string = Vec::new();
            loop {
                let mut byte = [0u8];
                self.file.read_exact(&mut byte)?;
                if byte[0] == 0 {
                    break;
                }
                string.push(byte[0]);
            }
            let string = String::from_utf8(string).map_err(|_| Error::InvalidData)?;

            // Store the string in the appropriate field
            if offset == dynamic_strings[0] {
                info.soname = Some(string);
            } else if offset == dynamic_strings[1] {
                info.needed.push(string);
            } else if offset == dynamic_strings[2] {
                info.rpath.push(string);
            } else if offset == dynamic_strings[3] {
                info.runpath = Some(string);
            }
        }

        Ok(())
    }
}