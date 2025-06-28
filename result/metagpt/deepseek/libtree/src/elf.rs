//! ELF文件解析模块

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::mem::size_of;
use std::ffi::CStr;

use crate::{
    Header32, Header64, Prog32, Prog64, Dyn32, Dyn64,
    ET_EXEC, ET_DYN, PT_LOAD, PT_DYNAMIC,
    DT_NULL, DT_NEEDED, DT_STRTAB, DT_SONAME, DT_RPATH, DT_RUNPATH, DT_STRSZ
};

#[derive(Debug)]
pub enum ElfError {
    InvalidMagic,
    InvalidClass,
    InvalidData,
    InvalidHeader,
    InvalidBits,
    InvalidEndianness,
    NoExecOrDyn,
    InvalidPHoff,
    InvalidProgHeader,
    CantStat,
    InvalidDynamicSection,
    InvalidDynamicArrayEntry,
    NoStrtab,
    InvalidSoname,
    InvalidRpath,
    InvalidRunpath,
    InvalidNeeded,
    DependencyNotFound,
    NoPTLoad,
    VaddrsNotOrdered,
    CouldNotOpenFile,
    IncompatibleISA,
    IoError(std::io::Error),
}

impl From<std::io::Error> for ElfError {
    fn from(err: std::io::Error) -> Self {
        ElfError::IoError(err)
    }
}

impl std::fmt::Display for ElfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct ElfFile {
    file: File,
    is_64bit: bool,
    is_little_endian: bool,
    header: Header64,
    program_headers: Vec<Prog64>,
    dynamic_entries: Vec<Dyn64>,
    strtab_offset: u64,
    strtab_size: u64,
}

impl ElfFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, ElfError> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;

        if magic != [0x7f, b'E', b'L', b'F'] {
            return Err(ElfError::InvalidMagic);
        }

        let mut class = [0u8; 1];
        file.read_exact(&mut class)?;
        let is_64bit = class[0] == 2;

        let mut endian = [0u8; 1];
        file.read_exact(&mut endian)?;
        let is_little_endian = endian[0] == 1;

        let header = if is_64bit {
            let mut header = Header64::default();
            file.read_exact(unsafe {
                std::slice::from_raw_parts_mut(
                    &mut header as *mut _ as *mut u8,
                    size_of::<Header64>()
                )
            })?;
            header
        } else {
            let mut header32 = Header32::default();
            file.read_exact(unsafe {
                std::slice::from_raw_parts_mut(
                    &mut header32 as *mut _ as *mut u8,
                    size_of::<Header32>()
                )
            })?;
            Header64::from(header32)
        };

        if header.e_type != ET_EXEC && header.e_type != ET_DYN {
            return Err(ElfError::NoExecOrDyn);
        }

        let mut elf = ElfFile {
            file,
            is_64bit,
            is_little_endian,
            header,
            program_headers: Vec::new(),
            dynamic_entries: Vec::new(),
            strtab_offset: 0,
            strtab_size: 0,
        };

        elf.load_program_headers()?;
        elf.load_dynamic_entries()?;

        Ok(elf)
    }

    fn load_program_headers(&mut self) -> Result<(), ElfError> {
        self.file.seek(SeekFrom::Start(self.header.e_phoff))?;

        for _ in 0..self.header.e_phnum {
            if self.is_64bit {
                let mut phdr = Prog64::default();
                self.file.read_exact(unsafe {
                    std::slice::from_raw_parts_mut(
                        &mut phdr as *mut _ as *mut u8,
                        size_of::<Prog64>()
                    )
                })?;
                self.program_headers.push(phdr);
            } else {
                let mut phdr32 = Prog32::default();
                self.file.read_exact(unsafe {
                    std::slice::from_raw_parts_mut(
                        &mut phdr32 as *mut _ as *mut u8,
                        size_of::<Prog32>()
                    )
                })?;
                self.program_headers.push(Prog64::from(phdr32));
            }
        }

        Ok(())
    }

    fn load_dynamic_entries(&mut self) -> Result<(), ElfError> {
        let dynamic_phdr = self.program_headers.iter()
            .find(|ph| ph.p_type == PT_DYNAMIC)
            .ok_or(ElfError::InvalidDynamicSection)?;

        self.file.seek(SeekFrom::Start(dynamic_phdr.p_offset))?;

        loop {
            if self.is_64bit {
                let mut dyn_entry = Dyn64::default();
                self.file.read_exact(unsafe {
                    std::slice::from_raw_parts_mut(
                        &mut dyn_entry as *mut _ as *mut u8,
                        size_of::<Dyn64>()
                    )
                })?;
                
                if dyn_entry.d_tag == DT_NULL {
                    break;
                }
                
                self.dynamic_entries.push(dyn_entry);
            } else {
                let mut dyn_entry32 = Dyn32::default();
                self.file.read_exact(unsafe {
                    std::slice::from_raw_parts_mut(
                        &mut dyn_entry32 as *mut _ as *mut u8,
                        size_of::<Dyn32>()
                    )
                })?;
                
                if dyn_entry32.d_tag == DT_NULL as i32 {
                    break;
                }
                
                self.dynamic_entries.push(Dyn64::from(dyn_entry32));
            }
        }

        Ok(())
    }

    pub fn get_needed_libs(&mut self) -> Result<Vec<String>, ElfError> {
        let strtab_phdr = self.dynamic_entries.iter()
            .find(|d| d.d_tag == DT_STRTAB)
            .ok_or(ElfError::NoStrtab)?;

        self.strtab_offset = strtab_phdr.d_val;
        self.strtab_size = self.dynamic_entries.iter()
            .find(|d| d.d_tag == DT_STRSZ)
            .map(|d| d.d_val)
            .unwrap_or(0);

        let mut strtab = vec![0u8; self.strtab_size as usize];
        self.file.seek(SeekFrom::Start(self.strtab_offset))?;
        self.file.read_exact(&mut strtab)?;

        let mut libs = Vec::new();
        for entry in &self.dynamic_entries {
            if entry.d_tag == DT_NEEDED {
                let name_ptr = entry.d_val as usize;
                if name_ptr >= strtab.len() {
                    return Err(ElfError::InvalidNeeded);
                }
                
                let cstr = CStr::from_bytes_until_nul(&strtab[name_ptr..])
                    .map_err(|_| ElfError::InvalidNeeded)?;
                libs.push(cstr.to_string_lossy().into_owned());
            }
        }

        Ok(libs)
    }
}