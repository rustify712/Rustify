use std::ptr;
use std::mem;
use std::fmt;
use std::cmp;
use std::ops;

const DEFAULT_SIZE: usize = 1024;

#[derive(Debug)]
pub struct Buffer {
    data: Vec<u8>,
    len: usize,
}

impl Buffer {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_SIZE)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        data.resize(capacity, 0);
        Self {
            data,
            len: 0,
        }
    }

    pub fn from_str(s: &str) -> Self {
        let mut buf = Self::with_capacity(s.len());
        buf.data[..s.len()].copy_from_slice(s.as_bytes());
        buf.len = s.len();
        buf
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    pub fn resize(&mut self, new_len: usize) {
        self.data.resize(new_len, 0);
        self.len = new_len;
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
        self.len = 0;
    }

    pub fn as_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(&self.data[..self.len]) }
    }

    pub fn append(&mut self, s: &str) -> Result<(), String> {
        let new_len = self.len + s.len();
        if new_len > self.data.capacity() {
            self.data.resize(new_len, 0);
        }
        self.data[self.len..new_len].copy_from_slice(s.as_bytes());
        self.len = new_len;
        Ok(())
    }

    pub fn prepend(&mut self, s: &str) -> Result<(), String> {
        let new_len = self.len + s.len();
        if new_len > self.data.capacity() {
            self.data.resize(new_len, 0);
        }
        self.data.copy_within(0..self.len, s.len());
        self.data[0..s.len()].copy_from_slice(s.as_bytes());
        self.len = new_len;
        Ok(())
    }

    pub fn slice(&self, start: usize, end: usize) -> Result<Self, String> {
        if start > end || end > self.len {
            return Err("Invalid slice range".to_string());
        }
        let mut new_buf = Self::with_capacity(end - start);
        new_buf.data[..end-start].copy_from_slice(&self.data[start..end]);
        new_buf.len = end - start;
        Ok(new_buf)
    }

    pub fn trim_left(&mut self) {
        let mut i = 0;
        while i < self.len && self.data[i].is_ascii_whitespace() {
            i += 1;
        }
        if i > 0 {
            self.data.copy_within(i..self.len, 0);
            self.len -= i;
        }
    }

    pub fn trim_right(&mut self) {
        while self.len > 0 && self.data[self.len-1].is_ascii_whitespace() {
            self.len -= 1;
        }
    }

    pub fn trim(&mut self) {
        self.trim_left();
        self.trim_right();
    }

    pub fn format(&mut self, args: std::fmt::Arguments) -> Result<(), String> {
        let s = args.to_string();
        self.append(&s)
    }

    pub fn appendf(&mut self, format: &str, args: std::fmt::Arguments) -> Result<(), String> {
        let s = format!("{}", args);
        self.append(&s)
    }

    pub fn index_of(&self, pattern: &str) -> Option<usize> {
        let haystack = self.as_str();
        haystack.find(pattern)
    }

    pub fn equals(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }

    pub fn fill(&mut self, byte: u8) {
        self.data.fill(byte);
    }

    pub fn print_hex(&self) {
        println!(" ");
        for (i, &byte) in self.data[..self.len].iter().enumerate() {
            print!(" {:02x}", byte);
            if (i + 1) % 8 == 0 {
                println!();
                print!(" ");
            }
        }
        println!();
    }
}