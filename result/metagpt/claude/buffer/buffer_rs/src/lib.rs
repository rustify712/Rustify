use std::fmt;
use std::str;

/// Default buffer size in bytes
pub const DEFAULT_SIZE: usize = 64;

/// A dynamic buffer implementation that provides safe string manipulation
#[derive(Debug, Clone)]
pub struct Buffer {
    data: Vec<u8>,
    len: usize,
}

impl Buffer {
    /// Creates a new buffer with default capacity
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_SIZE)
    }

    /// Creates a new buffer with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        data.resize(capacity, 0);
        Self {
            data,
            len: 0,
        }
    }

    /// Creates a new buffer from a string slice
    pub fn from_str(s: &str) -> Self {
        let mut buf = Self::with_capacity(s.len());
        buf.data[..s.len()].copy_from_slice(s.as_bytes());
        buf.len = s.len();
        buf
    }

    /// Creates a new buffer from a string slice with specified length
    pub fn from_str_with_len(s: &str, len: usize) -> Self {
        let mut buf = Self::with_capacity(len);
        let copy_len = std::cmp::min(s.len(), len);
        buf.data[..copy_len].copy_from_slice(&s.as_bytes()[..copy_len]);
        buf.len = copy_len;
        buf
    }

    /// Creates a new buffer with a copy of the given string
    pub fn from_string(s: String) -> Self {
        Self::from_str(&s)
    }

    /// Returns the current length of the buffer
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the total capacity of the buffer
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Resizes the buffer to the new length
    pub fn resize(&mut self, new_len: usize) -> Result<(), String> {
        let aligned_size = Self::nearest_multiple_of_1024(new_len);
        self.data.resize(aligned_size, 0);
        self.len = new_len;
        Ok(())
    }

    /// Compacts the buffer by removing excess capacity
    pub fn compact(&mut self) -> Result<usize, String> {
        let old_capacity = self.data.capacity();
        let mut new_data = Vec::with_capacity(self.len);
        new_data.extend_from_slice(&self.data[..self.len]);
        self.data = new_data;
        Ok(old_capacity - self.capacity())
    }

    /// Clears the buffer content
    pub fn clear(&mut self) {
        self.data.fill(0);
        self.len = 0;
    }

    /// Returns the buffer content as a string slice
    pub fn as_str(&self) -> Result<&str, str::Utf8Error> {
        str::from_utf8(&self.data[..self.len])
    }

    /// Appends a string slice to the buffer
    pub fn append(&mut self, s: &str) -> Result<(), String> {
        let new_len = self.len + s.len();
        if new_len > self.data.capacity() {
            self.resize(new_len)?;
        }
        self.data[self.len..new_len].copy_from_slice(s.as_bytes());
        self.len = new_len;
        Ok(())
    }

    /// Appends formatted text to the buffer
    pub fn appendf(&mut self, args: fmt::Arguments) -> Result<(), String> {
        self.append(&format!("{}", args))
    }

    /// Prepends a string slice to the buffer
    pub fn prepend(&mut self, s: &str) -> Result<(), String> {
        let new_len = self.len + s.len();
        if new_len > self.data.capacity() {
            self.resize(new_len)?;
        }
        self.data.copy_within(0..self.len, s.len());
        self.data[..s.len()].copy_from_slice(s.as_bytes());
        self.len = new_len;
        Ok(())
    }

    /// Creates a new buffer from a slice of this buffer
    pub fn slice(&self, start: usize, end: isize) -> Result<Self, String> {
        let end = if end < 0 {
            self.len as isize + end
        } else {
            end
        } as usize;

        if start > end || end > self.len {
            return Err("Invalid slice range".to_string());
        }

        let mut new_buf = Self::with_capacity(end - start);
        new_buf.data[..end-start].copy_from_slice(&self.data[start..end]);
        new_buf.len = end - start;
        Ok(new_buf)
    }

    /// Returns the index of the first occurrence of the pattern
    pub fn index_of(&self, pattern: &str) -> Option<usize> {
        self.as_str().ok()?.find(pattern)
    }

    /// Checks if this buffer equals another buffer
    pub fn equals(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        &self.data[..self.len] == &other.data[..other.len]
    }

    /// Fills the buffer with the specified byte
    pub fn fill(&mut self, byte: u8) {
        self.data[..self.len].fill(byte);
    }

    /// Trims whitespace from the left side of the buffer
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

    /// Trims whitespace from the right side of the buffer
    pub fn trim_right(&mut self) {
        while self.len > 0 && self.data[self.len-1].is_ascii_whitespace() {
            self.len -= 1;
        }
    }

    /// Trims whitespace from both sides of the buffer
    pub fn trim(&mut self) {
        self.trim_left();
        self.trim_right();
    }

    /// Prints the buffer content as hexadecimal
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

    /// Computes the nearest multiple of 1024 from a number
    fn nearest_multiple_of_1024(n: usize) -> usize {
        (n + 1023) & !1023
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

impl fmt::Display for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.as_str() {
            Ok(s) => write!(f, "{}", s),
            Err(_) => write!(f, "<invalid utf8>"),
        }
    }
}