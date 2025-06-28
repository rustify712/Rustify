use std::fmt::{self, Write};
use std::mem;
use std::ptr;
use std::str;
use std::ffi::CStr;
use std::cmp::min;

const BUFFER_DEFAULT_SIZE: usize = 64; // Equivalent to #define BUFFER_DEFAULT_SIZE 64

// Struct for the buffer.
pub struct Buffer {
    len: usize,
    alloc: Vec<u8>,
    data: String,
}

impl Buffer {
    // Compute the nearest multiple of `a` from `b`.
    fn nearest_multiple_of(a: usize, b: usize) -> usize {
        (b + (a - 1)) & !(a - 1)
    }

    // Allocate a new buffer with BUFFER_DEFAULT_SIZE.
    pub fn new() -> Self {
        Self::new_with_size(BUFFER_DEFAULT_SIZE)
    }

    // Allocate a new buffer with `n` bytes.
    pub fn new_with_size(n: usize) -> Self {
        let mut alloc = vec![0u8; n];
        alloc.push(0); // Null-terminate the buffer as in C/C++
        let data = String::new();
        Buffer { len: n, alloc, data }
    }

    // Allocate a new buffer with `str`.
    pub fn new_with_string(s: String) -> Self {
        Self::new_with_string_length(s.as_str(), s.len())
    }

    // Allocate a new buffer with `str` and `len`.
    pub fn new_with_string_length(s: &str, len: usize) -> Self {
        let mut alloc = s.as_bytes().to_vec();
        alloc.push(0); // Null-terminate the buffer as in C/C++
        let data = String::from(s);
        Buffer { len, alloc, data }
    }

    // Allocate a new buffer with a copy of `str`.
    pub fn new_with_copy(s: &str) -> Self {
        let mut buffer = Self::new_with_size(s.len());
        buffer.alloc.copy_from_slice(s.as_bytes());
        buffer.data = s.to_string();
        buffer
    }

    // Deallocate excess memory, return the number of bytes removed or -1.
    pub fn compact(&mut self) -> isize {
        let len = self.len();
        let rem = self.len - len;
        let mut buf = vec![0u8; len];
        buf.copy_from_slice(&self.data.as_bytes()[..len]);
        self.alloc = buf;
        self.len = len;
        self.data = String::from_utf8_lossy(&self.alloc).to_string();
        rem as isize
    }

    // Free the buffer.
    pub fn free(self) {
        // In Rust, the buffer will automatically be freed when it goes out of scope.
    }

    // Return buffer size.
    pub fn size(&self) -> usize {
        self.len
    }

    // Return string length.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    // Resize to hold `n` bytes.
    pub fn resize(&mut self, n: usize) -> Result<(), ()> {
        let n = Self::nearest_multiple_of(1024, n);
        self.len = n;
        if n + 1 > self.alloc.capacity() {
            self.alloc.resize(n + 1, 0); // Resize the allocation
        }
        self.alloc[n] = 0; // Null-terminate as in C/C++
        self.data = String::from_utf8_lossy(&self.alloc).to_string();
        Ok(())
    }

    // Append a printf-style formatted string to the buffer.
    pub fn appendf(&mut self, format: fmt::Arguments) -> Result<(), ()> {
        let length = self.len();
        let required = format.arg_count();
        if self.resize(length + required).is_err() {
            return Err(());
        }
        write!(self.data, "{}", format)?;
        Ok(())
    }

    // Append `str` to `self` and return 0 on success, -1 on failure.
    pub fn append(&mut self, s: &str) -> Result<(), ()> {
        self.append_n(s, s.len())
    }

    // Append the first `len` bytes from `str` to `self` and return 0 on success, -1 on failure.
    pub fn append_n(&mut self, s: &str, len: usize) -> Result<(), ()> {
        let prev_len = self.len();
        let needed = len + prev_len;
        if self.len > needed {
            self.data.push_str(s);
            return Ok(());
        }

        // Resize and then append
        self.resize(needed)?;
        self.data.push_str(s);
        Ok(())
    }

    // Prepend `str` to `self` and return 0 on success, -1 on failure.
    pub fn prepend(&mut self, s: &str) -> Result<(), ()> {
        let len = s.len();
        let prev_len = self.len();
        let needed = len + prev_len;

        if self.len > needed {
            self.data.insert_str(0, s);
            return Ok(());
        }

        self.resize(needed)?;
        self.data.insert_str(0, s);
        Ok(())
    }

    // Return a new buffer based on the `from..to` slice of `buf`, or None on error.
    pub fn slice(&self, from: usize, to: isize) -> Option<Self> {
        let len = self.len();
        if to < from as isize {
            return None;
        }
        let to = if to < 0 {
            len - to.abs() as usize
        } else {
            min(to as usize, len)
        };
        let n = to - from;
        let buf = self.data[from..to].to_string();
        Some(Self::new_with_string(buf))
    }

    // Return true if the buffers contain equivalent data.
    pub fn equals(&self, other: &Self) -> bool {
        self.data == other.data
    }

    // Return the index of the substring `str`, or -1 on failure.
    pub fn indexof(&self, s: &str) -> isize {
        self.data.find(s).map_or(-1, |idx| idx as isize)
    }

    // Trim leading whitespace.
    pub fn trim_left(&mut self) {
        self.data = self.data.trim_start().to_string();
    }

    // Trim trailing whitespace.
    pub fn trim_right(&mut self) {
        self.data = self.data.trim_end().to_string();
    }

    // Trim trailing and leading whitespace.
    pub fn trim(&mut self) {
        self.trim_left();
        self.trim_right();
    }

    // Fill the buffer with `c`.
    pub fn fill(&mut self, c: u8) {
        self.alloc.fill(c);
    }

    // Fill the buffer with 0.
    pub fn clear(&mut self) {
        self.fill(0);
    }

    // Print a hex dump of the buffer.
    pub fn print(&self) {
        let len = self.len();
        print!("\n ");
        for (i, &byte) in self.alloc.iter().enumerate() {
            print!(" {:02x}", byte);
            if (i + 1) % 8 == 0 {
                print!("\n ");
            }
        }
        println!();
    }
}

