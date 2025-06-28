// buffer.rs

// Define the default buffer size
const BUFFER_DEFAULT_SIZE: usize = 64;

// Define the Buffer struct
pub struct Buffer {
    len: usize,
    alloc: Vec<u8>,
    pub data: Vec<u8>, // Make data field public
}

impl Buffer {
    // Create a new buffer with the default size
    pub fn new() -> Self {
        Self::new_with_size(BUFFER_DEFAULT_SIZE)
    }

    // Create a new buffer with a specified size
    pub fn new_with_size(size: usize) -> Self {
        let alloc = vec![0; size + 1];
        Buffer {
            len: size,
            alloc: alloc.clone(),
            data: alloc,
        }
    }

    // Create a new buffer with a given string
    pub fn new_with_string(s: &str) -> Self {
        Self::new_with_string_length(s, s.len())
    }

    // Create a new buffer with a given string and length
    pub fn new_with_string_length(s: &str, len: usize) -> Self {
        let mut alloc = vec![0; len + 1];
        alloc[..len].copy_from_slice(s.as_bytes());
        Buffer {
            len,
            alloc: alloc.clone(),
            data: alloc,
        }
    }

    // Create a new buffer with a copy of a given string
    pub fn new_with_copy(s: &str) -> Self {
        let len = s.len();
        let mut buffer = Self::new_with_size(len);
        buffer.alloc[..len].copy_from_slice(s.as_bytes());
        buffer.data = buffer.alloc.clone();
        buffer
    }

    // Get the size of the buffer
    pub fn size(&self) -> usize {
        self.len
    }

    // Get the length of the string in the buffer
    pub fn length(&self) -> usize {
        self.data.iter().position(|&c| c == 0).unwrap_or(self.len)
    }

    // Free the buffer (handled automatically in Rust)
    pub fn free(self) {}

    // Resize the buffer to hold `n` bytes
    pub fn resize(&mut self, n: usize) -> Result<(), &'static str> {
        let new_size = nearest_multiple_of(1024, n);
        self.len = new_size;
        self.alloc.resize(new_size + 1, 0);
        self.data = self.alloc.clone();
        Ok(())
    }

    // Append a formatted string to the buffer
    pub fn appendf(&mut self, args: std::fmt::Arguments) -> Result<(), &'static str> {
        use std::fmt::Write;
        let mut formatted_string = String::new();
        formatted_string.write_fmt(args).map_err(|_| "Formatting error")?;
        self.append(&formatted_string)
    }

    // Append a string to the buffer
    pub fn append(&mut self, s: &str) -> Result<(), &'static str> {
        self.append_n(s, s.len())
    }

    // Append the first `len` bytes of a string to the buffer
    pub fn append_n(&mut self, s: &str, len: usize) -> Result<(), &'static str> {
        let prev_len = self.length();
        let needed = len + prev_len;

        if self.len > needed {
            self.data[prev_len..needed].copy_from_slice(&s.as_bytes()[..len]);
            Ok(())
        } else {
            Err("Not enough space")
        }
    }

    // Compact the buffer
    pub fn compact(&mut self) -> Result<usize, &'static str> {
        let len = self.length();
        let rem = self.len - len;
        self.alloc.truncate(len + 1);
        self.data = self.alloc.clone();
        self.len = len;
        Ok(rem)
    }

    // Print the buffer (for debugging purposes)
    pub fn print(&self) {
        println!("{}", String::from_utf8_lossy(&self.data));
    }

    // Clear the buffer
    pub fn clear(&mut self) {
        self.data.clear();
        self.alloc.clear();
        self.len = 0;
    }

    // Fill the buffer with a specific byte
    pub fn fill(&mut self, byte: u8) {
        for i in 0..self.len {
            self.data[i] = byte;
        }
    }

    // Trim whitespace from the left
    pub fn trim_left(&mut self) {
        let trimmed = self.data.iter().skip_while(|&&c| c.is_ascii_whitespace()).cloned().collect::<Vec<u8>>();
        self.data = trimmed.clone();
        self.alloc = trimmed;
    }

    // Trim whitespace from the right
    pub fn trim_right(&mut self) {
        let trimmed = self.data.iter().rev().skip_while(|&&c| c.is_ascii_whitespace()).cloned().collect::<Vec<u8>>();
        self.data = trimmed.iter().rev().cloned().collect();
        self.alloc = self.data.clone();
    }

    // Trim whitespace from both ends
    pub fn trim(&mut self) {
        self.trim_left();
        self.trim_right();
    }
}

// Helper function to compute the nearest multiple of `a` from `b`
fn nearest_multiple_of(a: usize, b: usize) -> usize {
    (b + (a - 1)) & !(a - 1)
}