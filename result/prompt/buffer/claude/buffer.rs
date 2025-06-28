//! Buffer implementation providing string manipulation functionality.

/// Default size for new buffers
const BUFFER_DEFAULT_SIZE: usize = 64;

/// A buffer structure for string manipulation
#[derive(Debug)]
pub struct Buffer {
    // Length of allocated memory
    len: usize,
    // Actual string data
    data: String,
}

impl Buffer {
    /// Create a new buffer with default size
    pub fn new() -> Self {
        Self::with_capacity(BUFFER_DEFAULT_SIZE)
    }

    /// Create a new buffer with specified capacity
    pub fn with_capacity(size: usize) -> Self {
        let mut data = String::with_capacity(size);
        // Initialize with null bytes to match C behavior
        data.extend(std::iter::repeat('\0').take(size));
        Self {
            len: size,
            data,
        }
    }

    /// Create a new buffer from an existing string
    pub fn from_string(s: String) -> Self {
        let len = s.len();
        Self {
            len,
            data: s,
        }
    }

    /// Create a new buffer with a copy of the provided string
    pub fn from_str(s: &str) -> Self {
        Self::from_string(s.to_string())
    }

    /// Compact the buffer by removing excess capacity
    pub fn compact(&mut self) -> Result<usize, std::io::Error> {
        let actual_len = self.length();
        let removed = self.len - actual_len;

        // Create new string with exact capacity
        let mut new_data = String::with_capacity(actual_len);
        new_data.push_str(&self.data[..actual_len]);

        self.len = actual_len;
        self.data = new_data;

        Ok(removed)
    }

    /// Return buffer capacity
    pub fn capacity(&self) -> usize {
        self.len
    }

    /// Return string length
    pub fn length(&self) -> usize {
        self.data.trim_end_matches('\0').len()
    }

    /// Resize buffer to hold n bytes
    pub fn resize(&mut self, n: usize) -> Result<(), std::io::Error> {
        // Calculate nearest multiple of 1024
        let n = (n + 1023) & !1023;

        self.data.resize(n, '\0');
        self.len = n;

        Ok(())
    }

    /// Append formatted string to buffer
    pub fn append_fmt(&mut self, args: std::fmt::Arguments<'_>) -> Result<(), std::io::Error> {
        use std::fmt::Write;
        write!(self.data, "{}", args)?;
        Ok(())
    }

    /// Append string to buffer
    pub fn append(&mut self, s: &str) -> Result<(), std::io::Error> {
        self.append_n(s, s.len())
    }

    /// Append n bytes of string to buffer
    pub fn append_n(&mut self, s: &str, n: usize) -> Result<(), std::io::Error> {
        let prev_len = self.length();
        let needed = n + prev_len;

        if self.len <= needed {
            self.resize(needed)?;
        }

        self.data.replace_range(prev_len..prev_len+n, &s[..n]);
        Ok(())
    }

    /// Prepend string to buffer
    pub fn prepend(&mut self, s: &str) -> Result<(), std::io::Error> {
        let needed = s.len() + self.length();

        if self.len <= needed {
            self.resize(needed)?;
        }

        // Create new string with prepended content
        let mut new_data = String::with_capacity(self.len);
        new_data.push_str(s);
        new_data.push_str(&self.data);
        self.data = new_data;

        Ok(())
    }

    /// Create a new buffer from a slice of this buffer
    pub fn slice(&self, from: usize, to: isize) -> Option<Self> {
        let len = self.length();

        // Convert negative index to positive
        let to = if to < 0 { len - (!to as usize) } else { to as usize };

        // Validate range
        if to < from || to > len {
            return None;
        }

        let slice = &self.data[from..to];
        Some(Self::from_str(slice))
    }

    /// Check if buffers contain equivalent data
    pub fn equals(&self, other: &Self) -> bool {
        self.data == other.data
    }

    /// Find index of substring
    pub fn index_of(&self, s: &str) -> Option<usize> {
        self.data.find(s)
    }

    /// Trim leading whitespace
    pub fn trim_left(&mut self) {
        let trimmed = self.data.trim_start().to_string();
        self.data = trimmed;
    }

    /// Trim trailing whitespace
    pub fn trim_right(&mut self) {
        let trimmed = self.data.trim_end().to_string();
        self.data = trimmed;
    }

    /// Trim both leading and trailing whitespace
    pub fn trim(&mut self) {
        let trimmed = self.data.trim().to_string();
        self.data = trimmed;
    }

    /// Fill buffer with specified character
    pub fn fill(&mut self, c: char) {
        self.data = std::iter::repeat(c).take(self.len).collect();
    }

    /// Clear buffer (fill with null bytes)
    pub fn clear(&mut self) {
        self.fill('\0');
    }

    /// Print hex dump of buffer contents
    pub fn print_hex(&self) {
        println!();
        print!(" ");

        // Print hex representation
        for (i, byte) in self.data.as_bytes().iter().enumerate() {
            print!(" {:02x}", byte);
            if (i + 1) % 8 == 0 {
                println!();
                print!(" ");
            }
        }
        println!();
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // Rust handles cleanup automatically
    }
}