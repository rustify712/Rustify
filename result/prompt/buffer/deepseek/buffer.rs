use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use std::mem;
use std::cmp;

const BUFFER_DEFAULT_SIZE: usize = 64;

/// Buffer struct.
struct Buffer {
    len: usize,
    alloc: Vec<u8>,
    data: *mut u8,
}

impl Buffer {
    /// Compute the nearest multiple of `a` from `b`.
    fn nearest_multiple_of(a: usize, b: usize) -> usize {
        (b + (a - 1)) & !(a - 1)
    }

    /// Allocate a new buffer with BUFFER_DEFAULT_SIZE.
    fn new() -> Option<Self> {
        Self::new_with_size(BUFFER_DEFAULT_SIZE)
    }

    /// Allocate a new buffer with `n` bytes.
    fn new_with_size(n: usize) -> Option<Self> {
        let mut alloc = vec![0; n + 1];
        let data = alloc.as_mut_ptr();
        Some(Buffer {
            len: n,
            alloc,
            data,
        })
    }

    /// Allocate a new buffer with `str`.
    fn new_with_string(str: &str) -> Option<Self> {
        Self::new_with_string_length(str, str.len())
    }

    /// Allocate a new buffer with `str` and `len`.
    fn new_with_string_length(str: &str, len: usize) -> Option<Self> {
        let mut alloc = Vec::with_capacity(len + 1);
        alloc.extend_from_slice(str.as_bytes());
        alloc.push(0); // Null-terminator
        let data = alloc.as_mut_ptr();
        Some(Buffer {
            len,
            alloc,
            data,
        })
    }

    /// Allocate a new buffer with a copy of `str`.
    fn new_with_copy(str: &str) -> Option<Self> {
        let len = str.len();
        let mut buffer = Self::new_with_size(len)?;
        unsafe {
            ptr::copy_nonoverlapping(str.as_ptr(), buffer.data, len);
        }
        Some(buffer)
    }

    /// Deallocate excess memory, the number of bytes removed or -1.
    fn compact(&mut self) -> isize {
        let len = self.length();
        let rem = self.len - len;
        let mut new_alloc = vec![0; len + 1];
        unsafe {
            ptr::copy_nonoverlapping(self.data, new_alloc.as_mut_ptr(), len);
        }
        self.alloc = new_alloc;
        self.len = len;
        self.data = self.alloc.as_mut_ptr();
        rem as isize
    }

    /// Return buffer size.
    fn size(&self) -> usize {
        self.len
    }

    /// Return string length.
    fn length(&self) -> usize {
        unsafe { CStr::from_ptr(self.data as *const c_char).to_bytes().len() }
    }

    /// Resize to hold `n` bytes.
    fn resize(&mut self, n: usize) -> Result<(), ()> {
        let new_size = Self::nearest_multiple_of(1024, n);
        self.alloc.resize(new_size + 1, 0);
        self.len = new_size;
        self.data = self.alloc.as_mut_ptr();
        Ok(())
    }

    /// Append a printf-style formatted string to the buffer.
    fn appendf(&mut self, format: &str, args: std::fmt::Arguments) -> Result<(), ()> {
        let length = self.length();
        let required = format.len(); // Simplified for demonstration
        self.resize(length + required)?;
        unsafe {
            let dst = self.data.add(length);
            let bytes_written = vsnprintf(dst as *mut c_char, required + 1, format.as_ptr() as *const c_char, args);
            if bytes_written < 0 {
                return Err(());
            }
        }
        Ok(())
    }

    /// Append `str` to `self` and return 0 on success, -1 on failure.
    fn append(&mut self, str: &str) -> Result<(), ()> {
        self.append_n(str, str.len())
    }

    /// Append the first `len` bytes from `str` to `self` and return 0 on success, -1 on failure.
    fn append_n(&mut self, str: &str, len: usize) -> Result<(), ()> {
        let prev = self.length();
        let needed = len + prev;
        if self.len <= needed {
            self.resize(needed)?;
        }
        unsafe {
            ptr::copy_nonoverlapping(str.as_ptr(), self.data.add(prev), len);
        }
        Ok(())
    }

    /// Prepend `str` to `self` and return 0 on success, -1 on failure.
    fn prepend(&mut self, str: &str) -> Result<(), ()> {
        let len = str.len();
        let prev = self.length();
        let needed = len + prev;
        if self.len <= needed {
            self.resize(needed)?;
        }
        unsafe {
            let dst = self.data.add(len);
            ptr::copy(self.data, dst, prev);
            ptr::copy_nonoverlapping(str.as_ptr(), self.data, len);
        }
        Ok(())
    }

    /// Return a new buffer based on the `from..to` slice of `buf`, or None on error.
    fn slice(&self, from: usize, to: isize) -> Option<Self> {
        let len = self.length();
        let to = if to < 0 {
            len - (-to) as usize
        } else {
            cmp::min(to as usize, len)
        };
        if to < from {
            return None;
        }
        let n = to - from;
        let mut new_buffer = Self::new_with_size(n)?;
        unsafe {
            ptr::copy_nonoverlapping(self.data.add(from), new_buffer.data, n);
        }
        Some(new_buffer)
    }

    /// Return true if the buffers contain equivalent data.
    fn equals(&self, other: &Self) -> bool {
        unsafe {
            CStr::from_ptr(self.data as *const c_char) == CStr::from_ptr(other.data as *const c_char)
        }
    }

    /// Return the index of the substring `str`, or -1 on failure.
    fn indexof(&self, str: &str) -> isize {
        unsafe {
            let cstr = CStr::from_ptr(self.data as *const c_char);
            let haystack = cstr.to_bytes();
            let needle = str.as_bytes();
            haystack.windows(needle.len()).position(|window| window == needle).map(|pos| pos as isize).unwrap_or(-1)
        }
    }

    /// Trim leading whitespace.
    fn trim_left(&mut self) {
        unsafe {
            while *self.data != 0 && (*self.data as char).is_whitespace() {
                self.data = self.data.add(1);
            }
        }
    }

    /// Trim trailing whitespace.
    fn trim_right(&mut self) {
        let len = self.length();
        unsafe {
            let mut i = len - 1;
            while i >= 0 && (*(self.data.add(i)) as char).is_whitespace() {
                *(self.data.add(i)) = 0;
                i -= 1;
            }
        }
    }

    /// Trim trailing and leading whitespace.
    fn trim(&mut self) {
        self.trim_left();
        self.trim_right();
    }

    /// Fill the buffer with `c`.
    fn fill(&mut self, c: u8) {
        unsafe {
            ptr::write_bytes(self.data, c, self.len);
        }
    }

    /// Fill the buffer with 0.
    fn clear(&mut self) {
        self.fill(0);
    }

    /// Print a hex dump of the buffer.
    fn print(&self) {
        println!("\n ");
        for i in 0..self.len {
            print!(" {:02x}", unsafe { *self.data.add(i) });
            if (i + 1) % 8 == 0 {
                println!("\n ");
            }
        }
        println!();
    }
}

// Placeholder for vsnprintf, which is not available in Rust's standard library.
// This would need to be implemented or replaced with a suitable alternative.
unsafe fn vsnprintf(dst: *mut c_char, size: usize, format: *const c_char, args: std::fmt::Arguments) -> isize {
    // Implementation would depend on the specific requirements and environment.
    // For now, this is a placeholder.
    0
}