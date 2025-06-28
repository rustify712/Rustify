use std::fmt;
use std::fmt::Write;

/// 默认缓冲区大小
const BUFFER_DEFAULT_SIZE: usize = 64;

/// Buffer 结构体，封装了一个动态字节缓冲区
pub struct Buffer {
    data: Vec<u8>,
}

impl Buffer {
    /// 创建一个新的 Buffer，使用默认大小
    pub fn new() -> Self {
        Buffer::with_size(BUFFER_DEFAULT_SIZE)
    }

    /// 使用指定大小创建一个新的 Buffer
    pub fn with_size(n: usize) -> Self {
        let size = Self::nearest_multiple_of(1024, n);
        let mut data = Vec::with_capacity(size);
        data.resize(size, 0);
        Buffer { data }
    }

    /// 返回缓冲区的分配大小
    pub fn size(&self) -> usize {
        self.data.capacity()
    }

    /// 返回缓冲区中当前使用的长度
    pub fn length(&self) -> usize {
        self.data.iter().position(|&x| x == 0).unwrap_or(self.data.len())
    }

    /// 清空缓冲区，填充为 0
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// 填充缓冲区为指定的字节
    pub fn fill(&mut self, byte: u8) {
        self.data.fill(byte);
    }

    /// 将缓冲区转换为 String（假设为有效的 UTF-8）
    pub fn to_string(&self) -> String {
        String::from_utf8_lossy(&self.data[..self.length()]).to_string()
    }

    /// 查找最近的 1024 的倍数
    fn nearest_multiple_of(a: usize, b: usize) -> usize {
        ((b + (a - 1)) / a) * a
    }
}

impl Buffer {
    /// 使用 &str 创建一个新的 Buffer（复制数据）
    pub fn from_str(s: &str) -> Self {
        let mut buffer = Buffer::new();
        buffer.append(s).expect("Failed to append string");
        buffer
    }

    /// 使用字节切片创建一个新的 Buffer（复制数据）
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut buffer = Buffer::new();
        buffer.append_n(bytes).expect("Failed to append bytes");
        buffer
    }
}

impl Buffer {
    /// 调整缓冲区的大小到至少 n 字节，向上取整到最近的 1024 的倍数
    pub fn resize(&mut self, n: usize) -> Result<(), ()> {
        let new_size = Self::nearest_multiple_of(1024, n);
        self.data.reserve(new_size - self.data.capacity());
        self.data.resize(new_size, 0);
        Ok(())
    }

    /// 压缩缓冲区，移除多余的空间
    pub fn compact(&mut self) -> Result<isize, ()> {
        let len = self.length();
        let rem = self.size().saturating_sub(len);
        self.data.truncate(len);
        self.data.shrink_to_fit();
        Ok(rem as isize)
    }
}

impl Buffer {
    /// 追加一个字符串到缓冲区
    pub fn append(&mut self, s: &str) -> Result<(), ()> {
        self.append_n(s.as_bytes())
    }

    /// 追加指定长度的字节到缓冲区
    pub fn append_n(&mut self, bytes: &[u8]) -> Result<(), ()> {
        let needed = self.length().saturating_add(bytes.len());
        if self.size() < needed {
            self.resize(needed)?;
        }
        self.data.splice(self.length()..self.length(), bytes.iter().cloned());
        Ok(())
    }

    /// 预先分配足够的空间，然后在开头插入字节
    pub fn prepend(&mut self, s: &str) -> Result<(), ()> {
        let bytes = s.as_bytes();
        let len = bytes.len();
        let needed = self.length().saturating_add(len);
        if self.size() < needed {
            self.resize(needed)?;
        }
        self.data.splice(0..0, bytes.iter().cloned());
        Ok(())
    }
}

impl Buffer {
    /// 追加格式化字符串，类似于 `printf` 风格
    pub fn appendf(&mut self, args: fmt::Arguments) -> Result<(), ()> {
        let mut formatted = String::new();
        formatted.write_fmt(args).map_err(|_| ())?;
        self.append(&formatted)
    }
}

impl Buffer {
    /// 创建一个基于 `from..to` 范围的新 Buffer
    pub fn slice(&self, from: usize, to: isize) -> Option<Buffer> {
        let len = self.length();
        let to = if to < 0 {
            len.checked_sub(to.unsigned_abs())?
        } else {
            to as usize
        };
        if to < from || from > len {
            return None;
        }
        let to = std::cmp::min(to, len);
        Some(Buffer {
            data: self.data[from..to].to_vec(),
        })
    }

    /// 比较两个 Buffer 是否相等
    pub fn equals(&self, other: &Buffer) -> bool {
        self.data[..self.length()] == other.data[..other.length()]
    }
}

impl Buffer {
    /// 查找子字符串的索引
    pub fn index_of(&self, pattern: &str) -> isize {
        if let Some(pos) = self.to_string().find(pattern) {
            pos as isize
        } else {
            -1
        }
    }
}

impl Buffer {
    /// 修剪左侧的空白字符
    pub fn trim_left(&mut self) {
        self.data = self.to_string().trim_start().as_bytes().to_vec();
        self.data.resize(self.size(), 0);
    }

    /// 修剪右侧的空白字符
    pub fn trim_right(&mut self) {
        self.data = self.to_string().trim_end().as_bytes().to_vec();
        self.data.resize(self.size(), 0);
    }

    /// 修剪两侧的空白字符
    pub fn trim(&mut self) {
        self.data = self.to_string().trim().as_bytes().to_vec();
        self.data.resize(self.size(), 0);
    }
}

impl Buffer {
    /// 打印缓冲区的十六进制转储
    pub fn print_hex_dump(&self) {
        println!("\n ");
        for (i, byte) in self.data.iter().enumerate().take(self.length()) {
            print!(" {:02x}", byte);
            if (i + 1) % 8 == 0 {
                println!();
                print!(" ");
            }
        }
        println!();
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.to_string();
        f.debug_struct("Buffer")
            .field("data", &s)
            .field("size", &self.size())
            .field("length", &self.length())
            .finish()
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

impl Eq for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        // Rust 会自动处理内存释放
        // 如果需要自定义释放逻辑，可以在这里实现
    }
}
