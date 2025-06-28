/// The version of the Binn library.
/// This constant defines the version of the Binn library as "3.0.0".
pub const BINN_VERSION: &str = "3.0.0";

/// A constant representing no bytes storage in Binn library.
/// This is used to indicate that no data is stored in certain data structures or operations.
pub const BINN_STORAGE_NOBYTES: u8 = 0x00;

/// Constant representing the storage type for a 64-bit integer (QWORD) in Binn.
/// This is used to identify and handle 64-bit integers during serialization and deserialization.
pub const BINN_STORAGE_QWORD: u8 = 0x80;

/// Constant representing the storage type for a word in Binn binary data.
/// This is used to identify and handle word-sized data in the Binn library.
const BINN_STORAGE_WORD: u8 = 0x40;

/// Constant representing the storage format for BLOB (Binary Large Object) type in Binn library.
/// This constant is used to identify and handle BLOB data in binary format.
/// Value: `0xC0`
pub const BINN_STORAGE_BLOB: u8 = 0xC0;

/// Forces the compiler to always inline the function.
/// This is used to optimize performance by reducing function call overhead.
#[inline(always)]
fn always_inline_function() {
    // Function implementation goes here
}

/// Constant representing the container type in Binn library.
pub const BINN_STORAGE_CONTAINER: u8 = 0xE0;

/// 表示无效的 Binn 数据结构，通常用于标识一个无效的二进制数据对象或操作失败的情况。
pub const INVALID_BINN: u32 = 0;

/// Constant representing the storage type for byte data in Binn library.
/// This constant is used to identify and handle byte type data during serialization and deserialization.
const BINN_STORAGE_BYTE: u8 = 0x20;

/// Constant representing the storage type for a 32-bit unsigned integer (DWORD).
pub const BINN_STORAGE_DWORD: u8 = 0x60;

/// Constant representing the storage format for string data in Binn.
/// This value is used to identify and handle string data during serialization and deserialization.
pub const BINN_STORAGE_STRING: u8 = 0xA0;

/// Constant representing the maximum storage capacity in Binn binary data.
/// This is used to limit or identify the storage capacity of container type data.
pub const BINN_STORAGE_MAX: u8 = BINN_STORAGE_CONTAINER;

/// 常量 `BINN_LIST` 定义了 Binn 库中列表数据类型的标识符，其值为 `0xE0`。
/// 该常量用于在二进制数据中标记列表类型，帮助库在序列化和反序列化过程中识别和处理列表数据。
pub const BINN_LIST: u8 = 0xE0;

/// A constant representing the maximum value mask in Binn library.
/// This mask is used to limit the maximum value range in binary data processing,
/// ensuring data validity and consistency.
pub const BINN_MAX_VALUE_MASK: u32 = 0xFFFFF;

/// Constant representing the map type in Binn library.
/// This is used to identify and handle map data in binary format.
/// Value: `0xE1`
pub const BINN_MAP: u8 = 0xE1;

/// 表示空值（NULL）的二进制标识符，其值为 `0x00`。
/// 该常量用于在二进制数据中标记空值，帮助开发者在创建或解析二进制数据结构时明确区分空值与其他数据类型。
pub const BINN_NULL: u8 = 0x00;

/// A 16-bit mask value used to extract or manipulate type information in binary data.
/// This mask helps the library distinguish and identify different data types (e.g., integers, strings)
/// during binary data processing, ensuring correct parsing and manipulation.
pub const BINN_TYPE_MASK16: u16 = 0x0FFF;

/// Constant representing the type identifier for an 8-bit integer in Binn library.
/// This is used to mark and identify 8-bit integer types in binary data.
/// Value: `0x21`
pub const BINN_INT8: u8 = 0x21;

/// A constant representing a 16-bit mask for extracting storage type information in Binn library.
/// This mask is used to identify and manipulate the storage format of binary data, ensuring correct parsing and operation.
pub const BINN_STORAGE_MASK16: u16 = 0xE000;

/// A constant representing a flag indicating whether there are more data blocks in the binary data.
/// This flag is used to determine if the current data block is part of a larger data structure.
pub const BINN_STORAGE_HAS_MORE: u8 = 0x10;

/// A constant representing the mask value `0x0F` used to extract the type information from binary data.
/// This mask is used in the Binn library to identify and handle different data types during serialization and deserialization.
pub const BINN_TYPE_MASK: u8 = 0x0F;

/// A constant representing the virtual storage type in Binn library.
/// This flag is used to identify virtual storage types in binary data processing.
pub const BINN_STORAGE_VIRTUAL: u32 = 0x80000;

/// A mask value `0xE0` used to extract the storage type information from binary data in the Binn library.
/// This mask helps the library quickly identify the storage format of the data (e.g., integers, strings, etc.),
/// ensuring correct parsing and exchange of data.
pub const BINN_STORAGE_MASK: u8 = 0xE0;

/// 常量表示二进制数据中的布尔值 `false`，其值为 `0x02`。
/// 该常量用于在 Binn 库中标识二进制数据中的 `false` 值，帮助库在序列化和反序列化过程中正确识别和处理布尔类型数据。
pub const BINN_FALSE: u8 = 0x02;

/// Constant representing the binary value for `true` in Binn library.
/// This constant is used to identify and handle boolean `true` values during serialization and deserialization.
pub const BINN_TRUE: u8 = 0x01;

/// Constant representing the binary object type in Binn library.
/// This is used to identify and handle object data in binary format.
/// Value: `0xE2`
pub const BINN_OBJECT: u8 = 0xE2;

/// The minimum number of bytes required for storage in the Binn library.
/// This constant is used to ensure compact and efficient storage of binary data.
pub const BINN_STORAGE_MIN: u8 = BINN_STORAGE_NOBYTES;

/// Constant representing the storage type for a 16-bit integer in Binn.
/// This constant is used to identify and handle 16-bit integers during serialization and deserialization.
pub const BINN_INT16: u8 = 0x41;

/// 无符号8位整数（即单字节整数）的类型标识符。
/// 该常量用于在 Binn 库中标识和区分不同的数据类型，使得库能够正确处理和存储二进制数据中的无符号8位整数。
pub const BINN_UINT8: u8 = 0x20;

/// Constant representing the storage type for a 16-bit unsigned integer in Binn.
/// This is used to identify and handle 16-bit unsigned integers during serialization and deserialization.
pub const BINN_UINT16: u8 = 0x40;

/// Constant representing the storage type for a 32-bit unsigned integer (DWORD).
/// This constant is used to identify and handle 32-bit unsigned integers during serialization and deserialization.
pub const BINN_UINT32: u8 = 0x60;

/// Constant representing the storage type for a 64-bit unsigned integer (QWORD) in Binn.
/// This constant is used to identify and handle 64-bit unsigned integers during serialization and deserialization.
pub const BINN_UINT64: u8 = 0x80;

/// Constant representing the storage type for a 32-bit integer in Binn.
/// This constant is used to identify and handle 32-bit integers during serialization and deserialization.
pub const BINN_INT32: u8 = 0x61;

/// Constant representing the storage type for a 64-bit integer in Binn library.
/// This constant is used to identify and handle 64-bit integers during serialization and deserialization.
pub const BINN_INT64: u8 = 0x81;

/// Constant representing the decimal type in Binn library.
/// This constant is used to identify and handle decimal data during serialization and deserialization.
pub const BINN_DECIMAL: u8 = 0xA4;

/// 常量 `BINN_DATE` 定义了 Binn 库中日期类型的标识符，其值为 `0xA2`。
/// 该常量用于在二进制数据中标记日期类型，帮助库在序列化和反序列化过程中识别和处理日期数据。
pub const BINN_DATE: u8 = 0xA2;

/// 常量 `BINN_TIME` 定义了 Binn 库中时间类型的标识符，其值为 `0xA3`。
/// 该常量用于在二进制数据中标记时间类型，帮助库在序列化和反序列化过程中识别和处理时间数据。
pub const BINN_TIME: u8 = 0xA3;

/// 常量 `BINN_SCHAR` 定义了 Binn 库中用于表示有符号 8 位整数的类型标识符，其值为 `0x21`。
/// 这个常量用于简化代码中对有符号字符类型的使用，使得开发者可以更直观地处理有符号字符数据。
pub const BINN_SCHAR: u8 = BINN_INT8;

/// 表示无符号8位整数（即无符号字符）的类型别名。
/// 该类型别名用于简化代码中对无符号字符类型的使用，使得开发者可以通过 `BINN_UCHAR` 来声明或处理8位无符号整数数据。
/// 在 Binn 库中，这种类型通常用于处理二进制数据中的单个字节，确保数据在不同系统间的兼容性和高效传输。
pub type BINN_UCHAR = u8;

/// Constant representing the currency string type in Binn library.
/// This constant is used to identify and handle currency string data during serialization and deserialization.
pub const BINN_CURRENCYSTR: u8 = 0xA5;

/// 常量 `BINN_DATETIME` 定义了 Binn 库中日期时间数据类型的标识符，其值为 `0xA1`。
/// 该常量用于在二进制数据中标记日期时间类型，帮助库在序列化和反序列化过程中识别和处理日期时间数据。
pub const BINN_DATETIME: u8 = 0xA1;

/// Constant representing the type identifier for a 32-bit floating point number in Binn library.
/// This is used to mark and identify 32-bit floating point types in binary data.
/// Value: `0x62`
pub const BINN_FLOAT32: u8 = 0x62;

/// 常量 `BINN_SINGLE_STR` 定义了 Binn 库中单字节字符串类型的标识符，其值为 `0xA6`。
/// 该常量用于在二进制数据中标记单字节字符串类型，帮助库在处理数据时识别和解析字符串。
pub const BINN_SINGLE_STR: u8 = 0xA6;

/// Constant representing the type identifier for a 64-bit floating point number in Binn library.
/// This is used to mark and identify 64-bit floating point types in binary data.
/// Value: `0x82`
pub const BINN_FLOAT64: u8 = 0x82;

/// Constant representing the type identifier for a double-precision floating-point number in Binn library.
/// This is used to mark and identify double-precision floating-point types in binary data.
/// Value: `0xA7`
pub const BINN_DOUBLE_STR: u8 = 0xA7;

/// 常量 `BINN_SINGLE` 定义为 `BINN_FLOAT32`，用于表示单精度浮点数类型。
/// 该常量在 Binn 库中用于简化代码中对单精度浮点数的引用，使得开发者能够更直观地处理二进制数据中的浮点数。
pub const BINN_SINGLE: u8 = BINN_FLOAT32;

/// 常量 `BINN_FLOAT` 定义为 `BINN_FLOAT32`，用于表示 32 位浮点数类型。
/// 这个常量简化了代码中对浮点数类型的引用，使得开发者能够更直观地处理二进制数据中的浮点数。
pub const BINN_FLOAT: u8 = BINN_FLOAT32;

/// Constant representing the currency type in Binn library.
/// This constant is used to identify and handle currency data during serialization and deserialization.
pub const BINN_CURRENCY: u8 = 0x83;

/// 常量 `BINN_BOOL` 定义了 Binn 库中用于标识布尔类型数据的标识符，其值为 `0x80061`。
/// 该常量用于在二进制数据中标记布尔类型，帮助库在序列化和反序列化过程中正确识别和处理布尔值。
pub const BINN_BOOL: u32 = 0x80061;

/// Constant representing the storage format for BLOB (Binary Large Object) type in Binn library.
/// This constant is used to identify and handle BLOB data in binary format.
/// Value: `0xC0`
pub const BINN_BLOB: u8 = 0xC0;

/// Constant representing the HTML data type in Binn library.
/// This constant is used to identify and handle HTML format data during serialization and deserialization.
pub const BINN_HTML: u16 = 0xB001;

/// Constant representing the XML format data type in Binn library.
/// This constant is used to identify and handle XML format data during serialization and deserialization.
pub const BINN_XML: u16 = 0xB002;

/// Constant representing the CSS data type in Binn library.
/// This constant is used to identify and handle CSS data during serialization and deserialization.
pub const BINN_CSS: u16 = 0xB005;

/// 常量 `BINN_GIF` 定义了 Binn 库中 GIF 图像数据类型的标识符，其值为 `0xD002`。
/// 该常量用于在二进制数据中标记 GIF 图像类型，帮助库在序列化和反序列化过程中识别和处理 GIF 数据。
pub const BINN_GIF: u16 = 0xD002;

/// Constant representing the JSON format data type in Binn library.
/// This constant is used to identify and handle JSON data during serialization and deserialization.
pub const BINN_JSON: u16 = 0xB003;

/// 常量 `BINN_JPEG` 定义了 Binn 库中 JPEG 图像类型的标识符，其值为 `0xD001`。
/// 该常量用于在二进制数据中标记 JPEG 图像，帮助库在序列化和反序列化过程中识别和处理 JPEG 数据。
pub const BINN_JPEG: u16 = 0xD001;

/// Constant representing the PNG format in Binn library.
/// This constant is used to identify and handle PNG format data during serialization and deserialization.
pub const BINN_PNG: u16 = 0xD003;

/// Constant representing the null data type family in Binn library.
/// This constant is used to identify and handle null or uninitialized data structures.
/// Value: `0xf1`
pub const BINN_FAMILY_NULL: u8 = 0xf1;

/// 常量 `BINN_JAVASCRIPT` 定义了一个值为 `0xB004` 的常量，用于标识与 JavaScript 相关的数据类型或格式。
/// 在 Binn 库中，这个常量用于在二进制数据中标记特定的数据类型，以便在处理或解析数据时能够识别和区分不同的数据格式。
pub const BINN_JAVASCRIPT: u16 = 0xB004;

/// Constant representing the integer family type in Binn library.
/// This constant is used to identify and handle integer type data during serialization and deserialization.
pub const BINN_FAMILY_INT: u8 = 0xf2;

/// Constant representing the family identifier for floating-point numbers in Binn library.
/// This constant is used to distinguish floating-point data types in binary data processing,
/// ensuring correct identification and manipulation of floating-point values.
pub const BINN_FAMILY_FLOAT: u8 = 0xf3;

/// 常量 `BINN_BMP` 定义了 Binn 库中 BMP 图像数据类型的标识符，其值为 `0xD004`。
/// 该常量用于在二进制数据中标记 BMP 图像类型，帮助库在序列化和反序列化过程中识别和处理 BMP 图像数据。
pub const BINN_BMP: u8 = 0xD004;

/// Constant representing the string data type family in Binn library.
/// This constant is used to identify and handle string data during serialization and deserialization.
pub const BINN_FAMILY_STRING: u8 = 0xf4;

/// Constant representing the family identifier for boolean type data in Binn library.
/// This constant is used to distinguish boolean type data in binary data processing,
/// ensuring correct identification and manipulation of boolean values.
pub const BINN_FAMILY_BOOL: u8 = 0xf6;

/// Constant representing the family type for no specific data type in Binn library.
/// This constant is used to identify data structures that do not belong to any specific data type family,
/// typically used as a default value or error state.
pub const BINN_FAMILY_NONE: u8 = 0x00;

/// Constant representing the BLOB (Binary Large Object) family type in Binn library.
/// This constant is used to identify and handle BLOB type data during serialization and deserialization.
pub const BINN_FAMILY_BLOB: u8 = 0xf5;

/// 常量 `BINN_SIGNED_INT` 定义了 Binn 库中用于表示有符号整数的数据类型标识符，其值为 11。
/// 这个标识符在库中用于区分不同的数据类型，使得库能够正确处理和存储有符号整数。
/// 通过这个常量，开发者可以在创建或解析二进制数据时明确指定数据类型，确保数据在序列化和反序列化过程中保持一致性。
pub const BINN_SIGNED_INT: u8 = 11;

/// Constant representing the type identifier for an unsigned integer in Binn library.
/// This constant is used to identify and handle unsigned integer types during serialization and deserialization.
pub const BINN_UNSIGNED_INT: u32 = 22;

/// Constant representing the binary data family type in Binn library.
/// This constant is used to identify and handle binary data structures during serialization and deserialization.
pub const BINN_FAMILY_BINN: u8 = 0xf7;

/// 常量 `BINN_STATIC` 定义了一个特殊的内存管理标志，表示某些数据结构的内存不需要手动释放。
/// 这个常量通常用于标记那些静态分配或生命周期由外部管理的二进制数据对象。
pub const BINN_STATIC: u32 = 0;

/// A constant representing a special memory release flag indicating that certain data does not need explicit memory deallocation.
/// This flag is typically used for temporary or short-lived data structures to avoid unnecessary memory management operations.
pub const BINN_TRANSIENT: i32 = -1;

/// 常量 `BINN_MAGIC` 定义了一个用于标识二进制数据结构的魔数（Magic Number），其值为 `0x1F22B11F`。
/// 这个魔数在二进制数据的头部使用，用于快速验证数据的格式和完整性。
pub const BINN_MAGIC: u32 = 0x1F22B11F;

/// 常量 `MIN_BINN_SIZE` 定义了二进制数据结构的最小大小，确保在创建或操作二进制数据时，数据结构至少占用 3 个字节的空间。
/// 这一限制用于保证数据的基本完整性和有效性，防止因数据过小而导致的错误或异常。
pub const MIN_BINN_SIZE: usize = 3;

/// Sets the type of a Binn item to `BINN_NULL`, indicating that the item is null.
///
/// # Arguments
/// * `item` - A mutable reference to the Binn item whose type will be set to `BINN_NULL`.
pub fn binn_set_null(item: &mut BinnItem) {
    item.type_ = BINN_NULL;
}

/// Checks if a Binn item is writable.
///
/// # Arguments
/// * `item` - A reference to the Binn item to check.
///
/// # Returns
/// Returns `true` if the item is writable, otherwise `false`.
pub fn binn_is_writable(item: &BinnItem) -> bool {
    item.writable
}

/// The maximum length of the binary data header, set to 9 bytes.
/// This constant is used to limit the size of the header in binary data structures,
/// ensuring that the header information does not exceed the predefined length.
pub const MAX_BINN_HEADER: usize = 9;

/// 常量 `CHUNK_SIZE` 定义了数据块的大小为 256 字节，用于在二进制数据处理过程中划分和管理内存。
/// 它在文件 `binn.c` 中作为内存分配和数据操作的基准单位，确保数据存储和读取时的高效性和一致性。
/// 通过使用这个固定大小的块，代码能够更好地管理内存资源，避免频繁的内存分配和释放，从而提升性能。
/// `CHUNK_SIZE` 与其他模块交互时，主要用于数据分块存储、压缩和传输，确保二进制数据在不同系统间的兼容性和高效交换。
pub const CHUNK_SIZE: usize = 256;

/// 将一个整数值设置到 Binn 数据结构中的指定项。
///
/// # 参数
/// - `item`: 指向 Binn 项的可变引用。
/// - `value`: 要设置的整数值。
///
/// # 返回值
/// 无返回值。
pub fn binn_set_int(item: &mut BinnItem, value: i32) {
    item.type_ = BINN_INT32;
    item.vint32 = value;
    item.ptr = &item.vint32 as *const _ as *mut _;
}

/// 将双精度浮点数四舍五入为最接近的整数。
///
/// # 参数
/// - `dbl`: 需要四舍五入的双精度浮点数。
///
/// # 返回值
/// 返回四舍五入后的整数。
pub fn roundval(dbl: f64) -> i32 {
    if dbl >= 0.0 {
        (dbl + 0.5) as i32
    } else {
        if (dbl - (dbl as i32 as f64)) <= -0.5 {
            dbl as i32
        } else {
            (dbl - 0.5) as i32
        }
    }
}

/// Sets a floating-point value in a Binn item.
///
/// # Arguments
/// * `item` - A mutable reference to the Binn item to be modified.
/// * `value` - The floating-point value to set.
pub fn binn_set_float(item: &mut BinnItem, value: f32) {
    item.type_ = BINN_FLOAT;
    item.vfloat = value;
    item.ptr = &item.vfloat as *const _ as *mut _;
}

/// 常量 `BINN_STRUCT` 定义了二进制数据结构中的特定类型标识符，其值为 `1`。
/// 该常量用于在二进制数据中标记特定的数据结构类型，帮助库在序列化和反序列化过程中识别和处理不同的数据格式。
pub const BINN_STRUCT: u8 = 1;

/// 常量 `BINN_BUFFER` 定义了一个值为 `2` 的常量，用于标识二进制数据存储中的缓冲区类型。
/// 它在 Binn 库中用于区分不同的数据存储方式，帮助系统在处理二进制数据时选择合适的存储策略。
pub const BINN_BUFFER: u8 = 2;

/// Compares two strings case-insensitively.
///
/// # Arguments
/// * `s1` - The first string to compare.
/// * `s2` - The second string to compare.
///
/// # Returns
/// Returns `true` if the strings are equal (case-insensitive), otherwise `false`.
pub fn stricmp(s1: &str, s2: &str) -> bool {
    s1.to_lowercase() == s2.to_lowercase()
}

/// Sets a boolean value in a Binn item.
///
/// # Arguments
/// * `item` - A mutable reference to the Binn item.
/// * `value` - The boolean value to set.
///
/// # Panics
/// This function will panic if the item is not writable.
pub fn binn_set_bool(item: &mut BinnItem, value: bool) {
    item.type_ = BINN_BOOL;
    item.vbool = value;
    item.ptr = Some(&item.vbool);
}

/// Sets a 64-bit unsigned integer value in a Binn data structure.
///
/// # Arguments
/// * `item` - A mutable reference to the Binn item to be modified.
/// * `value` - The 64-bit unsigned integer value to set.
pub fn binn_set_uint64(item: &mut BinnItem, value: u64) {
    item.type_ = BINN_UINT64;
    item.vuint64 = value;
    item.ptr = &item.vuint64 as *const _ as *mut _;
}

/// Sets a double-precision floating-point value in a Binn item.
///
/// # Arguments
/// * `item` - A mutable reference to the Binn item to be modified.
/// * `value` - The double-precision floating-point value to set.
pub fn binn_set_double(item: &mut BinnItem, value: f64) {
    item.type_ = BINN_DOUBLE;
    item.vdouble = value;
    item.ptr = &item.vdouble as *const _ as *mut _;
}

/// Compares the first `n` characters of two strings in a case-insensitive manner.
///
/// # Arguments
/// * `s1` - The first string to compare.
/// * `s2` - The second string to compare.
/// * `n` - The number of characters to compare.
///
/// # Returns
/// Returns `true` if the first `n` characters of `s1` and `s2` are equal, ignoring case.
/// Otherwise, returns `false`.
pub fn strnicmp(s1: &str, s2: &str, n: usize) -> bool {
    s1.chars().take(n).eq(s2.chars().take(n).map(|c| c.to_ascii_lowercase()))
}

/// An iterator for traversing binary data in the Binn library.
///
/// This struct is used to step through binary data, allowing for efficient parsing and manipulation.
/// It contains pointers to the current position and the limit of the data, along with type information,
/// total count of elements, and the current index.
#[derive(Debug, Clone, PartialEq)]
pub struct BinnIter {
    /// Pointer to the next element in the binary data.
    pub pnext: *mut u8,
    /// Pointer to the limit of the binary data.
    pub plimit: *mut u8,
    /// The type of the current element.
    pub type_: i32,
    /// The total number of elements in the binary data.
    pub count: i32,
    /// The current index in the iteration.
    pub current: i32,
}

/// Type alias for `BinnIter`.
pub type BinnIterStruct = BinnIter;

/// Sets an unsigned 32-bit integer value in a Binn item.
///
/// # Arguments
/// * `item` - A mutable reference to the Binn item to be modified.
/// * `value` - The unsigned 32-bit integer value to set.
pub fn binn_set_uint(item: &mut BinnItem, value: u32) {
    item.type_ = BINN_UINT32;
    item.vuint32 = value;
    item.ptr = &item.vuint32 as *const u32 as *mut u8;
}

/// A function pointer type for custom memory allocation.
/// This allows developers to replace the default memory allocation function with a custom one.
/// The function takes a `usize` parameter representing the size of memory to allocate and returns a pointer to the allocated memory.
pub static mut malloc_fn: Option<fn(usize) -> *mut u8> = None;

/// Creates a binary data type identifier based on the given storage type and data type index.
///
/// # Arguments
/// * `storage_type` - The storage type of the binary data.
/// * `data_type_index` - The index of the data type.
///
/// # Returns
/// Returns a `Result` containing the type identifier if successful, or an error message if the input is invalid.
pub fn binn_create_type(storage_type: u8, data_type_index: i32) -> Result<i32, &'static str> {
    if data_type_index < 0 {
        return Err("data_type_index cannot be negative");
    }
    if storage_type < BINN_STORAGE_MIN || storage_type > BINN_STORAGE_MAX {
        return Err("storage_type is out of valid range");
    }
    if data_type_index < 16 {
        Ok(storage_type as i32 | data_type_index)
    } else if data_type_index < 4096 {
        let storage_type = (storage_type | BINN_STORAGE_HAS_MORE) as i32;
        let storage_type = storage_type << 8;
        let data_type_index = data_type_index >> 4;
        Ok(storage_type | data_type_index)
    } else {
        Err("data_type_index is too large")
    }
}

/// A function pointer type for reallocating memory.
///
/// # Arguments
/// * `ptr` - A pointer to the previously allocated memory block.
/// * `len` - The new size of the memory block in bytes.
///
/// # Returns
/// Returns a pointer to the newly allocated memory block, or `None` if the allocation failed.
pub type ReallocFn = Option<unsafe extern "C" fn(ptr: Option<*mut u8>, len: usize) -> Option<*mut u8>>;

/// A global variable holding the reallocation function pointer.
///
/// This variable can be set to a custom reallocation function to override the default behavior.
pub static mut REALLOC_FN: ReallocFn = None;

/// Constant representing the storage format for string data in Binn.
/// This value is used to identify and handle string data during serialization and deserialization.
pub const BINN_STRING: u8 = 0xA0;

/// Iterates over a Binn map, calling the provided closure for each key-value pair.
///
/// # Arguments
/// * `map` - The Binn map to iterate over.
/// * `f` - A closure that takes a key and value as arguments.
pub fn binn_map_foreach<F>(map: &BinnMap, mut f: F)
where
    F: FnMut(&BinnValue, &BinnValue),
{
    let mut iter = BinnIter::new(map, BINN_MAP);
    while let Some((id, value)) = iter.next() {
        f(id, value);
    }
}

/// 设置 Binn 数据结构中的 64 位整数值
///
/// # 参数
/// - `item`: 指向 Binn 项的引用
/// - `value`: 要设置的 64 位整数值
///
/// # 返回值
/// 无
pub fn binn_set_int64(item: &mut BinnItem, value: i64) {
    item.type_ = BINN_INT64;
    item.vint64 = value;
    item.ptr = &item.vint64 as *const _ as *mut _;
}

/// Iterates over key-value pairs in a Binn object.
///
/// # Arguments
/// * `object` - The Binn object to iterate over.
/// * `key` - A mutable reference to store the current key.
/// * `value` - A mutable reference to store the current value.
///
/// # Returns
/// An iterator over the key-value pairs in the Binn object.
pub fn binn_object_foreach(object: &BinnObject, key: &mut BinnKey, value: &mut BinnValue) -> impl Iterator<Item = (&BinnKey, &BinnValue)> {
    let mut iter = BinnIter::new(object, BINN_OBJECT);
    std::iter::from_fn(move || {
        if binn_object_next(&mut iter, key, value) {
            Some((key, value))
        } else {
            None
        }
    })
}

/// 遍历 Binn 列表中的每个元素。
///
/// # 参数
/// - `list`: 要遍历的 Binn 列表。
/// - `value`: 用于存储当前元素的变量。
///
/// # 返回值
/// 返回一个迭代器，逐个访问列表中的元素。
pub fn binn_list_foreach(list: &BinnList) -> impl Iterator<Item = &BinnValue> {
    list.iter()
}

/// Calculates the minimum allocation size needed to accommodate the required size.
///
/// # Arguments
/// * `needed_size` - The size of memory needed.
/// * `alloc_size` - The current allocated size.
///
/// # Returns
/// Returns the calculated allocation size that is at least as large as `needed_size`.
fn calc_allocation(needed_size: usize, alloc_size: usize) -> usize {
    let mut calc_size = alloc_size;
    while calc_size < needed_size {
        calc_size <<= 1;  // same as *= 2
    }
    calc_size
}

/// Returns the storage size in bytes for a given storage type.
///
/// # Arguments
/// * `storage_type` - The storage type to get the size for.
///
/// # Returns
/// The size in bytes for the given storage type, or 0 if the type is invalid.
pub fn get_storage_size(storage_type: u8) -> usize {
    match storage_type {
        BINN_STORAGE_NOBYTES => 0,
        BINN_STORAGE_BYTE => 1,
        BINN_STORAGE_WORD => 2,
        BINN_STORAGE_DWORD => 4,
        BINN_STORAGE_QWORD => 8,
        _ => 0,
    }
}

/// Determines the type of data pointed to by the given pointer.
///
/// # Arguments
/// * `ptr` - A reference to the data to check.
///
/// # Returns
/// Returns `Some(BINN_STRUCT)` if the data is a binary structure, `Some(BINN_BUFFER)` if it is a buffer,
/// or `None` if the pointer is null.
pub fn binn_get_ptr_type(ptr: Option<&u32>) -> Option<u8> {
    ptr.map(|p| {
        match *p {
            BINN_MAGIC => BINN_STRUCT,
            _ => BINN_BUFFER,
        }
    })
}

/// 根据数据类型判断它是带符号整数还是无符号整数，并返回相应的分类标识。
///
/// # 参数
/// - `type_`: 数据类型标识符。
///
/// # 返回值
/// 返回 `BINN_SIGNED_INT` 表示带符号整数，返回 `BINN_UNSIGNED_INT` 表示无符号整数，否则返回 `0`。
pub fn int_type(type_: u8) -> u8 {
    match type_ {
        BINN_INT8 | BINN_INT16 | BINN_INT32 | BINN_INT64 => BINN_SIGNED_INT,
        BINN_UINT8 | BINN_UINT16 | BINN_UINT32 | BINN_UINT64 => BINN_UNSIGNED_INT,
        _ => 0,
    }
}

/// Determines the family category of a given Binn type.
///
/// # Arguments
/// * `type_` - The Binn type to classify.
///
/// # Returns
/// Returns the family category of the given type.
pub fn type_family(type_: u8) -> u8 {
    match type_ {
        BINN_LIST | BINN_MAP | BINN_OBJECT => BINN_FAMILY_BINN,
        BINN_INT8 | BINN_INT16 | BINN_INT32 | BINN_INT64 | BINN_UINT8 | BINN_UINT16 | BINN_UINT32 | BINN_UINT64 => BINN_FAMILY_INT,
        BINN_FLOAT32 | BINN_FLOAT64 | BINN_SINGLE_STR | BINN_DOUBLE_STR => BINN_FAMILY_FLOAT,
        BINN_STRING | BINN_HTML | BINN_CSS | BINN_XML | BINN_JSON | BINN_JAVASCRIPT => BINN_FAMILY_STRING,
        BINN_BLOB | BINN_JPEG | BINN_GIF | BINN_PNG | BINN_BMP => BINN_FAMILY_BLOB,
        BINN_DECIMAL | BINN_CURRENCY | BINN_DATE | BINN_TIME | BINN_DATETIME => BINN_FAMILY_STRING,
        BINN_BOOL => BINN_FAMILY_BOOL,
        BINN_NULL => BINN_FAMILY_NULL,
        _ => BINN_FAMILY_NONE,
    }
}

/// A local variable of type `Binn` used to store binary data temporarily.
/// This variable is typically used for parsing or generating binary formats.
let local_value: Binn;

/// Checks if a string represents a valid integer.
///
/// # Arguments
/// * `s` - A string slice to check.
///
/// # Returns
/// Returns `true` if the string represents a valid integer, otherwise `false`.
pub fn is_integer(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    if chars.next() == Some('-') {
        if chars.next().is_none() {
            return false;
        }
    }
    chars.all(|c| c.is_ascii_digit())
}

/// Checks if the given pointer points to a valid binary data structure.
///
/// # Arguments
/// * `ptr` - A pointer to the binary data to check.
///
/// # Returns
/// Returns `true` if the pointer is valid and the data contains the magic value `BINN_MAGIC`,
/// otherwise returns `false`.
pub fn binn_is_struct(ptr: Option<*const u8>) -> bool {
    if let Some(ptr) = ptr {
        unsafe {
            *(ptr as *const u32) == BINN_MAGIC
        }
    } else {
        false
    }
}

/// Copies a floating-point value from the source to the destination, converting between different precisions if necessary.
///
/// # Arguments
/// * `source` - A reference to the source floating-point value.
/// * `dest` - A mutable reference to the destination where the value will be copied.
/// * `source_type` - The type of the source floating-point value.
/// * `dest_type` - The type of the destination floating-point value.
///
/// # Returns
/// Returns `Ok(())` if the conversion and copy were successful, otherwise returns `Err(())`.
pub fn copy_float_value(source: &f32, dest: &mut f64, source_type: BinnType, dest_type: BinnType) -> Result<(), ()> {
    match source_type {
        BinnType::Float32 => {
            *dest = *source as f64;
            Ok(())
        }
        BinnType::Float64 => {
            *dest = *source as f64;
            Ok(())
        }
        _ => Err(()),
    }
}


/// 检查一个字符串是否表示一个有效的浮点数。
///
/// # 参数
/// - `s`: 输入的字符串。
///
/// # 返回值
/// 返回 `true` 如果字符串是一个有效的浮点数，否则返回 `false`。
pub fn is_float(s: &str) -> bool {
    let mut number_found = false;
    let mut chars = s.chars();

    // 跳过开头的负号
    if let Some('-') = chars.next() {
        // 如果负号后没有字符，返回 false
        if chars.as_str().is_empty() {
            return false;
        }
    }

    for c in chars {
        if c == '.' || c == ',' {
            if !number_found {
                return false;
            }
        } else if c.is_ascii_digit() {
            number_found = true;
        } else {
            return false;
        }
    }

    number_found
}

/// Copies a 16-bit unsigned integer from the source to the destination, handling endianness.
///
/// # Arguments
/// * `dest` - A mutable reference to the destination `u16` value.
/// * `src` - A reference to the source `u16` value.
pub fn copy_be16(dest: &mut u16, src: &u16) {
    if cfg!(target_endian = "little") {
        *dest = src.to_be();
    } else {
        *dest = *src;
    }
}

/// Sets custom memory allocation functions for the Binn library.
///
/// # Arguments
/// * `new_malloc` - A function pointer for memory allocation.
/// * `new_realloc` - A function pointer for memory reallocation.
/// * `new_free` - A function pointer for memory deallocation.
pub fn binn_set_alloc_functions(new_malloc: Option<fn(usize) -> *mut u8>, new_realloc: Option<unsafe extern "C" fn(Option<*mut u8>, usize) -> Option<*mut u8>>, new_free: Option<unsafe extern "C" fn(*mut u8)>) {
    unsafe {
        malloc_fn = new_malloc;
        REALLOC_FN = new_realloc;
        free_fn = new_free;
    }
}

/// Copies a 64-bit unsigned integer from the source to the destination,
/// handling endianness and memory alignment.
///
/// # Arguments
/// * `dest` - A mutable reference to the destination `u64`.
/// * `source` - A reference to the source `u64`.
pub fn copy_be64(dest: &mut u64, source: &u64) {
    if cfg!(target_endian = "little") {
        *dest = source.to_be();
    } else {
        *dest = *source;
    }
}

/// Extracts the storage type and extra type information from a given `long_type`.
///
/// # Arguments
/// * `long_type` - The type identifier to be parsed.
///
/// # Returns
/// Returns a `Result` containing a tuple of `(storage_type, extra_type)` if successful,
/// or an error message if the `long_type` is invalid.
pub fn binn_get_type_info(long_type: i32) -> Result<(i32, i32), &'static str> {
    let mut storage_type;
    let mut extra_type;

    if long_type < 0 {
        return Err("Invalid long_type");
    } else if long_type <= 0xff {
        storage_type = long_type & BINN_STORAGE_MASK as i32;
        extra_type = long_type & BINN_TYPE_MASK as i32;
    } else if long_type <= 0xffff {
        storage_type = long_type & BINN_STORAGE_MASK16 as i32;
        storage_type >>= 8;
        extra_type = long_type & BINN_TYPE_MASK16 as i32;
        extra_type >>= 4;
    } else if long_type & BINN_STORAGE_VIRTUAL as i32 != 0 {
        return binn_get_type_info(long_type & 0xffff);
    } else {
        return Err("Invalid long_type");
    }

    Ok((storage_type, extra_type))
}

/// Converts the given data to a format suitable for writing to binary.
///
/// # Arguments
/// * `ptype` - A mutable reference to the data type.
/// * `ppvalue` - A mutable reference to the value pointer.
/// * `psize` - A mutable reference to the size of the data.
///
/// # Returns
/// Returns `Ok(())` if the conversion is successful, otherwise returns an error message.
pub fn get_write_converted_data(ptype: &mut i32, ppvalue: &mut Option<&mut dyn std::any::Any>, psize: &mut usize) -> Result<(), &'static str> {
    let type_ = *ptype;

    if ppvalue.is_none() {
        match type_ {
            BINN_NULL | BINN_TRUE | BINN_FALSE => {},
            BINN_STRING | BINN_BLOB => {
                if *psize == 0 {}
                else {
                    return Err("Invalid value pointer");
                }
            },
            _ => return Err("Invalid value pointer"),
        }
    }

    match type_ {
        BINN_SINGLE => {
            if let Some(value) = ppvalue {
                if let Some(float_value) = value.downcast_mut::<f32>() {
                    let d1 = *float_value as f64;
                    let pstr = format!("{:.17e}", d1);
                    *ppvalue = Some(Box::leak(pstr.into_boxed_str()) as &mut dyn std::any::Any);
                    *ptype = BINN_SINGLE_STR;
                }
            }
        },
        BINN_DOUBLE => {
            if let Some(value) = ppvalue {
                if let Some(double_value) = value.downcast_mut::<f64>() {
                    let pstr = format!("{:.17e}", *double_value);
                    *ppvalue = Some(Box::leak(pstr.into_boxed_str()) as &mut dyn std::any::Any);
                    *ptype = BINN_DOUBLE_STR;
                }
            }
        },
        BINN_BOOL => {
            if let Some(value) = ppvalue {
                if let Some(bool_value) = value.downcast_mut::<bool>() {
                    *ptype = if *bool_value { BINN_TRUE } else { BINN_FALSE };
                }
            }
        },
        BINN_DECIMAL | BINN_CURRENCYSTR | BINN_DATE | BINN_DATETIME | BINN_TIME => {
            // Temporary implementation, return Ok for now
            return Ok(());
        },
        _ => return Err("Unsupported type"),
    }

    Ok(())
}

/// Copies a 32-bit integer from the source to the destination, handling endianness and alignment.
///
/// # Arguments
/// * `dest` - A mutable reference to the destination u32.
/// * `src` - A reference to the source u32.
pub fn copy_be32(dest: &mut u32, src: &u32) {
    #[cfg(target_endian = "little")]
    {
        let src_bytes = src.to_ne_bytes();
        *dest = u32::from_ne_bytes([src_bytes[3], src_bytes[2], src_bytes[1], src_bytes[0]]);
    }
    #[cfg(target_endian = "big")]
    {
        #[cfg(feature = "only_aligned_access")]
        {
            if (src as *const u32 as usize) % std::mem::align_of::<u32>() == 0 {
                *dest = *src;
            } else {
                let src_bytes = src.to_ne_bytes();
                *dest = u32::from_ne_bytes([src_bytes[0], src_bytes[1], src_bytes[2], src_bytes[3]]);
            }
        }
        #[cfg(not(feature = "only_aligned_access"))]
        {
            *dest = *src;
        }
    }
}

/// Copies raw data from the source to the destination based on the data storage type.
///
/// # Arguments
/// * `source` - A reference to the source data.
/// * `dest` - A mutable reference to the destination data.
/// * `data_store` - The type of data storage, determining how the data is copied.
///
/// # Returns
/// Returns `Ok(())` if the copy was successful, otherwise returns an error message.
pub fn copy_raw_value(source: &[u8], dest: &mut [u8], data_store: DataStore) -> Result<(), &'static str> {
    match data_store {
        DataStore::NoBytes => {},
        DataStore::Byte => {
            if source.len() >= 1 && dest.len() >= 1 {
                dest[0] = source[0];
            } else {
                return Err("Insufficient data length for Byte copy");
            }
        },
        DataStore::Word => {
            if source.len() >= 2 && dest.len() >= 2 {
                dest[..2].copy_from_slice(&source[..2]);
            } else {
                return Err("Insufficient data length for Word copy");
            }
        },
        DataStore::Dword => {
            if source.len() >= 4 && dest.len() >= 4 {
                dest[..4].copy_from_slice(&source[..4]);
            } else {
                return Err("Insufficient data length for Dword copy");
            }
        },
        DataStore::Qword => {
            if source.len() >= 8 && dest.len() >= 8 {
                dest[..8].copy_from_slice(&source[..8]);
            } else {
                return Err("Insufficient data length for Qword copy");
            }
        },
        DataStore::Blob | DataStore::String | DataStore::Container => {
            if source.len() >= std::mem::size_of::<usize>() && dest.len() >= std::mem::size_of::<usize>() {
                let src_ptr = source.as_ptr() as *const usize;
                let dest_ptr = dest.as_mut_ptr() as *mut usize;
                unsafe {
                    *dest_ptr = *src_ptr;
                }
            } else {
                return Err("Insufficient data length for pointer copy");
            }
        },
        _ => return Err("Invalid data storage type"),
    }
    Ok(())
}

/// Enum representing different data storage types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataStore {
    NoBytes,
    Byte,
    Word,
    Dword,
    Qword,
    Blob,
    String,
    Container,
}

/// 将字符串解析为 64 位有符号整数。
///
/// # 参数
/// - `s`: 要解析的字符串。
///
/// # 返回值
/// 返回 `Result<i64, ParseIntError>`，表示解析结果。如果解析成功，返回 `Ok(i64)`，否则返回 `Err(ParseIntError)`。
pub fn atoi64(s: &str) -> Result<i64, std::num::ParseIntError> {
    s.parse::<i64>()
}

/// Duplicates a block of memory.
///
/// # Arguments
/// * `src` - A reference to the source memory block.
/// * `size` - The size of the memory block to duplicate.
///
/// # Returns
/// Returns `Some(Vec<u8>)` containing the duplicated memory block if successful,
/// otherwise returns `None` if `src` is `None` or `size` is less than or equal to 0.
pub fn binn_memdup(src: Option<&[u8]>, size: usize) -> Option<Vec<u8>> {
    if src.is_none() || size == 0 {
        return None;
    }
    let src = src.unwrap();
    let mut dest = Vec::with_capacity(size);
    dest.copy_from_slice(src);
    Some(dest)
}

/// Represents the core data structure for handling binary data in the Binn library.
/// This struct is used to create, read, and manipulate binary data, supporting various data types
/// such as integers, floats, booleans, and more.
#[derive(Debug, Clone, PartialEq)]
pub struct Binn {
    /// Magic number to identify this memory block as a Binn structure.
    pub header: u32,
    /// Indicates whether the struct is allocated using `malloc_fn` or is on the stack.
    pub allocated: bool,
    /// Indicates whether the struct is writable.
    pub writable: bool,
    /// Indicates whether the container header is not written to the buffer.
    pub dirty: bool,
    /// Pointer to the buffer.
    pub pbuf: Option<Box<[u8]>>,
    /// Indicates whether the buffer is pre-allocated.
    pub pre_allocated: bool,
    /// Size of the allocated memory.
    pub alloc_size: usize,
    /// Size of the used memory.
    pub used_size: usize,
    /// Type of the data.
    pub type_: i32,
    /// Pointer to the data.
    pub ptr: Option<Box<dyn std::any::Any>>,
    /// Size of the data.
    pub size: usize,
    /// Number of items in the data.
    pub count: usize,
    /// Function to free memory (used only when type is `BINN_STRING` or `BINN_BLOB`).
    pub freefn: Option<fn(Box<dyn std::any::Any>)>,
    /// Union of various data types.
    pub value: BinnValue,
    /// Flag to disable integer compression.
    pub disable_int_compression: bool,
}

/// Represents the union of various data types in the Binn structure.
#[derive(Debug, Clone, PartialEq)]
pub enum BinnValue {
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float(f32),
    Double(f64),
    Bool(bool),
}

/// Safely copies an integer value from a source to a destination, converting between different integer types.
///
/// # Arguments
/// * `psource` - A reference to the source data.
/// * `pdest` - A mutable reference to the destination data.
/// * `source_type` - The type of the source data.
/// * `dest_type` - The type of the destination data.
///
/// # Returns
/// Returns `Ok(())` if the copy was successful, or an error message if the conversion is not possible.
pub fn copy_int_value(psource: &u8, pdest: &mut u8, source_type: u8, dest_type: u8) -> Result<(), &'static str> {
    let mut vuint64: u64 = 0;
    let mut vint64: i64 = 0;

    match source_type {
        BINN_INT8 => vint64 = *(psource as *const u8 as *const i8) as i64,
        BINN_INT16 => vint64 = *(psource as *const u8 as *const i16) as i64,
        BINN_INT32 => vint64 = *(psource as *const u8 as *const i32) as i64,
        BINN_INT64 => vint64 = *(psource as *const u8 as *const i64),
        BINN_UINT8 => vuint64 = *psource as u64,
        BINN_UINT16 => vuint64 = *(psource as *const u8 as *const u16) as u64,
        BINN_UINT32 => vuint64 = *(psource as *const u8 as *const u32) as u64,
        BINN_UINT64 => vuint64 = *(psource as *const u8 as *const u64),
        _ => return Err("Invalid source type"),
    }

    // Copy from int64 to uint64, if possible
    if int_type(source_type) == BINN_UNSIGNED_INT && int_type(dest_type) == BINN_SIGNED_INT {
        if vuint64 > i64::MAX as u64 {
            return Err("Value exceeds INT64_MAX");
        }
        vint64 = vuint64 as i64;
    } else if int_type(source_type) == BINN_SIGNED_INT && int_type(dest_type) == BINN_UNSIGNED_INT {
        if vint64 < 0 {
            return Err("Value is negative");
        }
        vuint64 = vint64 as u64;
    }

    match dest_type {
        BINN_INT8 => {
            if vint64 < i8::MIN as i64 || vint64 > i8::MAX as i64 {
                return Err("Value exceeds INT8 range");
            }
            *(pdest as *mut u8 as *mut i8) = vint64 as i8;
        }
        BINN_INT16 => {
            if vint64 < i16::MIN as i64 || vint64 > i16::MAX as i64 {
                return Err("Value exceeds INT16 range");
            }
            *(pdest as *mut u8 as *mut i16) = vint64 as i16;
        }
        BINN_INT32 => {
            if vint64 < i32::MIN as i64 || vint64 > i32::MAX as i64 {
                return Err("Value exceeds INT32 range");
            }
            *(pdest as *mut u8 as *mut i32) = vint64 as i32;
        }
        BINN_INT64 => {
            *(pdest as *mut u8 as *mut i64) = vint64;
        }
        BINN_UINT8 => {
            if vuint64 > u8::MAX as u64 {
                return Err("Value exceeds UINT8 range");
            }
            *pdest = vuint64 as u8;
        }
        BINN_UINT16 => {
            if vuint64 > u16::MAX as u64 {
                return Err("Value exceeds UINT16 range");
            }
            *(pdest as *mut u8 as *mut u16) = vuint64 as u16;
        }
        BINN_UINT32 => {
            if vuint64 > u32::MAX as u64 {
                return Err("Value exceeds UINT32 range");
            }
            *(pdest as *mut u8 as *mut u32) = vuint64 as u32;
        }
        BINN_UINT64 => {
            *(pdest as *mut u8 as *mut u64) = vuint64;
        }
        _ => return Err("Invalid destination type"),
    }

    Ok(())
}

/// Determines the storage type for a given Binn type.
///
/// # Arguments
/// * `type_` - The Binn type to get the storage type for.
///
/// # Returns
/// Returns the storage type if successful, or an error if the type is invalid.
pub fn binn_get_read_storage(type_: i32) -> Result<i32, &'static str> {
    match type_ {
        BINN_SINGLE_STR => Ok(BINN_STORAGE_DWORD),
        BINN_DOUBLE_STR => Ok(BINN_STORAGE_QWORD),
        BINN_BOOL | BINN_TRUE | BINN_FALSE => Ok(BINN_STORAGE_DWORD),
        _ => {
            let (storage_type, _) = binn_get_type_info(type_)?;
            Ok(storage_type)
        }
    }
}

/// Determines the storage type for a given Binn type.
///
/// # Arguments
/// * `type_` - The Binn type to get the storage type for.
///
/// # Returns
/// Returns the storage type for the given Binn type.
pub fn binn_get_write_storage(type_: i32) -> i32 {
    match type_ {
        BINN_SINGLE_STR | BINN_DOUBLE_STR => BINN_STORAGE_STRING,
        BINN_BOOL => BINN_STORAGE_NOBYTES,
        _ => {
            let (storage_type, _) = binn_get_type_info(type_).unwrap();
            storage_type
        }
    }
}

/// Compresses an integer value to the smallest possible storage type.
///
/// # Arguments
/// * `pstorage_type` - A mutable reference to the storage type.
/// * `ptype` - A mutable reference to the type.
/// * `psource` - A reference to the source value.
///
/// # Returns
/// Returns a reference to the compressed value.
pub fn compress_int(pstorage_type: &mut i32, ptype: &mut i32, psource: &i64) -> &i64 {
    let mut storage_type = *pstorage_type;
    if storage_type == BINN_STORAGE_BYTE {
        return psource;
    }

    let type_ = *ptype;
    let mut type2 = 0;
    let mut vint = 0;
    let mut vuint = 0;

    match type_ {
        BINN_INT64 => {
            vint = *psource;
            if vint >= 0 {
                vuint = vint as u64;
                type2 = match vuint {
                    _ if vuint <= u8::MAX as u64 => BINN_UINT8,
                    _ if vuint <= u16::MAX as u64 => BINN_UINT16,
                    _ if vuint <= u32::MAX as u64 => BINN_UINT32,
                    _ => type_,
                };
            } else {
                type2 = match vint {
                    _ if vint >= i8::MIN as i64 => BINN_INT8,
                    _ if vint >= i16::MIN as i64 => BINN_INT16,
                    _ if vint >= i32::MIN as i64 => BINN_INT32,
                    _ => type_,
                };
            }
        }
        BINN_INT32 => {
            vint = *psource as i32 as i64;
            if vint >= 0 {
                vuint = vint as u64;
                type2 = match vuint {
                    _ if vuint <= u8::MAX as u64 => BINN_UINT8,
                    _ if vuint <= u16::MAX as u64 => BINN_UINT16,
                    _ if vuint <= u32::MAX as u64 => BINN_UINT32,
                    _ => type_,
                };
            } else {
                type2 = match vint {
                    _ if vint >= i8::MIN as i64 => BINN_INT8,
                    _ if vint >= i16::MIN as i64 => BINN_INT16,
                    _ if vint >= i32::MIN as i64 => BINN_INT32,
                    _ => type_,
                };
            }
        }
        BINN_INT16 => {
            vint = *psource as i16 as i64;
            if vint >= 0 {
                vuint = vint as u64;
                type2 = match vuint {
                    _ if vuint <= u8::MAX as u64 => BINN_UINT8,
                    _ if vuint <= u16::MAX as u64 => BINN_UINT16,
                    _ if vuint <= u32::MAX as u64 => BINN_UINT32,
                    _ => type_,
                };
            } else {
                type2 = match vint {
                    _ if vint >= i8::MIN as i64 => BINN_INT8,
                    _ if vint >= i16::MIN as i64 => BINN_INT16,
                    _ if vint >= i32::MIN as i64 => BINN_INT32,
                    _ => type_,
                };
            }
        }
        BINN_UINT64 => {
            vuint = *psource as u64;
            type2 = match vuint {
                _ if vuint <= u8::MAX as u64 => BINN_UINT8,
                _ if vuint <= u16::MAX as u64 => BINN_UINT16,
                _ if vuint <= u32::MAX as u64 => BINN_UINT32,
                _ => type_,
            };
        }
        BINN_UINT32 => {
            vuint = *psource as u32 as u64;
            type2 = match vuint {
                _ if vuint <= u8::MAX as u64 => BINN_UINT8,
                _ if vuint <= u16::MAX as u64 => BINN_UINT16,
                _ if vuint <= u32::MAX as u64 => BINN_UINT32,
                _ => type_,
            };
        }
        BINN_UINT16 => {
            vuint = *psource as u16 as u64;
            type2 = match vuint {
                _ if vuint <= u8::MAX as u64 => BINN_UINT8,
                _ if vuint <= u16::MAX as u64 => BINN_UINT16,
                _ if vuint <= u32::MAX as u64 => BINN_UINT32,
                _ => type_,
            };
        }
        _ => {}
    }

    if type2 != 0 && type2 != type_ {
        *ptype = type2;
        *pstorage_type = binn_get_write_storage(type2);
    }

    psource
}


/// Sets a binary data block (BLOB) in a Binn item.
///
/// # Arguments
/// * `item` - A mutable reference to the Binn item to be modified.
/// * `ptr` - A pointer to the binary data.
/// * `size` - The size of the binary data.
/// * `pfree` - An optional memory deallocation function.
///
/// # Returns
/// Returns `Ok(())` if the operation was successful, otherwise returns `Err(())`.
pub fn binn_set_blob(item: &mut Binn, ptr: *const u8, size: usize, pfree: Option<fn(Box<dyn std::any::Any>)>) -> Result<(), ()> {
    if item.is_null() || ptr.is_null() {
        return Err(());
    }

    if pfree == Some(BINN_TRANSIENT) {
        let data = unsafe { std::slice::from_raw_parts(ptr, size) };
        item.ptr = Some(Box::new(data.to_vec()));
        item.freefn = Some(free_fn);
    } else {
        item.ptr = Some(Box::new(unsafe { std::slice::from_raw_parts(ptr, size) }));
        item.freefn = pfree;
    }

    item.type_ = BINN_BLOB;
    item.size = size;
    Ok(())
}

/// Reads a map ID from a binary data slice.
///
/// # Arguments
/// * `data` - A mutable slice of bytes representing the binary data.
///
/// # Returns
/// Returns `Some(id)` if the ID was successfully read, otherwise `None`.
pub fn read_map_id(data: &mut [u8]) -> Option<i32> {
    let mut p = 0;
    let c = data[p];
    p += 1;

    let extra_bytes = if c & 0x80 != 0 {
        ((c & 0x60) >> 5) + 1
    } else {
        0
    };

    if p + extra_bytes > data.len() {
        return None;
    }

    let type_ = c & 0xE0;
    let sign = c & 0x10;

    let id = match type_ {
        0x00 => (c & 0x3F) as i32,
        0x80 => {
            let id = (c & 0x0F) as i32;
            let id = (id << 8) | data[p] as i32;
            p += 1;
            id
        }
        0xA0 => {
            let id = (c & 0x0F) as i32;
            let id = (id << 8) | data[p] as i32;
            p += 1;
            let id = (id << 8) | data[p] as i32;
            p += 1;
            id
        }
        0xC0 => {
            let id = (c & 0x0F) as i32;
            let id = (id << 8) | data[p] as i32;
            p += 1;
            let id = (id << 8) | data[p] as i32;
            p += 1;
            let id = (id << 8) | data[p] as i32;
            p += 1;
            id
        }
        0xE0 => {
            let mut id = 0;
            for i in 0..4 {
                id = (id << 8) | data[p + i] as i32;
            }
            p += 4;
            id
        }
        _ => return None,
    };

    let id = if sign != 0 { -id } else { id };

    Some(id)
}

/// Advances the data position in a binary stream based on the storage type.
///
/// # Arguments
/// * `data` - A slice of binary data representing the current position in the stream.
/// * `limit` - A slice of binary data representing the limit of the stream.
///
/// # Returns
/// Returns `Some(&[u8])` with the updated position if successful, or `None` if the position exceeds the limit.
pub fn advance_data_pos(data: &[u8], limit: &[u8]) -> Option<&[u8]> {
    if data.is_empty() || data.as_ptr() > limit.as_ptr() {
        return None;
    }

    let byte = data[0];
    let storage_type = byte & BINN_STORAGE_MASK;
    let mut data = &data[1..];

    if byte & BINN_STORAGE_HAS_MORE != 0 {
        data = &data[1..];
    }

    match storage_type {
        BINN_STORAGE_NOBYTES => {},
        BINN_STORAGE_BYTE => {
            data = &data[1..];
        },
        BINN_STORAGE_WORD => {
            data = &data[2..];
        },
        BINN_STORAGE_DWORD => {
            data = &data[4..];
        },
        BINN_STORAGE_QWORD => {
            data = &data[8..];
        },
        BINN_STORAGE_BLOB | BINN_STORAGE_STRING => {
            if data.is_empty() {
                return None;
            }
            let mut data_size = data[0] as usize;
            if data_size & 0x80 != 0 {
                if data.len() < 4 {
                    return None;
                }
                data_size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize & 0x7FFFFFFF;
                data = &data[4..];
            } else {
                data = &data[1..];
            }
            if data.len() < data_size {
                return None;
            }
            data = &data[data_size..];
            if storage_type == BINN_STORAGE_STRING {
                data = &data[1..]; // Skip null terminator.
            }
        },
        BINN_STORAGE_CONTAINER => {
            if data.is_empty() {
                return None;
            }
            let mut data_size = data[0] as usize;
            if data_size & 0x80 != 0 {
                if data.len() < 4 {
                    return None;
                }
                data_size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize & 0x7FFFFFFF;
                data = &data[4..];
            }
            data_size -= 1; // Remove the type byte already added before.
            if data.len() < data_size {
                return None;
            }
            data = &data[data_size..];
        },
        _ => return None,
    }

    if data.as_ptr() > limit.as_ptr() {
        return None;
    }

    Some(data)
}

impl Binn {
    /// Checks if the current allocation is sufficient to accommodate additional data.
    /// If not, it attempts to reallocate memory to fit the new data.
    ///
    /// # Arguments
    /// * `add_size` - The size of the additional data to be accommodated.
    ///
    /// # Returns
    /// Returns `Ok(())` if the allocation is successful, otherwise returns an error message.
    pub fn check_allocation(&mut self, add_size: usize) -> Result<(), &'static str> {
        if self.used_size + add_size > self.alloc_size {
            if self.pre_allocated {
                return Err("Pre-allocated buffer cannot be resized");
            }
            let new_size = calc_allocation(self.used_size + add_size, self.alloc_size);
            let mut new_buf = Vec::with_capacity(new_size);
            if let Some(ref buf) = self.pbuf {
                new_buf.extend_from_slice(buf);
            }
            self.pbuf = Some(new_buf.into_boxed_slice());
            self.alloc_size = new_size;
        }
        Ok(())
    }
}

/// Checks if the `Binn` structure is a container type (list, map, or object).
///
/// # Returns
/// Returns `true` if the `Binn` structure is a container type, otherwise `false`.
pub fn is_container(&self) -> bool {
    match self.type_ {
        BINN_LIST | BINN_MAP | BINN_OBJECT => true,
        _ => false,
    }
}

/// Validates the binary data header and extracts type, count, size, and header size.
///
/// # Arguments
/// * `pbuf` - A reference to the binary data buffer.
/// * `ptype` - An optional mutable reference to store the type of the binary data.
/// * `pcount` - An optional mutable reference to store the count of elements.
/// * `psize` - An optional mutable reference to store the size of the binary data.
/// * `pheadersize` - An optional mutable reference to store the size of the header.
///
/// # Returns
/// Returns `Ok(())` if the header is valid, otherwise returns an error message.
pub fn is_valid_binn_header(
    pbuf: &[u8],
    ptype: Option<&mut u8>,
    pcount: Option<&mut u32>,
    psize: Option<&mut usize>,
    pheadersize: Option<&mut usize>,
) -> Result<(), &'static str> {
    if pbuf.is_empty() {
        return Err("Buffer is empty");
    }

    let mut p = pbuf;
    let plimit = if let Some(size) = psize {
        if *size > 0 {
            Some(&pbuf[*size - 1])
        } else {
            None
        }
    } else {
        None
    };

    // Get the type
    let byte = p[0];
    p = &p[1..];
    if (byte & BINN_STORAGE_MASK) != BINN_STORAGE_CONTAINER || (byte & BINN_STORAGE_HAS_MORE) != 0 {
        return Err("Invalid storage type");
    }
    let type_ = byte;

    match type_ {
        BINN_LIST | BINN_MAP | BINN_OBJECT => {},
        _ => return Err("Invalid container type"),
    }

    // Get the size
    if let Some(limit) = plimit {
        if p.as_ptr() > limit {
            return Err("Buffer overflow");
        }
    }
    let int32 = p[0];
    let (size, p) = if int32 & 0x80 != 0 {
        if let Some(limit) = plimit {
            if p.as_ptr().add(4) > limit {
                return Err("Buffer overflow");
            }
        }
        let size = u32::from_be_bytes([p[0], p[1], p[2], p[3]]) & 0x7FFFFFFF;
        (size as usize, &p[4..])
    } else {
        (int32 as usize, &p[1..])
    };

    // Get the count
    if let Some(limit) = plimit {
        if p.as_ptr() > limit {
            return Err("Buffer overflow");
        }
    }
    let int32 = p[0];
    let (count, p) = if int32 & 0x80 != 0 {
        if let Some(limit) = plimit {
            if p.as_ptr().add(4) > limit {
                return Err("Buffer overflow");
            }
        }
        let count = u32::from_be_bytes([p[0], p[1], p[2], p[3]]) & 0x7FFFFFFF;
        (count, &p[4..])
    } else {
        (int32 as u32, &p[1..])
    };

    if size < MIN_BINN_SIZE || count < 0 {
        return Err("Invalid size or count");
    }

    // Return the values
    if let Some(ptype) = ptype {
        *ptype = type_;
    }
    if let Some(pcount) = pcount {
        *pcount = count;
    }
    if let Some(psize) = psize {
        if *psize == 0 {
            *psize = size;
        }
    }
    if let Some(pheadersize) = pheadersize {
        *pheadersize = p.as_ptr() as usize - pbuf.as_ptr() as usize;
    }

    Ok(())
}

/// Allocates and initializes a new `Binn` structure.
///
/// This function allocates memory for a new `Binn` structure, initializes it to zero,
/// sets the header to `BINN_MAGIC`, and marks it as allocated.
///
/// # Returns
/// Returns `Some(Box<Binn>)` if the allocation and initialization are successful,
/// otherwise returns `None`.
pub fn binn_alloc_item() -> Option<Box<Binn>> {
    let mut item = Box::new(Binn {
        header: BINN_MAGIC,
        allocated: true,
        writable: false,
        dirty: false,
        pbuf: None,
        pre_allocated: false,
        alloc_size: 0,
        used_size: 0,
        type_: 0,
        ptr: None,
        size: 0,
        count: 0,
        freefn: None,
        value: BinnValue::Int8(0),
        disable_int_compression: false,
    });

    Some(item)
}

/// Sets a string value in a Binn item.
///
/// # Arguments
/// * `item` - A mutable reference to the Binn item to be modified.
/// * `str` - The string to set.
/// * `pfree` - An optional function pointer for memory deallocation.
///
/// # Returns
/// Returns `Ok(())` if the operation was successful, otherwise returns an error.
pub fn binn_set_string(item: &mut Binn, str: &str, pfree: Option<Box<dyn FnOnce(Box<dyn std::any::Any>)>>) -> Result<(), &'static str> {
    if str.is_empty() {
        return Err("String cannot be empty");
    }

    if let Some(free_fn) = pfree {
        item.ptr = Some(Box::new(str.as_bytes().to_vec()) as Box<dyn std::any::Any>);
        item.freefn = Some(free_fn);
    } else {
        item.ptr = Some(Box::new(str.as_bytes().to_vec()) as Box<dyn std::any::Any>);
        item.freefn = None;
    }

    item.type_ = BINN_STRING;
    Ok(())
}

impl Drop for Binn {
    /// 释放与 `Binn` 结构体相关的内存资源。
    ///
    /// 如果 `Binn` 结构体是可写的且不是预分配的，则释放 `pbuf` 指向的内存。
    /// 如果 `Binn` 结构体有自定义的释放函数 `freefn`，则调用它来释放 `ptr` 指向的内存。
    /// 如果 `Binn` 结构体是动态分配的，则释放整个 `Binn` 结构体；否则，将 `Binn` 结构体清零并重新设置 `header` 为 `BINN_MAGIC`。
    fn drop(&mut self) {
        if self.writable && !self.pre_allocated {
            self.pbuf = None;
        }

        if let Some(freefn) = self.freefn {
            if let Some(ptr) = self.ptr.take() {
                freefn(ptr);
            }
        }

        if self.allocated {
            // Rust 的 Box 会自动释放内存，无需手动调用 free_fn
        } else {
            *self = Binn {
                header: BINN_MAGIC,
                allocated: false,
                writable: false,
                dirty: false,
                pbuf: None,
                pre_allocated: false,
                alloc_size: 0,
                used_size: 0,
                type_: 0,
                ptr: None,
                size: 0,
                count: 0,
                freefn: None,
                value: BinnValue::Int8(0),
                disable_int_compression: false,
            };
        }
    }
}

/// Copies a value from the source to the destination, ensuring type compatibility.
///
/// # Arguments
/// * `psource` - A reference to the source data.
/// * `pdest` - A mutable reference to the destination data.
/// * `source_type` - The type of the source data.
/// * `dest_type` - The type of the destination data.
/// * `data_store` - The type of data storage, determining how the data is copied.
///
/// # Returns
/// Returns `Ok(())` if the copy was successful, otherwise returns `Err(())`.
pub fn copy_value(psource: &[u8], pdest: &mut [u8], source_type: u8, dest_type: u8, data_store: DataStore) -> Result<(), ()> {
    if type_family(source_type) != type_family(dest_type) {
        return Err(());
    }

    if type_family(source_type) == BINN_FAMILY_INT && source_type != dest_type {
        copy_int_value(psource, pdest, source_type, dest_type)
    } else if type_family(source_type) == BINN_FAMILY_FLOAT && source_type != dest_type {
        copy_float_value(psource, pdest, source_type, dest_type)
    } else {
        copy_raw_value(psource, pdest, data_store)
    }
}

/// Saves the header information of the Binn structure to the buffer.
///
/// This function writes the type, size, and count of the Binn structure to the buffer,
/// ensuring that the data is correctly formatted for storage or transmission.
///
/// # Returns
/// Returns `Ok(())` if the header was successfully saved, otherwise returns an error message.
pub fn save_header(&mut self) -> Result<(), &'static str> {
    if self.pbuf.is_none() {
        return Err("Buffer is not allocated");
    }

    let pbuf = self.pbuf.as_mut().unwrap();
    let mut p = pbuf.as_mut_ptr();
    let mut size = self.used_size;

    // Write the count
    if self.count > 127 {
        p = unsafe { p.offset(-4) };
        size += 3;
        let int32 = self.count | 0x80000000;
        copy_be32(unsafe { &mut *(p as *mut u32) }, &int32);
    } else {
        p = unsafe { p.offset(-1) };
        unsafe { *p = self.count as u8; }
    }

    // Write the size
    if size > 127 {
        p = unsafe { p.offset(-4) };
        size += 3;
        let int32 = size | 0x80000000;
        copy_be32(unsafe { &mut *(p as *mut u32) }, &int32);
    } else {
        p = unsafe { p.offset(-1) };
        unsafe { *p = size as u8; }
    }

    // Write the type
    p = unsafe { p.offset(-1) };
    unsafe { *p = self.type_ as u8; }

    // Update the Binn structure
    self.ptr = Some(pbuf);
    self.size = size;
    self.dirty = false;

    Ok(())
}

/// Initializes a `Binn` structure with the specified type and size.
///
/// # Arguments
/// * `item` - A mutable reference to the `Binn` structure to be initialized.
/// * `type_` - The type of the binary data (e.g., `BINN_LIST`, `BINN_MAP`, `BINN_OBJECT`).
/// * `size` - The size of the memory to allocate. If `0`, a default size will be used.
/// * `pointer` - An optional pointer to pre-allocated memory. If `None`, memory will be dynamically allocated.
///
/// # Returns
/// Returns `Ok(())` if the initialization is successful, otherwise returns an error message.
pub fn binn_create(item: &mut Binn, type_: i32, size: usize, pointer: Option<Box<[u8]>>) -> Result<(), &'static str> {
    match type_ {
        BINN_LIST | BINN_MAP | BINN_OBJECT => {},
        _ => return Err("Invalid type"),
    }

    if size < MIN_BINN_SIZE {
        if pointer.is_some() {
            return Err("Invalid size for pre-allocated memory");
        } else {
            size = 0;
        }
    }

    *item = Binn::default();

    if let Some(ptr) = pointer {
        item.pre_allocated = true;
        item.pbuf = Some(ptr);
        item.alloc_size = size;
    } else {
        item.pre_allocated = false;
        let alloc_size = if size == 0 { CHUNK_SIZE } else { size };
        let pbuf = Box::new([0u8; alloc_size]);
        item.pbuf = Some(pbuf);
        item.alloc_size = alloc_size;
    }

    item.header = BINN_MAGIC;
    item.writable = true;
    item.used_size = MAX_BINN_HEADER;
    item.type_ = type_;
    item.dirty = true;

    Ok(())
}

/// Converts the value in the `Binn` structure to a string representation.
///
/// # Arguments
/// * `value` - A reference to the `Binn` structure.
///
/// # Returns
/// Returns `Ok(String)` containing the string representation of the value if successful,
/// otherwise returns `Err(&'static str)` if the value cannot be converted.
pub fn binn_get_str(value: &Binn) -> Result<String, &'static str> {
    if value.type_ == BINN_STRING {
        if let Some(ptr) = &value.ptr {
            if let Some(s) = ptr.downcast_ref::<String>() {
                return Ok(s.clone());
            }
        }
        return Err("Invalid string pointer");
    }

    match type_family(value.type_) {
        BINN_FAMILY_INT => {
            let vint = copy_int_value(value.ptr.as_ref().ok_or("Invalid pointer")?, &mut 0, value.type_, BINN_INT64)?;
            Ok(format!("{}", vint))
        }
        BINN_FAMILY_FLOAT => {
            let vdouble = match value.type_ {
                BINN_FLOAT => value.vfloat as f64,
                BINN_DOUBLE => value.vdouble,
                _ => return Err("Invalid float type"),
            };
            Ok(format!("{}", vdouble))
        }
        BINN_FAMILY_BOOL => {
            Ok(if value.vbool { "true".to_string() } else { "false".to_string() })
        }
        _ => Err("Unsupported type"),
    }
}

/// 根据 Binn 对象的存储类型返回相应的值指针。
///
/// # 参数
/// - `self`: 当前 Binn 对象的引用。
///
/// # 返回值
/// 返回一个指向存储值的指针，如果存储类型为 `BINN_STORAGE_NOBYTES`、`BINN_STORAGE_WORD`、`BINN_STORAGE_DWORD` 或 `BINN_STORAGE_QWORD`，则返回 `vint32` 的引用；否则返回 `ptr`。
pub fn store_value(&self) -> Option<&dyn std::any::Any> {
    match binn_get_read_storage(self.type_).unwrap() {
        BINN_STORAGE_NOBYTES | BINN_STORAGE_WORD | BINN_STORAGE_DWORD | BINN_STORAGE_QWORD => {
            Some(&self.value)
        }
        _ => self.ptr.as_ref().map(|ptr| ptr.as_ref()),
    }
}

/// Extracts a double-precision floating-point value from the binary data structure.
///
/// # Arguments
/// * `pfloat` - A mutable reference to store the extracted floating-point value.
///
/// # Returns
/// Returns `Ok(())` if the extraction was successful, otherwise returns an error message.
pub fn binn_get_double(&self, pfloat: &mut f64) -> Result<(), &'static str> {
    if pfloat.is_null() {
        return Err("Invalid pointer");
    }

    match type_family(self.type_) {
        BINN_FAMILY_INT => {
            let vint = self.copy_int_value(BINN_INT64)?;
            *pfloat = vint as f64;
            Ok(())
        }
        _ => {
            match self.type_ {
                BINN_FLOAT => {
                    *pfloat = self.vfloat as f64;
                    Ok(())
                }
                BINN_DOUBLE => {
                    *pfloat = self.vdouble;
                    Ok(())
                }
                BINN_STRING => {
                    let s = unsafe { std::ffi::CStr::from_ptr(self.ptr as *const i8) };
                    let s = s.to_str().map_err(|_| "Invalid string")?;
                    if is_integer(s) {
                        *pfloat = s.parse::<i64>().map_err(|_| "Invalid integer")? as f64;
                    } else if is_float(s) {
                        *pfloat = s.parse::<f64>().map_err(|_| "Invalid float")?;
                    } else {
                        return Err("Invalid string format");
                    }
                    Ok(())
                }
                BINN_BOOL => {
                    *pfloat = self.vbool as i64 as f64;
                    Ok(())
                }
                _ => Err("Unsupported type"),
            }
        }
    }
}

/// Extracts a 64-bit integer from a binary data structure.
///
/// # Arguments
/// * `value` - A reference to the binary data structure.
/// * `pint` - A mutable reference to store the extracted 64-bit integer.
///
/// # Returns
/// Returns `Ok(())` if the extraction is successful, otherwise returns an error message.
pub fn binn_get_int64(value: &Binn, pint: &mut i64) -> Result<(), &'static str> {
    if value.ptr.is_none() || pint.is_null() {
        return Err("Invalid input");
    }

    match type_family(value.type_) {
        BINN_FAMILY_INT => {
            copy_int_value(value.ptr.as_ref().unwrap(), pint, value.type_, BINN_INT64)
        }
        BINN_FAMILY_FLOAT => {
            match value.type_ {
                BINN_FLOAT => {
                    if value.vfloat < i64::MIN as f32 || value.vfloat > i64::MAX as f32 {
                        return Err("Value out of range");
                    }
                    *pint = roundval(value.vfloat as f64);
                    Ok(())
                }
                BINN_DOUBLE => {
                    if value.vdouble < i64::MIN as f64 || value.vdouble > i64::MAX as f64 {
                        return Err("Value out of range");
                    }
                    *pint = roundval(value.vdouble);
                    Ok(())
                }
                _ => Err("Unsupported type"),
            }
        }
        BINN_FAMILY_STRING => {
            let s = unsafe { std::ffi::CStr::from_ptr(value.ptr.as_ref().unwrap() as *const u8 as *const i8) };
            let s = s.to_str().map_err(|_| "Invalid string")?;
            if is_integer(s) {
                *pint = s.parse::<i64>().map_err(|_| "Invalid integer")?;
                Ok(())
            } else if is_float(s) {
                *pint = roundval(s.parse::<f64>().map_err(|_| "Invalid float")?);
                Ok(())
            } else {
                Err("Invalid number format")
            }
        }
        BINN_FAMILY_BOOL => {
            *pint = value.vbool as i64;
            Ok(())
        }
        _ => Err("Unsupported type"),
    }
}

/// Checks if a string represents a boolean value and converts it to a boolean.
///
/// # Arguments
/// * `s` - The string to check.
/// * `pbool` - A mutable reference to store the boolean value.
///
/// # Returns
/// Returns `Ok(())` if the string represents a valid boolean value, otherwise returns `Err(())`.
pub fn is_bool_str(s: &str, pbool: &mut bool) -> Result<(), ()> {
    if s.is_empty() {
        return Err(());
    }

    match s.to_lowercase().as_str() {
        "true" | "yes" | "on" | "1" => {
            *pbool = true;
            Ok(())
        }
        "false" | "no" | "off" | "0" => {
            *pbool = false;
            Ok(())
        }
        _ => {
            if let Ok(vint) = s.parse::<i64>() {
                *pbool = vint != 0;
                Ok(())
            } else if let Ok(vdouble) = s.parse::<f64>() {
                *pbool = vdouble != 0.0;
                Ok(())
            } else {
                Err(())
            }
        }
    }
}

/// Extracts a boolean value from a binary data structure.
///
/// # Arguments
/// * `value` - The binary data structure to extract the boolean value from.
/// * `pbool` - A mutable reference to store the extracted boolean value.
///
/// # Returns
/// Returns `Ok(())` if the boolean value was successfully extracted, otherwise returns `Err(())`.
pub fn binn_get_bool(value: &Binn, pbool: &mut bool) -> Result<(), ()> {
    if value.ptr.is_none() || pbool.is_none() {
        return Err(());
    }

    match type_family(value.type_) {
        BINN_FAMILY_INT => {
            let vint = copy_int_value(value.ptr.unwrap(), BINN_INT64)?;
            *pbool = vint != 0;
            Ok(())
        }
        _ => match value.type_ {
            BINN_BOOL => {
                *pbool = value.vbool;
                Ok(())
            }
            BINN_FLOAT => {
                *pbool = value.vfloat != 0.0;
                Ok(())
            }
            BINN_DOUBLE => {
                *pbool = value.vdouble != 0.0;
                Ok(())
            }
            BINN_STRING => {
                let s = unsafe { std::ffi::CStr::from_ptr(value.ptr.unwrap() as *const i8) };
                is_bool_str(s.to_str().unwrap(), pbool)
            }
            _ => Err(()),
        },
    }
}


/// Extracts a 32-bit integer from the binary data structure.
///
/// # Arguments
/// * `pint` - A mutable reference to store the extracted 32-bit integer.
///
/// # Returns
/// Returns `Ok(())` if the extraction is successful, otherwise returns an error message.
pub fn get_int32(&self, pint: &mut i32) -> Result<(), &'static str> {
    if self.ptr.is_none() || pint.is_null() {
        return Err("Invalid input");
    }

    match type_family(self.type_) {
        BINN_FAMILY_INT => {
            copy_int_value(self.ptr.as_ref().unwrap(), pint, self.type_, BINN_INT32)
        }
        _ => {
            match self.type_ {
                BINN_FLOAT => {
                    if self.vfloat < i32::MIN as f32 || self.vfloat > i32::MAX as f32 {
                        return Err("Value out of range");
                    }
                    *pint = roundval(self.vfloat);
                    Ok(())
                }
                BINN_DOUBLE => {
                    if self.vdouble < i32::MIN as f64 || self.vdouble > i32::MAX as f64 {
                        return Err("Value out of range");
                    }
                    *pint = roundval(self.vdouble);
                    Ok(())
                }
                BINN_STRING => {
                    let s = unsafe { std::ffi::CStr::from_ptr(self.ptr.as_ref().unwrap() as *const u8 as *const i8) };
                    let s = s.to_str().map_err(|_| "Invalid string")?;
                    if is_integer(s) {
                        *pint = s.parse::<i32>().map_err(|_| "Invalid integer")?;
                        Ok(())
                    } else if is_float(s) {
                        *pint = roundval(s.parse::<f64>().map_err(|_| "Invalid float")?);
                        Ok(())
                    } else {
                        Err("Invalid number format")
                    }
                }
                BINN_BOOL => {
                    *pint = self.vbool as i32;
                    Ok(())
                }
                _ => Err("Unsupported type"),
            }
        }
    }
}

/// Initializes the memory pointed to by `pvalue` to zero based on the given type.
///
/// # Arguments
/// * `pvalue` - A mutable reference to the memory to be zeroed.
/// * `type_` - The type of the data, used to determine the storage size.
pub fn zero_value(pvalue: &mut [u8], type_: i32) {
    match binn_get_read_storage(type_) {
        Ok(BINN_STORAGE_NOBYTES) => {},
        Ok(BINN_STORAGE_BYTE) => pvalue[0] = 0,
        Ok(BINN_STORAGE_WORD) => {
            let value = &mut pvalue[..2];
            value.copy_from_slice(&[0, 0]);
        },
        Ok(BINN_STORAGE_DWORD) => {
            let value = &mut pvalue[..4];
            value.copy_from_slice(&[0, 0, 0, 0]);
        },
        Ok(BINN_STORAGE_QWORD) => {
            let value = &mut pvalue[..8];
            value.copy_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
        },
        Ok(BINN_STORAGE_BLOB) | Ok(BINN_STORAGE_STRING) | Ok(BINN_STORAGE_CONTAINER) => {
            // For complex types, set the pointer to None
            *pvalue = [0; 8]; // Assuming pointer size is 8 bytes
        },
        _ => {},
    }
}

/// Searches for a key in binary data.
///
/// # Arguments
/// * `data` - A slice of binary data to search in.
/// * `header_size` - The size of the header in the binary data.
/// * `key` - The key to search for.
///
/// # Returns
/// Returns `Some(&[u8])` with the data of the matching key if found, otherwise `None`.
pub fn search_for_key(data: &[u8], header_size: usize, key: &str) -> Option<&[u8]> {
    let key_len = key.len();
    let mut p = &data[header_size..];
    let plimit = &data[data.len() - 1];

    while p.as_ptr() <= plimit.as_ptr() {
        let len = p[0] as usize;
        p = &p[1..];

        if len > 0 {
            if p.len() >= len && key_len == len && p[..len].eq_ignore_ascii_case(key.as_bytes()) {
                return Some(&p[len..]);
            }
            p = &p[len..];
        } else if key_len == 0 {
            return Some(p);
        }

        p = match advance_data_pos(p, plimit) {
            Some(new_p) => new_p,
            None => break,
        };
    }

    None
}

/// Searches for a specific ID in a binary data slice.
///
/// # Arguments
/// * `data` - A slice of binary data to search in.
/// * `header_size` - The size of the header in the binary data.
/// * `size` - The total size of the binary data.
/// * `numitems` - The number of items in the binary data.
/// * `id` - The ID to search for.
///
/// # Returns
/// Returns `Some(&[u8])` with the data of the matching item if found, otherwise `None`.
pub fn search_for_id(data: &[u8], header_size: usize, size: usize, numitems: usize, id: i32) -> Option<&[u8]> {
    let base = data.as_ptr();
    let plimit = unsafe { base.add(size) };
    let mut p = unsafe { base.add(header_size) };

    for _ in 0..numitems {
        let int32 = read_map_id(unsafe { std::slice::from_raw_parts(p, (plimit as usize) - (p as usize)) })?;
        if int32 == id {
            return Some(unsafe { std::slice::from_raw_parts(p, (plimit as usize) - (p as usize)) });
        }
        p = advance_data_pos(unsafe { std::slice::from_raw_parts(p, (plimit as usize) - (p as usize)) }, unsafe { std::slice::from_raw_parts(plimit, 0) })?.as_ptr();
        if p < base {
            break;
        }
    }

    None
}

/// 从二进制对象中提取与指定键关联的 8 位有符号整数值。
///
/// # 参数
/// - `obj`: 指向二进制对象的引用。
/// - `key`: 要查找的键名。
///
/// # 返回值
/// 返回一个 `Result<i8, &'static str>`，表示提取的整数值或错误信息。
pub fn binn_object_int8(obj: &Binn, key: &str) -> Result<i8, &'static str> {
    let mut value: i8 = 0;
    binn_object_get(obj, key, BINN_INT8, &mut value, None)?;
    Ok(value)
}

/// 从二进制对象中提取与指定键关联的列表数据。
///
/// # 参数
/// - `obj`: 指向二进制对象的引用。
/// - `key`: 要查找的键名。
///
/// # 返回值
/// 返回一个 `Option<&BinnList>`，表示与键名关联的列表数据。如果找不到键或类型不匹配，则返回 `None`。
pub fn binn_object_list(obj: &Binn, key: &str) -> Option<&BinnList> {
    let mut value = None;
    binn_object_get(obj, key, BINN_LIST, &mut value, None);
    value
}

/// 从二进制对象中提取与指定键关联的无符号32位整数值。
///
/// # 参数
/// - `key`: 键名，用于查找对应的值。
///
/// # 返回值
/// 返回 `Result<u32, &'static str>`，表示提取的无符号32位整数值或错误信息。
pub fn binn_object_uint32(&self, key: &str) -> Result<u32, &'static str> {
    let mut value: u32 = 0;
    self.binn_object_get(key, BINN_UINT32, &mut value, None)?;
    Ok(value)
}

/// 从二进制对象中获取与指定键关联的列表数据。
///
/// # 参数
/// - `key`: 键名，用于查找对应的列表数据。
/// - `pvalue`: 用于存储结果的指针。
///
/// # 返回值
/// 返回 `Ok(())` 如果操作成功，否则返回 `Err(&'static str)`。
pub fn binn_object_get_list(&self, key: &str, pvalue: &mut Option<Box<Binn>>) -> Result<(), &'static str> {
    self.binn_object_get(key, BINN_LIST, pvalue, None)
}

/// Sets a key-value pair in the Binn object and automatically frees the value object.
///
/// # Arguments
/// * `obj` - A mutable reference to the Binn object.
/// * `key` - The key string.
/// * `value` - The value object to be set and freed.
///
/// # Returns
/// Returns `Ok(true)` if the operation was successful, otherwise returns an error.
pub fn binn_object_set_new(obj: &mut Binn, key: &str, value: Option<Box<Binn>>) -> Result<bool, &'static str> {
    let retval = obj.set_value(key, value.as_ref().ok_or("Value cannot be null")?)?;
    if let Some(v) = value {
        drop(v); // Automatically frees the memory when `v` goes out of scope
    }
    Ok(retval)
}

/// 从二进制对象中提取与指定键关联的双精度浮点数值。
///
/// # 参数
/// - `key`: 键名，用于查找对应的值。
///
/// # 返回值
/// 返回 `Result<f64, &'static str>`，表示提取的双精度浮点数值或错误信息。
pub fn binn_object_double(&self, key: &str) -> Result<f64, &'static str> {
    let mut value = 0.0;
    self.binn_object_get(key, BINN_FLOAT64, &mut value, None)?;
    Ok(value)
}

/// Extracts a binary large object (BLOB) from the binary object associated with the specified key.
///
/// # Arguments
/// * `key` - The key associated with the BLOB data.
/// * `size` - A mutable reference to store the size of the BLOB data.
///
/// # Returns
/// Returns `Some(&[u8])` containing the BLOB data if the key exists, otherwise returns `None`.
pub fn binn_object_blob(&self, key: &str, size: &mut usize) -> Option<&[u8]> {
    self.binn_object_get(key, BINN_BLOB, size)
}

/// Extracts a floating-point value from a binary object associated with the given key.
///
/// # Arguments
/// * `key` - The key associated with the floating-point value.
///
/// # Returns
/// Returns `Ok(f32)` containing the floating-point value if successful, otherwise returns `Err(&'static str)`.
pub fn binn_object_float(&self, key: &str) -> Result<f32, &'static str> {
    let mut value = 0.0f32;
    self.binn_object_get(key, BINN_FLOAT32, &mut value, None)?;
    Ok(value)
}

/// Extracts a boolean value from a binary object associated with the specified key.
///
/// # Arguments
/// * `key` - The key associated with the boolean value.
///
/// # Returns
/// Returns `Ok(bool)` if the boolean value was successfully extracted, otherwise returns `Err(&'static str)`.
pub fn binn_object_bool(&self, key: &str) -> Result<bool, &'static str> {
    let mut value = false;
    self.binn_object_get(key, BINN_BOOL, &mut value, None)?;
    Ok(value)
}

/// Extracts a double-precision floating-point value from the binary object.
///
/// # Arguments
/// * `key` - The key associated with the value to be extracted.
/// * `pvalue` - A mutable reference to store the extracted double-precision floating-point value.
///
/// # Returns
/// Returns `Ok(())` if the extraction was successful, otherwise returns an error message.
pub fn binn_object_get_double(&self, key: &str, pvalue: &mut f64) -> Result<(), &'static str> {
    self.binn_object_get(key, BINN_FLOAT64, pvalue, None)
}

/// Extracts a map from the binary object associated with the specified key.
///
/// # Arguments
/// * `key` - The key to search for.
/// * `pvalue` - A mutable reference to store the extracted map.
///
/// # Returns
/// Returns `Ok(())` if the map was successfully extracted, otherwise returns an error message.
pub fn binn_object_get_map(&self, key: &str, pvalue: &mut Option<Box<BinnMap>>) -> Result<(), &'static str> {
    self.binn_object_get(key, BINN_MAP, pvalue, None)
}

/// 从二进制对象中提取与指定键关联的字符串值。
///
/// # 参数
/// - `key`: 要查找的键名。
///
/// # 返回值
/// 返回 `Some(String)` 包含与键名关联的字符串值，如果键不存在或值不是字符串类型，则返回 `None`。
pub fn binn_object_str(&self, key: &str) -> Option<String> {
    let mut value: *mut i8 = std::ptr::null_mut();
    if binn_object_get(self, key, BINN_STRING, &mut value, std::ptr::null_mut()) {
        if !value.is_null() {
            let c_str = unsafe { std::ffi::CStr::from_ptr(value) };
            return Some(c_str.to_string_lossy().into_owned());
        }
    }
    None
}

/// 从二进制对象中提取与指定键关联的32位整数值。
///
/// # 参数
/// - `key`: 要查找的键名。
///
/// # 返回值
/// 返回 `Result<i32, &'static str>`，表示提取的整数值或错误信息。
pub fn binn_object_int32(&self, key: &str) -> Result<i32, &'static str> {
    let mut value = 0;
    self.binn_object_get(key, BINN_INT32, &mut value, None)?;
    Ok(value)
}

/// Extracts an unsigned 16-bit integer from a binary object.
///
/// # Arguments
/// * `key` - The key to look up in the binary object.
///
/// # Returns
/// Returns `Ok(u16)` if the value is successfully extracted, otherwise returns `Err(&'static str)`.
pub fn binn_object_uint16(&self, key: &str) -> Result<u16, &'static str> {
    let mut value: u16 = 0;
    self.binn_object_get(key, BINN_UINT16, &mut value, None)?;
    Ok(value)
}

/// 检查二进制对象中是否存在一个指定键，并且该键对应的值为 `NULL`。
///
/// # 参数
/// - `obj`: 指向二进制对象的引用。
/// - `key`: 要查找的键名。
///
/// # 返回值
/// 返回 `true` 如果键存在且值为 `NULL`，否则返回 `false`。
pub fn binn_object_null(obj: &Binn, key: &str) -> bool {
    binn_object_get(obj, key, BINN_NULL, None, None).is_ok()
}

/// 从二进制对象中获取与指定键关联的 64 位整数值。
///
/// # 参数
/// - `obj`: 指向二进制对象的引用。
/// - `key`: 键名，用于查找对应的值。
/// - `pvalue`: 用于存储提取的 64 位整数值的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功提取并存储了整数值，否则返回 `Err("error message")`。
pub fn binn_object_get_int64(obj: &Binn, key: &str, pvalue: &mut i64) -> Result<(), &'static str> {
    binn_object_get(obj, key, BINN_INT64, pvalue, None)
}

/// 从二进制对象中提取与指定键关联的无符号64位整数值。
///
/// # 参数
/// - `key`: 键名，用于查找对应的值。
///
/// # 返回值
/// 返回 `Result<u64, &'static str>`，表示提取的无符号64位整数值或错误信息。
pub fn binn_object_uint64(&self, key: &str) -> Result<u64, &'static str> {
    let mut value: u64 = 0;
    self.binn_object_get(key, BINN_UINT64, &mut value, None)?;
    Ok(value)
}

/// 从二进制对象中提取与指定键关联的二进制数据块（blob）。
///
/// # 参数
/// - `obj`: 指向二进制对象的引用。
/// - `key`: 要查找的键名。
/// - `pvalue`: 用于存储数据块的指针。
/// - `psize`: 用于存储数据块大小的指针。
///
/// # 返回值
/// 返回 `Ok(())` 如果操作成功，否则返回 `Err(&'static str)`。
pub fn binn_object_get_blob(obj: &Binn, key: &str, pvalue: &mut Option<&[u8]>, psize: &mut usize) -> Result<(), &'static str> {
    binn_object_get(obj, key, BINN_BLOB, pvalue, psize)
}

/// 从二进制对象中获取一个与指定键关联的 32 位整数值。
///
/// # 参数
/// - `obj`: 指向二进制对象的引用。
/// - `key`: 要查找的键名。
/// - `pvalue`: 用于存储结果的 32 位整数的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果操作成功，否则返回 `Err("error message")`。
pub fn binn_object_get_int32(obj: &Binn, key: &str, pvalue: &mut i32) -> Result<(), &'static str> {
    binn_object_get(obj, key, BINN_INT32, pvalue, None)
}

/// 从二进制对象中提取与指定键关联的 16 位有符号整数值。
///
/// # 参数
/// - `key`: 要查找的键名。
///
/// # 返回值
/// 返回一个 `Result<i16, &'static str>`，表示提取的整数值或错误信息。
pub fn binn_object_int16(&self, key: &str) -> Result<i16, &'static str> {
    let mut value: i16 = 0;
    self.binn_object_get(key, BINN_INT16, &mut value, None)?;
    Ok(value)
}

/// 从二进制对象中提取与指定键关联的 64 位无符号整数值。
///
/// # 参数
/// - `key`: 键名，用于查找对应的值。
/// - `pvalue`: 用于存储提取的 64 位无符号整数值的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果操作成功，否则返回 `Err(&'static str)`。
pub fn binn_object_get_uint64(&self, key: &str, pvalue: &mut u64) -> Result<(), &'static str> {
    self.binn_object_get(key, BINN_UINT64, pvalue, None)
}

/// 从二进制对象中获取一个 8 位有符号整数。
///
/// # 参数
/// - `key`: 键名，用于查找对应的值。
///
/// # 返回值
/// 返回 `Result<i8, &'static str>`，表示提取的整数值或错误信息。
pub fn binn_object_get_int8(&self, key: &str) -> Result<i8, &'static str> {
    let mut value: i8 = 0;
    self.binn_object_get(key, BINN_INT8, &mut value, None)?;
    Ok(value)
}

/// 从二进制对象中提取一个无符号16位整数（`uint16`）。
///
/// # 参数
/// - `key`: 键名，用于查找对应的值。
///
/// # 返回值
/// 返回 `Result<u16, &'static str>`，表示提取的无符号16位整数值或错误信息。
pub fn binn_object_get_uint16(&self, key: &str) -> Result<u16, &'static str> {
    let mut value: u16 = 0;
    self.binn_object_get(key, BINN_UINT16, &mut value, None)?;
    Ok(value)
}

/// 从二进制对象中提取嵌套的二进制对象。
///
/// # 参数
/// - `obj`: 指向二进制对象的引用。
/// - `key`: 要查找的键名。
/// - `pvalue`: 用于存储结果的指针。
///
/// # 返回值
/// 返回 `Ok(())` 如果操作成功，否则返回 `Err(&'static str)`。
pub fn binn_object_get_object(obj: &Binn, key: &str, pvalue: &mut Option<Box<Binn>>) -> Result<(), &'static str> {
    binn_object_get(obj, key, BINN_OBJECT, pvalue, None)
}

/// 从二进制对象中提取与指定键关联的映射数据。
///
/// # 参数
/// - `obj`: 指向二进制对象的引用。
/// - `key`: 要查找的键名。
///
/// # 返回值
/// 返回一个 `Option<&BinnMap>`，表示与键名关联的映射数据。如果找不到键或类型不匹配，则返回 `None`。
pub fn binn_object_map(obj: &Binn, key: &str) -> Option<&BinnMap> {
    let mut value = None;
    binn_object_get(obj, key, BINN_MAP, &mut value, None);
    value
}

/// 从二进制对象中提取与指定键关联的无符号8位整数值。
///
/// # 参数
/// - `key`: 键名，用于查找对应的值。
///
/// # 返回值
/// 返回 `Result<u8, &'static str>`，表示提取的无符号8位整数值或错误信息。
pub fn binn_object_uint8(&self, key: &str) -> Result<u8, &'static str> {
    let mut value: u8 = 0;
    self.binn_object_get(key, BINN_UINT8, &mut value, None)?;
    Ok(value)
}

/// 从二进制对象中提取一个 16 位整数（`int16`）。
///
/// # 参数
/// - `key`: 键名，用于查找对应的值。
/// - `value`: 用于存储提取的 16 位整数的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功提取并存储了值，否则返回错误信息。
pub fn binn_object_get_int16(&self, key: &str, value: &mut i16) -> Result<(), &'static str> {
    self.binn_object_get(key, BINN_INT16, value, None)
}

/// Extracts a floating-point value from a binary object associated with the given key.
///
/// # Arguments
/// * `key` - The key associated with the floating-point value.
/// * `pvalue` - A mutable reference to store the extracted floating-point value.
///
/// # Returns
/// Returns `Ok(())` if the extraction is successful, otherwise returns an error message.
pub fn binn_object_get_float(&self, key: &str, pvalue: &mut f32) -> Result<(), &'static str> {
    self.binn_object_get(key, BINN_FLOAT32, pvalue, None)
}

/// 从二进制对象中提取一个无符号32位整数值。
///
/// # 参数
/// - `key`: 键名，用于查找对应的值。
/// - `pvalue`: 用于存储提取的32位无符号整数值的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果操作成功，否则返回 `Err(&'static str)`。
pub fn binn_object_get_uint32(&self, key: &str, pvalue: &mut u32) -> Result<(), &'static str> {
    self.binn_object_get(key, BINN_UINT32, pvalue, None)
}

/// 从二进制对象中提取与指定键关联的 64 位有符号整数值。
///
/// # 参数
/// - `key`: 键名，用于查找对应的值。
///
/// # 返回值
/// 返回 `Result<i64, &'static str>`，表示提取的整数值或错误信息。
pub fn binn_object_int64(&self, key: &str) -> Result<i64, &'static str> {
    let mut value: i64 = 0;
    self.binn_object_get(key, BINN_INT64, &mut value, None)?;
    Ok(value)
}

/// Creates a new binary list data structure.
///
/// # Arguments
/// * `list` - A mutable reference to the `Binn` structure to be initialized as a list.
///
/// # Returns
/// Returns `Ok(())` if the list was successfully created, otherwise returns an error message.
pub fn binn_create_list(list: &mut Binn) -> Result<(), &'static str> {
    binn_create(list, BINN_LIST, 0, None)
}

/// Creates a new binary map structure.
///
/// # Arguments
/// * `map` - A mutable reference to the `Binn` structure to be initialized as a map.
///
/// # Returns
/// Returns `Ok(())` if the map was successfully created, otherwise returns an error message.
pub fn binn_create_map(map: &mut Binn) -> Result<(), &'static str> {
    binn_create(map, BINN_MAP, 0, None)
}

/// Creates a new binary object of type `BINN_OBJECT`.
///
/// # Arguments
/// * `object` - A mutable reference to the `Binn` structure to be initialized.
///
/// # Returns
/// Returns `Ok(())` if the object was successfully created, otherwise returns an error message.
pub fn binn_create_object(object: &mut Binn) -> Result<(), &'static str> {
    binn_create(object, BINN_OBJECT, 0, None)
}

/// Returns a pointer to the data stored in the Binn structure.
///
/// # Arguments
/// * `self` - A mutable reference to the Binn structure.
///
/// # Returns
/// Returns `Ok(Some(&mut dyn std::any::Any))` if the data is successfully retrieved,
/// `Ok(None)` if the data is a buffer, or an error message if the pointer is invalid.
pub fn binn_ptr(&mut self) -> Result<Option<&mut dyn std::any::Any>, &'static str> {
    match binn_get_ptr_type(Some(&self.header)) {
        Some(BINN_STRUCT) => {
            if self.writable && self.dirty {
                self.save_header()?;
            }
            Ok(self.ptr.as_mut().map(|ptr| ptr.as_mut()))
        }
        Some(BINN_BUFFER) => Ok(None),
        _ => Err("Invalid pointer type"),
    }
}

/// 创建一个新的 `Binn` 结构体实例。
///
/// # 参数
/// - `type_`: 数据类型标识符。
/// - `size`: 数据大小。
/// - `pointer`: 可选的内存指针。
///
/// # 返回值
/// 返回 `Result<Box<Binn>, &'static str>`，表示成功时返回 `Box<Binn>`，失败时返回错误信息。
pub fn binn_new(type_: i32, size: usize, pointer: Option<Box<[u8]>>) -> Result<Box<Binn>, &'static str> {
    let mut item = Box::new(Binn::default());
    binn_create(&mut item, type_, size, pointer)?;
    item.allocated = true;
    Ok(item)
}

/// 获取二进制数据结构中的元素数量。
///
/// # 参数
/// - `ptr`: 指向二进制数据的引用。
///
/// # 返回值
/// 返回 `Ok(i32)` 表示元素数量，如果指针类型无效则返回 `Err(&'static str)`。
pub fn binn_count(ptr: &[u8]) -> Result<i32, &'static str> {
    match binn_get_ptr_type(Some(&ptr[0])) {
        Some(BINN_STRUCT) => {
            let item = unsafe { &*(ptr.as_ptr() as *const Binn) };
            Ok(item.count as i32)
        }
        Some(BINN_BUFFER) => {
            let nitems = binn_buf_count(ptr)?;
            Ok(nitems)
        }
        _ => Err("Invalid pointer type"),
    }
}

/// 获取二进制缓冲区中的元素数量。
///
/// # 参数
/// - `pbuf`: 指向二进制缓冲区的引用。
///
/// # 返回值
/// 返回 `Ok(i32)` 表示元素数量，如果头部信息无效则返回 `Err(&'static str)`。
pub fn binn_buf_count(pbuf: &[u8]) -> Result<i32, &'static str> {
    let mut nitems = 0;
    is_valid_binn_header(pbuf, None, Some(&mut nitems), None, None)?;
    Ok(nitems)
}

/// 获取二进制数据的大小。
///
/// # 参数
/// - `ptr`: 指向二进制数据的指针。
///
/// # 返回值
/// 返回二进制数据的大小，如果指针无效则返回 0。
pub fn binn_size(ptr: Option<&Binn>) -> usize {
    match binn_get_ptr_type(ptr.map(|p| &p.header)) {
        Some(BINN_STRUCT) => {
            let item = ptr.unwrap();
            if item.writable && item.dirty {
                item.save_header().unwrap();
            }
            item.size
        }
        Some(BINN_BUFFER) => binn_buf_size(ptr.unwrap().pbuf.as_ref().unwrap()),
        _ => 0,
    }
}

/// 获取二进制缓冲区的大小。
///
/// # 参数
/// - `pbuf`: 指向二进制缓冲区的指针。
///
/// # 返回值
/// 返回二进制缓冲区的大小，如果缓冲区无效则返回 0。
fn binn_buf_size(pbuf: &[u8]) -> usize {
    if let Ok((_, _, size, _)) = is_valid_binn_header(pbuf, None, None, None, None) {
        size
    } else {
        0
    }
}

/// Determines the type of binary data in a buffer.
///
/// # Arguments
/// * `pbuf` - A reference to the binary data buffer.
///
/// # Returns
/// Returns `Ok(type)` if the buffer contains valid binary data, otherwise returns `Err(INVALID_BINN)`.
pub fn binn_buf_type(pbuf: &[u8]) -> Result<i32, u32> {
    let mut type_ = 0;
    if is_valid_binn_header(pbuf, Some(&mut type_), None, None, None).is_ok() {
        Ok(type_)
    } else {
        Err(INVALID_BINN)
    }
}

/// Determines the type of binary data pointed to by the given pointer.
///
/// # Arguments
/// * `ptr` - A reference to the binary data to check.
///
/// # Returns
/// Returns the type of the binary data if successful, otherwise returns `-1`.
pub fn binn_type(ptr: Option<&u32>) -> i32 {
    match binn_get_ptr_type(ptr) {
        Some(BINN_STRUCT) => {
            let item = unsafe { &*(ptr.unwrap() as *const Binn) };
            item.type_
        }
        Some(BINN_BUFFER) => {
            let pbuf = unsafe { std::slice::from_raw_parts(ptr.unwrap() as *const u8, 1) };
            binn_buf_type(pbuf).unwrap_or(-1)
        }
        _ => -1,
    }
}

/// Creates a new binary data object containing the specified value.
///
/// # Arguments
/// * `type_` - The type of the data (e.g., `BINN_STRING`, `BINN_BLOB`).
/// * `pvalue` - A reference to the data to be stored.
/// * `size` - The size of the data in bytes.
/// * `freefn` - An optional function to free the data when the object is dropped.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` if the object was successfully created, otherwise returns `Err(&'static str)`.
pub fn binn_value(type_: i32, pvalue: &[u8], size: usize, freefn: Option<fn(Box<dyn std::any::Any>)>) -> Result<Box<Binn>, &'static str> {
    let storage_type = binn_get_type_info(type_)?.0;
    let mut item = binn_alloc_item().ok_or("Failed to allocate item")?;
    item.type_ = type_;

    match storage_type {
        BINN_STORAGE_NOBYTES => {},
        BINN_STORAGE_STRING => {
            let size = if size == 0 { pvalue.len() + 1 } else { size };
            if freefn == Some(BINN_TRANSIENT) {
                item.ptr = Some(Box::new(pvalue.to_vec()) as Box<dyn std::any::Any>);
                item.freefn = Some(free_fn);
                item.size = size - 1;
            } else {
                item.ptr = Some(Box::new(pvalue) as Box<dyn std::any::Any>);
                item.freefn = freefn;
                item.size = size;
            }
        }
        BINN_STORAGE_BLOB | BINN_STORAGE_CONTAINER => {
            if freefn == Some(BINN_TRANSIENT) {
                item.ptr = Some(Box::new(pvalue.to_vec()) as Box<dyn std::any::Any>);
                item.freefn = Some(free_fn);
            } else {
                item.ptr = Some(Box::new(pvalue) as Box<dyn std::any::Any>);
                item.freefn = freefn;
            }
            item.size = size;
        }
        _ => {
            item.ptr = Some(Box::new(item.vint32) as Box<dyn std::any::Any>);
            copy_raw_value(pvalue, item.ptr.as_mut().unwrap(), storage_type)?;
        }
    }

    Ok(item)
}

impl Binn {
    /// Adds a value of a specified type to the binary data structure.
    ///
    /// # Arguments
    /// * `type_` - The type of the value to add.
    /// * `value` - A slice containing the value data.
    ///
    /// # Returns
    /// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
    pub fn add_value(&mut self, type_: i32, value: &[u8]) -> Result<(), &'static str> {
        let (storage_type, extra_type) = binn_get_type_info(type_)?;

        if value.is_empty() {
            match storage_type {
                BINN_STORAGE_NOBYTES => {},
                BINN_STORAGE_BLOB | BINN_STORAGE_STRING => {},
                _ => return Err("Invalid value pointer"),
            }
        }

        let mut compressed_value = value;
        if type_family(type_) == BINN_FAMILY_INT && !self.disable_int_compression {
            compressed_value = compress_int(&storage_type, &type_, value);
        }

        let size = match storage_type {
            BINN_STORAGE_NOBYTES => 0,
            BINN_STORAGE_BYTE => 1,
            BINN_STORAGE_WORD => 2,
            BINN_STORAGE_DWORD => 4,
            BINN_STORAGE_QWORD => 8,
            BINN_STORAGE_BLOB => {
                if value.len() < 0 {
                    return Err("Invalid size for BLOB");
                }
                value.len() + 4
            }
            BINN_STORAGE_STRING => {
                if value.len() < 0 {
                    return Err("Invalid size for string");
                }
                let len = if value.is_empty() {
                    value.len()
                } else {
                    value.len()
                };
                len + 5
            }
            BINN_STORAGE_CONTAINER => {
                if value.len() <= 0 {
                    return Err("Invalid size for container");
                }
                value.len()
            }
            _ => return Err("Invalid storage type"),
        };

        let arg_size = size + 2;
        self.check_allocation(arg_size)?;

        let p = &mut self.pbuf.as_mut().unwrap()[self.used_size..];

        if storage_type != BINN_STORAGE_CONTAINER {
            if type_ > 255 {
                let type16 = type_ as u16;
                copy_be16(&mut p[..2], &type16);
                self.used_size += 2;
            } else {
                p[0] = type_ as u8;
                self.used_size += 1;
            }
        }

        match storage_type {
            BINN_STORAGE_NOBYTES => {},
            BINN_STORAGE_BYTE => {
                p[0] = value[0];
                self.used_size += 1;
            }
            BINN_STORAGE_WORD => {
                copy_be16(&mut p[..2], &u16::from_ne_bytes([value[0], value[1]]));
                self.used_size += 2;
            }
            BINN_STORAGE_DWORD => {
                copy_be32(&mut p[..4], &u32::from_ne_bytes([value[0], value[1], value[2], value[3]]));
                self.used_size += 4;
            }
            BINN_STORAGE_QWORD => {
                copy_be64(&mut p[..8], &u64::from_ne_bytes([
                    value[0], value[1], value[2], value[3],
                    value[4], value[5], value[6], value[7],
                ]));
                self.used_size += 8;
            }
            BINN_STORAGE_BLOB | BINN_STORAGE_STRING => {
                if value.len() > 127 {
                    let int32 = (value.len() | 0x80000000) as u32;
                    copy_be32(&mut p[..4], &int32);
                    self.used_size += 4;
                } else {
                    p[0] = value.len() as u8;
                    self.used_size += 1;
                }
                p[..value.len()].copy_from_slice(value);
                if storage_type == BINN_STORAGE_STRING {
                    p[value.len()] = 0;
                    self.used_size += value.len() + 1;
                } else {
                    self.used_size += value.len();
                }
            }
            BINN_STORAGE_CONTAINER => {
                p[..value.len()].copy_from_slice(value);
                self.used_size += value.len();
            }
        }

        self.dirty = true;
        Ok(())
    }
}

impl Binn {
    /// Extracts and parses a value from binary data into the `Binn` structure.
    ///
    /// # Arguments
    /// * `p` - A slice of binary data to parse.
    ///
    /// # Returns
    /// Returns `Ok(())` if the value was successfully parsed and stored, otherwise returns an error message.
    pub fn get_value(&mut self, p: &[u8]) -> Result<(), &'static str> {
        if p.is_empty() {
            return Err("Empty binary data");
        }

        let mut p = p;
        let p2 = p; // Save for use with BINN_STORAGE_CONTAINER

        // Read the data type
        let byte = p[0];
        p = &p[1..];
        let storage_type = byte & BINN_STORAGE_MASK;
        let data_type = if byte & BINN_STORAGE_HAS_MORE != 0 {
            let next_byte = p[0];
            p = &p[1..];
            ((byte as i32) << 8) | (next_byte as i32)
        } else {
            byte as i32
        };

        self.type_ = data_type;

        match storage_type {
            BINN_STORAGE_NOBYTES => {},
            BINN_STORAGE_BYTE => {
                self.vuint8 = p[0];
                self.ptr = Some(Box::new(self.vuint8));
            },
            BINN_STORAGE_WORD => {
                let mut vint16 = 0;
                copy_be16(&mut vint16, &u16::from_be_bytes([p[0], p[1]]));
                self.vint16 = vint16;
                self.ptr = Some(Box::new(self.vint16));
                p = &p[2..];
            },
            BINN_STORAGE_DWORD => {
                let mut vint32 = 0;
                copy_be32(&mut vint32, &u32::from_be_bytes([p[0], p[1], p[2], p[3]]));
                self.vint32 = vint32;
                self.ptr = Some(Box::new(self.vint32));
                p = &p[4..];
            },
            BINN_STORAGE_QWORD => {
                let mut vint64 = 0;
                copy_be64(&mut vint64, &u64::from_be_bytes([p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]]));
                self.vint64 = vint64;
                self.ptr = Some(Box::new(self.vint64));
                p = &p[8..];
            },
            BINN_STORAGE_BLOB | BINN_STORAGE_STRING => {
                let mut data_size = p[0] as usize;
                if data_size & 0x80 != 0 {
                    data_size = u32::from_be_bytes([p[0], p[1], p[2], p[3]]) as usize & 0x7FFFFFFF;
                    p = &p[4..];
                } else {
                    p = &p[1..];
                }
                self.size = data_size;
                self.ptr = Some(Box::new(p[..data_size].to_vec()));
                p = &p[data_size..];
            },
            BINN_STORAGE_CONTAINER => {
                self.ptr = Some(Box::new(p2.to_vec()));
                if is_valid_binn_header(p2, None, &mut self.count, &mut self.size, None).is_err() {
                    return Err("Invalid Binn header");
                }
            },
            _ => return Err("Invalid storage type"),
        }

        // Convert the returned value, if needed
        match self.type_ {
            BINN_TRUE => {
                self.type_ = BINN_BOOL;
                self.vbool = true;
                self.ptr = Some(Box::new(self.vbool));
            },
            BINN_FALSE => {
                self.type_ = BINN_BOOL;
                self.vbool = false;
                self.ptr = Some(Box::new(self.vbool));
            },
            _ => {},
        }

        Ok(())
    }
}

/// Creates a new binary list data structure.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` containing the new list if successful, otherwise returns an error message.
pub fn binn_list() -> Result<Box<Binn>, &'static str> {
    binn_new(BINN_LIST, 0, None)
}

/// Creates a new binary object of type `BINN_OBJECT`.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` containing the newly created binary object if successful,
/// otherwise returns an error message.
pub fn binn_object() -> Result<Box<Binn>, &'static str> {
    binn_new(BINN_OBJECT, 0, None)
}

/// Creates a new binary map data structure.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` containing the new map if successful, otherwise returns an error message.
pub fn binn_map() -> Result<Box<Binn>, &'static str> {
    binn_new(BINN_MAP, 0, None)
}

/// Initializes a binary data iterator for sequential reading.
///
/// # Arguments
/// * `iter` - A mutable reference to the `BinnIter` to be initialized.
/// * `ptr` - A reference to the binary data.
/// * `expected_type` - The expected type of the binary data.
///
/// # Returns
/// Returns `Ok(())` if the iterator is successfully initialized, otherwise returns an error message.
pub fn binn_iter_init(iter: &mut BinnIter, ptr: &[u8], expected_type: i32) -> Result<(), &'static str> {
    if ptr.is_empty() || iter.is_null() {
        return Err("Invalid input");
    }

    let mut type_ = 0;
    let mut count = 0;
    let mut size = 0;
    let mut header_size = 0;

    // Check the header
    is_valid_binn_header(ptr, Some(&mut type_), Some(&mut count), Some(&mut size), Some(&mut header_size))?;

    if type_ != expected_type {
        return Err("Type mismatch");
    }

    iter.plimit = unsafe { ptr.as_ptr().add(size - 1) };
    iter.pnext = unsafe { ptr.as_ptr().add(header_size) };
    iter.count = count;
    iter.current = 0;
    iter.type_ = type_;

    Ok(())
}

/// 释放或重置 `Binn` 结构体的内存资源。
///
/// 如果 `Binn` 是动态分配的，则返回 `None`，因为 Rust 会自动释放内存。
/// 如果 `Binn` 不是动态分配的，则将其内容重置为初始状态。
///
/// # 参数
/// - `self`: 当前 `Binn` 对象的可变引用。
///
/// # 返回值
/// 返回 `Option<&mut dyn std::any::Any>`，表示释放后的数据指针。
pub fn binn_release(&mut self) -> Option<&mut dyn std::any::Any> {
    if self.allocated {
        // Rust 会自动释放内存，无需手动调用 free_fn
        None
    } else {
        // 重置结构体内容
        *self = Binn {
            header: BINN_MAGIC,
            allocated: false,
            writable: false,
            dirty: false,
            pbuf: None,
            pre_allocated: false,
            alloc_size: 0,
            used_size: 0,
            type_: 0,
            ptr: None,
            size: 0,
            count: 0,
            freefn: None,
            value: BinnValue::Int8(0),
            disable_int_compression: false,
        };
        self.binn_ptr().ok()
    }
}

/// Creates a new binary data object containing a double-precision floating-point value.
///
/// # Arguments
/// * `value` - The double-precision floating-point value to be stored.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` if the object was successfully created, otherwise returns `Err(&'static str)`.
pub fn binn_double(value: f64) -> Result<Box<Binn>, &'static str> {
    binn_value(BINN_DOUBLE, &value, 0, None)
}

/// 将带符号的 8 位整数封装为 Binn 二进制数据格式。
///
/// # 参数
/// - `value`: 要封装的 8 位整数。
///
/// # 返回值
/// 返回 `Result<Box<Binn>, &'static str>`，表示成功时返回 `Box<Binn>`，失败时返回错误信息。
pub fn binn_int8(value: i8) -> Result<Box<Binn>, &'static str> {
    binn_value(BINN_INT8, &value, 0, None)
}

/// Copies an existing binary data structure.
///
/// # Arguments
/// * `old` - A reference to the existing binary data structure.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` containing the copied binary data structure if successful,
/// otherwise returns an error message.
pub fn binn_copy(old: &Binn) -> Result<Box<Binn>, &'static str> {
    let mut type_ = 0;
    let mut count = 0;
    let mut size = 0;
    let mut header_size = 0;

    let old_ptr = binn_ptr(old)?;
    if !is_valid_binn_header(old_ptr, Some(&mut type_), Some(&mut count), Some(&mut size), Some(&mut header_size)) {
        return Err("Invalid binary data header");
    }

    let mut item = binn_new(type_, size - header_size + MAX_BINN_HEADER, None)?;
    let dest = &mut item.pbuf.as_mut().unwrap()[MAX_BINN_HEADER..];
    dest.copy_from_slice(&old_ptr[header_size..size]);
    item.used_size = MAX_BINN_HEADER + size - header_size;
    item.count = count as usize;

    Ok(item)
}

/// 创建一个包含二进制数据的 Binn 对象。
///
/// # 参数
/// - `data`: 二进制数据。
/// - `freefn`: 可选的内存释放函数。
///
/// # 返回值
/// 返回 `Result<Box<Binn>, &'static str>`，表示成功时返回 `Box<Binn>`，失败时返回错误信息。
pub fn binn_blob(data: Vec<u8>, freefn: Option<Box<dyn FnOnce(Box<dyn std::any::Any>)>>) -> Result<Box<Binn>, &'static str> {
    binn_value(BINN_BLOB, &data, data.len(), freefn)
}

/// Creates a Binn object representing a null value.
///
/// # Returns
/// Returns a `Result<Box<Binn>, &'static str>` containing the Binn object if successful,
/// otherwise returns an error message.
#[inline(always)]
pub fn binn_null() -> Result<Box<Binn>, &'static str> {
    binn_value(BINN_NULL, &[], 0, None)
}

/// Creates a new Binn object containing a boolean value.
///
/// # Arguments
/// * `value` - The boolean value to be stored in the Binn object.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` if the object was successfully created, otherwise returns `Err(&'static str)`.
pub fn binn_bool(value: bool) -> Result<Box<Binn>, &'static str> {
    binn_value(BINN_BOOL, &value, 0, None)
}

/// Validates the binary data and checks its type, count, and size.
///
/// # Arguments
/// * `ptr` - A pointer to the binary data.
/// * `ptype` - An optional mutable reference to store the type of the binary data.
/// * `pcount` - An optional mutable reference to store the count of elements.
/// * `psize` - An optional mutable reference to store the size of the binary data.
///
/// # Returns
/// Returns `Ok(())` if the data is valid, otherwise returns an error message.
pub fn binn_is_valid(
    ptr: Option<&[u8]>,
    ptype: Option<&mut u8>,
    pcount: Option<&mut u32>,
    psize: Option<&mut usize>,
) -> Result<(), &'static str> {
    let pbuf = ptr.ok_or("Invalid pointer")?;
    let (type_, count, size, header_size) = is_valid_binn_header(pbuf, ptype, pcount, psize)?;

    if let Some(psize) = psize {
        if *psize > 0 && *psize != size {
            return Err("Size mismatch");
        }
    }
    if let Some(pcount) = pcount {
        if *pcount > 0 && *pcount != count {
            return Err("Count mismatch");
        }
    }
    if let Some(ptype) = ptype {
        if *ptype != 0 && *ptype != type_ {
            return Err("Type mismatch");
        }
    }

    let mut p = &pbuf[header_size..];
    let plimit = &pbuf[size - 1];

    for _ in 0..count {
        match type_ {
            BINN_OBJECT => {
                let len = p[0];
                p = &p[1..];
                p = &p[len as usize..];
            }
            BINN_MAP => {
                read_map_id(&mut p, plimit)?;
            }
            _ => {}
        }
        p = advance_data_pos(p, plimit).ok_or("Invalid data position")?;
    }

    Ok(())
}

/// Loads binary data into a `Binn` structure.
///
/// # Arguments
/// * `data` - A slice of binary data.
/// * `value` - A mutable reference to the `Binn` structure to be initialized.
///
/// # Returns
/// Returns `Ok(())` if the data is successfully loaded, otherwise returns an error message.
pub fn binn_load(data: &[u8], value: &mut Binn) -> Result<(), &'static str> {
    if data.is_empty() || value.is_null() {
        return Err("Invalid input");
    }
    *value = Binn::default();
    value.header = BINN_MAGIC;
    binn_is_valid(Some(data), Some(&mut value.type_), Some(&mut value.count), Some(&mut value.size))?;
    value.ptr = Some(Box::new(data.to_vec()));
    Ok(())
}

/// Opens binary data and returns a `Binn` structure.
///
/// # Arguments
/// * `data` - A slice of binary data.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` if the data is successfully opened, otherwise returns an error message.
pub fn binn_open(data: &[u8]) -> Result<Box<Binn>, &'static str> {
    let mut item = Box::new(Binn::default());
    binn_load(data, &mut item)?;
    item.allocated = true;
    Ok(item)
}


/// Creates a new binary data object containing an unsigned 32-bit integer.
///
/// # Arguments
/// * `value` - The unsigned 32-bit integer to be stored.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` if the object was successfully created, otherwise returns `Err(&'static str)`.
pub fn binn_uint32(value: u32) -> Result<Box<Binn>, &'static str> {
    binn_value(BINN_UINT32, &value, 0, None)
}

/// Encapsulates a 64-bit signed integer into a Binn object.
///
/// # Arguments
/// * `value` - The 64-bit signed integer to encapsulate.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` if the encapsulation is successful, otherwise returns an error message.
pub fn binn_int64(value: i64) -> Result<Box<Binn>, &'static str> {
    binn_value(BINN_INT64, &value, 0, None)
}

/// Creates a new Binn object containing a string.
///
/// # Arguments
/// * `str` - The string to be stored in the Binn object.
/// * `freefn` - An optional function to free the string when the Binn object is dropped.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` if the object was successfully created, otherwise returns `Err(&'static str)`.
pub fn binn_string(str: &str, freefn: Option<Box<dyn FnOnce(Box<dyn std::any::Any>)>>) -> Result<Box<Binn>, &'static str> {
    binn_value(BINN_STRING, str.as_bytes(), 0, freefn)
}

/// Encapsulates a 32-bit integer into a Binn binary data format.
///
/// # Arguments
/// * `value` - The 32-bit integer to be encapsulated.
///
/// # Returns
/// Returns `Result<Box<Binn>, &'static str>` containing the Binn object if successful,
/// otherwise returns an error message.
pub fn binn_int32(value: i32) -> Result<Box<Binn>, &'static str> {
    binn_value(BINN_INT32, &value, 0, None)
}

/// Creates a new Binn object containing a 16-bit signed integer.
///
/// # Arguments
/// * `value` - The 16-bit signed integer to be stored in the Binn object.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` if the object was successfully created, otherwise returns `Err(&'static str)`.
pub fn binn_int16(value: i16) -> Result<Box<Binn>, &'static str> {
    binn_value(BINN_INT16, &value, 0, None)
}

/// Creates a new Binn object containing a floating-point value.
///
/// # Arguments
/// * `value` - The floating-point value to be stored in the Binn object.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` if the object was successfully created, otherwise returns `Err(&'static str)`.
pub fn binn_float(value: f32) -> Result<Box<Binn>, &'static str> {
    binn_value(BINN_FLOAT, &value, 0, None)
}

/// Creates a new binary data object containing an unsigned 8-bit integer.
///
/// # Arguments
/// * `value` - The unsigned 8-bit integer to be stored in the binary data object.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` if the object was successfully created, otherwise returns `Err(&'static str)`.
pub fn binn_uint8(value: u8) -> Result<Box<Binn>, &'static str> {
    binn_value(BINN_UINT8, &value, 0, None)
}

/// Creates a new Binn object containing a 64-bit unsigned integer.
///
/// # Arguments
/// * `value` - The 64-bit unsigned integer to be stored in the Binn object.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` containing the new Binn object if successful, otherwise returns an error message.
pub fn binn_uint64(value: u64) -> Result<Box<Binn>, &'static str> {
    binn_value(BINN_UINT64, &value, 0, None)
}

/// Creates a new Binn object containing an unsigned 16-bit integer value.
///
/// # Arguments
/// * `value` - The unsigned 16-bit integer value to be stored in the Binn object.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` if the Binn object was successfully created, otherwise returns an error message.
pub fn binn_uint16(value: u16) -> Result<Box<Binn>, &'static str> {
    binn_value(BINN_UINT16, &value, 0, None)
}

impl Binn {
    /// Adds a raw value to the binary list.
    ///
    /// # Arguments
    /// * `type_` - The type of the value to add.
    /// * `pvalue` - A slice containing the value data.
    /// * `size` - The size of the data.
    ///
    /// # Returns
    /// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
    fn binn_list_add_raw(&mut self, type_: i32, pvalue: &[u8], size: usize) -> Result<(), &'static str> {
        if self.type_ != BINN_LIST || !self.writable {
            return Err("Invalid list or not writable");
        }

        self.add_value(type_, pvalue)?;
        self.count += 1;

        Ok(())
    }

    /// Adds a value to the binary list after converting it to the appropriate format.
    ///
    /// # Arguments
    /// * `type_` - The type of the value to add.
    /// * `pvalue` - A slice containing the value data.
    /// * `size` - The size of the data.
    ///
    /// # Returns
    /// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
    pub fn binn_list_add(&mut self, type_: i32, pvalue: &[u8], size: usize) -> Result<(), &'static str> {
        let (converted_type, converted_value, converted_size) = get_write_converted_data(type_, pvalue, size)?;
        self.binn_list_add_raw(converted_type, converted_value, converted_size)
    }
}

/// 从二进制映射中获取指定 ID 的值。
///
/// # 参数
/// - `ptr`: 指向二进制数据的引用。
/// - `id`: 要查找的标识符。
/// - `value`: 用于存储结果的 `Binn` 结构体的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功找到并存储了值，否则返回 `Err(&'static str)`。
pub fn binn_map_get_value(ptr: &[u8], id: i32, value: &mut Binn) -> Result<(), &'static str> {
    let ptr = binn_ptr(ptr)?;
    if ptr.is_empty() {
        return Err("Invalid pointer");
    }

    // 检查头部信息
    let (type_, count, size, header_size) = is_valid_binn_header(ptr, None, None, None, None)?;
    if type_ != BINN_MAP {
        return Err("Type mismatch");
    }
    if count == 0 {
        return Err("Empty map");
    }

    // 搜索指定 ID 的值
    let p = search_for_id(ptr, header_size, size, count as usize, id)
        .ok_or("ID not found")?;

    // 获取值并存储到 `value` 中
    value.get_value(p)
}

/// Advances the iterator to the next element in the binary list and retrieves its value.
///
/// # Arguments
/// * `iter` - A mutable reference to the `BinnIter` iterator.
/// * `value` - A mutable reference to the `Binn` structure to store the retrieved value.
///
/// # Returns
/// Returns `Ok(true)` if the next element was successfully retrieved, otherwise returns `Err("error message")`.
pub fn binn_list_next(iter: &mut BinnIter, value: &mut Binn) -> Result<bool, &'static str> {
    if iter.pnext.is_null() || iter.pnext > iter.plimit || iter.current > iter.count || iter.type_ != BINN_LIST {
        return Err("Invalid iterator state");
    }

    iter.current += 1;
    if iter.current > iter.count {
        return Ok(false);
    }

    let pnow = iter.pnext;
    iter.pnext = advance_data_pos(unsafe { std::slice::from_raw_parts(pnow, (iter.plimit as usize) - (pnow as usize)) }, unsafe { std::slice::from_raw_parts(iter.plimit, 0) })?;
    if iter.pnext.is_null() || iter.pnext < pnow {
        return Err("Invalid data position");
    }

    value.get_value(unsafe { std::slice::from_raw_parts(pnow, (iter.plimit as usize) - (pnow as usize)) })?;
    Ok(true)
}

impl Binn {
    /// Adds a key-value pair to the binary object.
    ///
    /// # Arguments
    /// * `key` - The key to add.
    /// * `type_` - The type of the value.
    /// * `value` - The value data.
    ///
    /// # Returns
    /// Returns `Ok(())` if the key-value pair was successfully added, otherwise returns an error message.
    pub fn binn_object_set_raw(&mut self, key: &str, type_: i32, value: &[u8]) -> Result<(), &'static str> {
        if self.type_ != BINN_OBJECT || !self.writable {
            return Err("Invalid binary object or not writable");
        }

        if key.is_empty() || key.len() > 255 {
            return Err("Invalid key length");
        }

        if self.search_for_key(key).is_some() {
            return Err("Key already exists");
        }

        let key_len = key.len();
        self.check_allocation(1 + key_len)?;

        let p = &mut self.pbuf.as_mut().unwrap()[self.used_size..];
        p[0] = key_len as u8;
        p[1..1 + key_len].copy_from_slice(key.as_bytes());
        self.used_size += 1 + key_len;

        self.add_value(type_, value)?;
        self.count += 1;

        Ok(())
    }

    /// Sets a key-value pair in the binary object.
    ///
    /// # Arguments
    /// * `key` - The key to set.
    /// * `type_` - The type of the value.
    /// * `value` - The value data.
    ///
    /// # Returns
    /// Returns `Ok(())` if the key-value pair was successfully set, otherwise returns an error message.
    pub fn binn_object_set(&mut self, key: &str, type_: i32, value: &[u8]) -> Result<(), &'static str> {
        let (converted_type, converted_value, converted_size) = self.get_write_converted_data(type_, value)?;
        self.binn_object_set_raw(key, converted_type, converted_value)
    }
}

/// 从二进制列表中获取指定位置的元素值。
///
/// # 参数
/// - `ptr`: 指向二进制数据的引用。
/// - `pos`: 元素的位置索引（从1开始）。
/// - `value`: 用于存储结果的 `Binn` 结构体的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功获取元素值，否则返回错误信息。
pub fn binn_list_get_value(ptr: &[u8], pos: usize, value: &mut Binn) -> Result<(), &'static str> {
    let ptr = binn_ptr(ptr)?;
    if ptr.is_empty() {
        return Err("Invalid pointer");
    }

    // 检查头部信息
    let (type_, count, size, header_size) = is_valid_binn_header(ptr, None, None, None, None)?;

    if type_ != BINN_LIST {
        return Err("Not a BINN_LIST");
    }
    if count == 0 {
        return Err("Empty list");
    }
    if pos == 0 || pos > count {
        return Err("Invalid position");
    }
    let pos = pos - 1;  // 转换为从0开始的索引

    let base = ptr.as_ptr();
    let plimit = unsafe { base.add(size) };
    let mut p = unsafe { base.add(header_size) };

    for _ in 0..pos {
        p = advance_data_pos(unsafe { std::slice::from_raw_parts(p, (plimit as usize) - (p as usize)) }, unsafe { std::slice::from_raw_parts(plimit, 0) })?.as_ptr();
        if p < base {
            return Err("Invalid position");
        }
    }

    value.get_value(unsafe { std::slice::from_raw_parts(p, (plimit as usize) - (p as usize)) })
}

/// 从二进制对象中提取指定键的值，并将其存储在 `Binn` 结构体中。
///
/// # 参数
/// - `key`: 要查找的键名。
/// - `value`: 用于存储结果的 `Binn` 结构体。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功提取并存储了值，否则返回错误信息。
pub fn binn_object_get_value(&self, key: &str, value: &mut Binn) -> Result<(), &'static str> {
    let ptr = self.binn_ptr()?;
    if key.is_empty() || value.is_null() {
        return Err("Invalid input");
    }

    let (type_, count, size, header_size) = is_valid_binn_header(ptr, None, None, None, None)?;
    if type_ != BINN_OBJECT {
        return Err("Type mismatch");
    }
    if count == 0 {
        return Err("Empty object");
    }

    let p = search_for_key(ptr, header_size, key)?;
    value.get_value(p)?;
    Ok(())
}

/// Adds a key-value pair to the binary map.
///
/// # Arguments
/// * `id` - The identifier of the key.
/// * `type_` - The type of the value.
/// * `pvalue` - A reference to the value data.
/// * `size` - The size of the value data.
///
/// # Returns
/// Returns `Ok(())` if the key-value pair was successfully added, otherwise returns an error message.
pub fn binn_map_set_raw(&mut self, id: i32, type_: i32, pvalue: &[u8], size: usize) -> Result<(), &'static str> {
    if self.type_ != BINN_MAP || !self.writable {
        return Err("Invalid map or not writable");
    }

    // Check if the ID already exists
    if let Some(_) = search_for_id(self.pbuf.as_ref().unwrap(), MAX_BINN_HEADER, self.used_size, self.count, id) {
        return Err("ID already exists");
    }

    // Ensure enough space is allocated
    self.check_allocation(5)?;  // max 5 bytes used for the id.

    let base = self.used_size;
    let mut p = base;
    let sign = (id < 0) as u8;
    let id = if sign != 0 { -id } else { id };

    if id <= 0x3F {
        self.pbuf.as_mut().unwrap()[p] = (sign << 6) | (id as u8);
        p += 1;
    } else if id <= 0xFFF {
        self.pbuf.as_mut().unwrap()[p] = 0x80 | (sign << 4) | ((id & 0xF00) >> 8) as u8;
        p += 1;
        self.pbuf.as_mut().unwrap()[p] = (id & 0xFF) as u8;
        p += 1;
    } else if id <= 0xFFFFF {
        self.pbuf.as_mut().unwrap()[p] = 0xA0 | (sign << 4) | ((id & 0xF0000) >> 16) as u8;
        p += 1;
        self.pbuf.as_mut().unwrap()[p] = ((id & 0xFF00) >> 8) as u8;
        p += 1;
        self.pbuf.as_mut().unwrap()[p] = (id & 0xFF) as u8;
        p += 1;
    } else if id <= 0xFFFFFFF {
        self.pbuf.as_mut().unwrap()[p] = 0xC0 | (sign << 4) | ((id & 0xF000000) >> 24) as u8;
        p += 1;
        self.pbuf.as_mut().unwrap()[p] = ((id & 0xFF0000) >> 16) as u8;
        p += 1;
        self.pbuf.as_mut().unwrap()[p] = ((id & 0xFF00) >> 8) as u8;
        p += 1;
        self.pbuf.as_mut().unwrap()[p] = (id & 0xFF) as u8;
        p += 1;
    } else {
        self.pbuf.as_mut().unwrap()[p] = 0xE0;
        p += 1;
        copy_be32(&mut self.pbuf.as_mut().unwrap()[p..p + 4], &(id as u32));
        p += 4;
    }

    let id_size = p - base;
    self.used_size += id_size;

    if self.add_value(type_, pvalue).is_err() {
        self.used_size -= id_size;
        return Err("Failed to add value");
    }

    self.count += 1;
    Ok(())
}

/// Adds a key-value pair to the binary map after converting the data format.
///
/// # Arguments
/// * `id` - The identifier of the key.
/// * `type_` - The type of the value.
/// * `pvalue` - A reference to the value data.
/// * `size` - The size of the value data.
///
/// # Returns
/// Returns `Ok(())` if the key-value pair was successfully added, otherwise returns an error message.
pub fn binn_map_set(&mut self, id: i32, type_: i32, pvalue: &[u8], size: usize) -> Result<(), &'static str> {
    let (new_type, new_pvalue, new_size) = get_write_converted_data(type_, pvalue, size)?;
    self.binn_map_set_raw(id, new_type, new_pvalue, new_size)
}

/// Adds an 8-bit signed integer to the Binn list.
///
/// # Arguments
/// * `value` - The 8-bit signed integer to add.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
pub fn binn_list_add_int8(&mut self, value: i8) -> Result<(), &'static str> {
    self.binn_list_add(BINN_INT8, &value, 0)
}

/// Adds an unsigned 8-bit integer to the Binn list.
///
/// # Arguments
/// * `value` - The unsigned 8-bit integer to add.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
pub fn binn_list_add_uint8(&mut self, value: u8) -> Result<(), &'static str> {
    self.binn_list_add(BINN_UINT8, &value, 0)
}

/// Reads the next key-value pair from the binary data structure.
///
/// # Arguments
/// * `expected_type` - The expected type of the binary data (e.g., `BINN_MAP`, `BINN_OBJECT`).
/// * `iter` - A mutable reference to the iterator.
/// * `pid` - An optional mutable reference to store the ID (for `BINN_MAP`).
/// * `pkey` - An optional mutable reference to store the key (for `BINN_OBJECT`).
/// * `value` - A mutable reference to the `Binn` structure to store the value.
///
/// # Returns
/// Returns `Ok(())` if the next key-value pair was successfully read, otherwise returns an error message.
pub fn binn_read_next_pair(
    expected_type: i32,
    iter: &mut BinnIter,
    pid: Option<&mut i32>,
    pkey: Option<&mut String>,
    value: &mut Binn,
) -> Result<(), &'static str> {
    if iter.pnext.is_null() || iter.pnext > iter.plimit || iter.current > iter.count || iter.type_ != expected_type {
        return Err("Invalid iterator");
    }

    iter.current += 1;
    if iter.current > iter.count {
        return Err("Iterator out of bounds");
    }

    let mut p = iter.pnext;

    match expected_type {
        BINN_MAP => {
            let id = read_map_id(unsafe { std::slice::from_raw_parts(p, (iter.plimit as usize) - (p as usize)) })
                .ok_or("Failed to read map ID")?;
            if let Some(pid) = pid {
                *pid = id;
            }
        }
        BINN_OBJECT => {
            let len = unsafe { *p } as usize;
            p = unsafe { p.add(1) };
            let key = unsafe { std::slice::from_raw_parts(p, len) };
            p = unsafe { p.add(len) };
            if p > iter.plimit {
                return Err("Invalid key length");
            }
            if let Some(pkey) = pkey {
                *pkey = String::from_utf8(key.to_vec()).map_err(|_| "Invalid UTF-8 key")?;
            }
        }
        _ => return Err("Unsupported type"),
    }

    iter.pnext = advance_data_pos(unsafe { std::slice::from_raw_parts(p, (iter.plimit as usize) - (p as usize)) }, unsafe {
        std::slice::from_raw_parts(iter.plimit, 0)
    })
    .ok_or("Failed to advance data position")?;
    if iter.pnext < p {
        return Err("Invalid data position");
    }

    value.get_value(unsafe { std::slice::from_raw_parts(p, (iter.plimit as usize) - (p as usize)) })
}


/// Reads a key-value pair from the binary data structure at the specified position.
///
/// # Arguments
/// * `expected_type` - The expected type of the binary data (e.g., `BINN_MAP`, `BINN_OBJECT`).
/// * `ptr` - A reference to the binary data.
/// * `pos` - The position index of the key-value pair to read.
/// * `pid` - An optional mutable reference to store the ID (for `BINN_MAP`).
/// * `pkey` - An optional mutable reference to store the key (for `BINN_OBJECT`).
/// * `value` - A mutable reference to the `Binn` structure to store the value.
///
/// # Returns
/// Returns `Ok(())` if the key-value pair was successfully read, otherwise returns an error message.
pub fn binn_read_pair(
    expected_type: i32,
    ptr: &[u8],
    pos: i32,
    pid: Option<&mut i32>,
    pkey: Option<&mut String>,
    value: &mut Binn,
) -> Result<(), &'static str> {
    let mut type_ = 0;
    let mut count = 0;
    let mut size = 0;
    let mut header_size = 0;

    // Check the header
    is_valid_binn_header(ptr, Some(&mut type_), Some(&mut count), Some(&mut size), Some(&mut header_size))?;

    if type_ != expected_type || count == 0 || pos < 1 || pos > count {
        return Err("Invalid position or type");
    }

    let mut p = &ptr[header_size..];
    let base = ptr.as_ptr();
    let plimit = unsafe { base.add(size - 1) };

    for i in 0..count {
        match type_ {
            BINN_MAP => {
                let id = read_map_id(&mut p)?;
                if p.as_ptr() > plimit {
                    return Err("Invalid data position");
                }
                if i + 1 == pos {
                    if let Some(pid) = pid {
                        *pid = id;
                    }
                    return value.get_value(p);
                }
            }
            BINN_OBJECT => {
                let len = p[0] as usize;
                p = &p[1..];
                if p.as_ptr() > plimit {
                    return Err("Invalid data position");
                }
                let key = &p[..len];
                p = &p[len..];
                if p.as_ptr() > plimit {
                    return Err("Invalid data position");
                }
                if i + 1 == pos {
                    if let Some(pkey) = pkey {
                        *pkey = String::from_utf8_lossy(key).to_string();
                    }
                    return value.get_value(p);
                }
            }
            _ => return Err("Unsupported type"),
        }

        p = advance_data_pos(p, unsafe { std::slice::from_raw_parts(plimit, 0) })?;
        if p.as_ptr() < base {
            return Err("Invalid data position");
        }
    }

    Err("Position not found")
}

/// Adds a 64-bit signed integer to the Binn list.
///
/// # Arguments
/// * `value` - The 64-bit signed integer to add.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
pub fn binn_list_add_int64(&mut self, value: i64) -> Result<(), &'static str> {
    self.binn_list_add(BINN_INT64, &value, 0)
}

/// Adds a 32-bit floating-point value to the Binn list.
///
/// # Arguments
/// * `value` - The 32-bit floating-point value to add.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
pub fn binn_list_add_float(&mut self, value: f32) -> Result<(), &'static str> {
    self.binn_list_add(BINN_FLOAT32, &value, 0)
}

/// Adds a list to the binary list.
///
/// # Arguments
/// * `list2` - The list to be added.
///
/// # Returns
/// Returns `Ok(())` if the list was successfully added, otherwise returns an error message.
pub fn binn_list_add_list(&mut self, list2: &Binn) -> Result<(), &'static str> {
    self.binn_list_add(BINN_LIST, binn_ptr(list2)?, binn_size(list2))
}

/// Adds a map to the Binn list.
///
/// # Arguments
/// * `map` - A reference to the Binn map to be added.
///
/// # Returns
/// Returns `Ok(())` if the map was successfully added, otherwise returns an error message.
pub fn binn_list_add_map(&mut self, map: &Binn) -> Result<(), &'static str> {
    self.binn_list_add(BINN_MAP, binn_ptr(map)?, binn_size(map)?)
}

/// Adds a 64-bit unsigned integer to the Binn list.
///
/// # Arguments
/// * `value` - The 64-bit unsigned integer to add.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
pub fn binn_list_add_uint64(&mut self, value: u64) -> Result<(), &'static str> {
    self.binn_list_add(BINN_UINT64, &value, 0)
}

/// Adds a double-precision floating-point value to the Binn list.
///
/// # Arguments
/// * `value` - The double-precision floating-point value to add.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
pub fn binn_list_add_double(&mut self, value: f64) -> Result<(), &'static str> {
    self.binn_list_add(BINN_FLOAT64, &value, 0)
}

/// Adds a 32-bit unsigned integer to the Binn list.
///
/// # Arguments
/// * `value` - The 32-bit unsigned integer to add.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
pub fn binn_list_add_uint32(&mut self, value: u32) -> Result<(), &'static str> {
    self.binn_list_add(BINN_UINT32, &value, 0)
}

/// Adds a string to the Binn list.
///
/// # Arguments
/// * `list` - A mutable reference to the Binn list.
/// * `str` - The string to add.
///
/// # Returns
/// Returns `Ok(())` if the string was successfully added, otherwise returns an error message.
pub fn binn_list_add_str(list: &mut Binn, str: &str) -> Result<(), &'static str> {
    binn_list_add(list, BINN_STRING, str.as_bytes(), 0)
}

/// Adds a boolean value to the binary list.
///
/// # Arguments
/// * `value` - The boolean value to add.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
pub fn binn_list_add_bool(&mut self, value: bool) -> Result<(), &'static str> {
    self.binn_list_add(BINN_BOOL, &value, 0)
}

/// Adds a 16-bit signed integer to the Binn list.
///
/// # Arguments
/// * `value` - The 16-bit signed integer to add.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
pub fn binn_list_add_int16(&mut self, value: i16) -> Result<(), &'static str> {
    self.binn_list_add(BINN_INT16, &value, 0)
}

/// Adds a 16-bit unsigned integer to the Binn list.
///
/// # Arguments
/// * `value` - The 16-bit unsigned integer to add.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
pub fn binn_list_add_uint16(&mut self, value: u16) -> Result<(), &'static str> {
    self.binn_list_add(BINN_UINT16, &value, 0)
}

/// Adds a 32-bit integer to the Binn list.
///
/// # Arguments
/// * `value` - The 32-bit integer to add.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
pub fn binn_list_add_int32(&mut self, value: i32) -> Result<(), &'static str> {
    self.binn_list_add(BINN_INT32, &value.to_ne_bytes(), 0)
}

/// Adds a null value to the Binn list.
///
/// # Returns
/// Returns `Ok(())` if the null value was successfully added, otherwise returns an error message.
pub fn binn_list_add_null(&mut self) -> Result<(), &'static str> {
    self.binn_list_add(BINN_NULL, &[], 0)
}

/// Adds a `Binn` value to the list.
///
/// # Arguments
/// * `value` - A reference to the `Binn` value to add.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
pub fn binn_list_add_value(&mut self, value: &Binn) -> Result<(), &'static str> {
    self.binn_list_add(value.type_, value.binn_ptr()?, value.binn_size())
}

/// Adds a binary data block (blob) to the Binn list.
///
/// # Arguments
/// * `list` - A mutable reference to the Binn list.
/// * `data` - A slice containing the binary data.
///
/// # Returns
/// Returns `Ok(())` if the blob was successfully added, otherwise returns an error message.
pub fn binn_list_add_blob(list: &mut Binn, data: &[u8]) -> Result<(), &'static str> {
    list.binn_list_add(BINN_BLOB, data)
}

/// Adds an object to the Binn list.
///
/// # Arguments
/// * `obj` - A reference to the object to be added.
///
/// # Returns
/// Returns `Ok(())` if the object was successfully added, otherwise returns an error message.
pub fn binn_list_add_object(&mut self, obj: &Binn) -> Result<(), &'static str> {
    let ptr = obj.binn_ptr()?;
    let size = binn_size(Some(obj));
    self.binn_list_add(BINN_OBJECT, ptr, size)
}

/// 从二进制映射中获取指定 ID 的值，并返回一个指向该值的指针。
///
/// # 参数
/// - `ptr`: 指向二进制数据的引用。
/// - `id`: 要查找的标识符。
///
/// # 返回值
/// 返回 `Result<Box<Binn>, &'static str>`，表示成功时返回 `Box<Binn>`，失败时返回错误信息。
pub fn binn_map_value(ptr: &[u8], id: i32) -> Result<Box<Binn>, &'static str> {
    let mut value = Box::new(Binn::default());
    binn_map_get_value(ptr, id, &mut value)?;
    value.allocated = true;
    Ok(value)
}

/// 从二进制映射中读取指定 ID 的值，并返回该值的指针。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 要查找的标识符。
/// - `ptype`: 可选参数，用于存储值的类型。
/// - `psize`: 可选参数，用于存储值的大小。
///
/// # 返回值
/// 返回 `Result<Option<&dyn std::any::Any>, &'static str>`，表示成功时返回值的指针，失败时返回错误信息。
pub fn binn_map_read(map: &Binn, id: i32, ptype: Option<&mut i32>, psize: Option<&mut usize>) -> Result<Option<&dyn std::any::Any>, &'static str> {
    let mut value = Binn::default();
    binn_map_get_value(map, id, &mut value)?;

    if let Some(ptype) = ptype {
        *ptype = value.type_;
    }
    if let Some(psize) = psize {
        *psize = value.size;
    }

    if cfg!(target_endian = "little") {
        Ok(value.store_value())
    } else {
        Ok(value.ptr.as_ref().map(|ptr| ptr.as_ref()))
    }
}

/// Reads the next value from the binary list iterator.
///
/// # Arguments
/// * `iter` - A mutable reference to the binary list iterator.
/// * `ptype` - An optional mutable reference to store the type of the value.
/// * `psize` - An optional mutable reference to store the size of the value.
///
/// # Returns
/// Returns `Ok(Some(&dyn std::any::Any))` containing the value if successful, otherwise returns `Err(&'static str)`.
pub fn binn_list_read_next(iter: &mut BinnIter, ptype: Option<&mut i32>, psize: Option<&mut usize>) -> Result<Option<&dyn std::any::Any>, &'static str> {
    let mut value = Binn::default();
    if !binn_list_next(iter, &mut value)? {
        return Ok(None);
    }
    if let Some(ptype) = ptype {
        *ptype = value.type_;
    }
    if let Some(psize) = psize {
        *psize = value.size;
    }
    if cfg!(target_endian = "little") {
        Ok(value.store_value())
    } else {
        Ok(value.ptr.as_ref().map(|ptr| ptr.as_ref()))
    }
}

/// 从二进制列表迭代器中获取下一个值，并返回一个指向该值的指针。
///
/// # 参数
/// - `iter`: 指向二进制列表迭代器的可变引用。
///
/// # 返回值
/// 返回 `Result<Box<Binn>, &'static str>`，表示成功时返回 `Box<Binn>`，失败时返回错误信息。
pub fn binn_list_next_value(iter: &mut BinnIter) -> Result<Box<Binn>, &'static str> {
    let mut value = Box::new(Binn::default());
    if binn_list_next(iter, &mut value)? {
        value.allocated = true;
        Ok(value)
    } else {
        Err("Failed to get next value")
    }
}

/// Sets a list value in the Binn object with the specified key.
///
/// # Arguments
/// * `key` - The key to associate with the list.
/// * `list` - The list to be stored in the Binn object.
///
/// # Returns
/// Returns `Ok(())` if the operation was successful, otherwise returns an error message.
pub fn binn_object_set_list(&mut self, key: &str, list: &Binn) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_LIST, list.binn_ptr()?, list.binn_size())
}

/// 从二进制映射中获取指定 ID 的值，并将其存储在提供的指针中。
///
/// # 参数
/// - `id`: 要查找的键的标识符。
/// - `type_`: 期望的数据类型。
/// - `pvalue`: 存储值的指针。
/// - `psize`: 可选参数，用于返回数据的大小。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功获取并存储了值，否则返回 `Err(&'static str)`。
pub fn binn_map_get(&self, id: i32, type_: i32, pvalue: &mut [u8], psize: &mut Option<usize>) -> Result<(), &'static str> {
    let storage_type = binn_get_read_storage(type_)?;
    if storage_type != BINN_STORAGE_NOBYTES && pvalue.is_empty() {
        return Err("Invalid value pointer");
    }

    zero_value(pvalue, type_);

    let mut value = Binn::default();
    binn_map_get_value(self, id, &mut value)?;

    copy_value(value.ptr.as_ref().unwrap(), pvalue, value.type_, type_, storage_type)?;

    if let Some(size) = psize {
        *size = value.size;
    }

    Ok(())
}

/// Sets a value in the binary object with the specified key.
///
/// # Arguments
/// * `key` - The key to associate with the value.
/// * `value` - The value to store in the binary object.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_object_set_value(&mut self, key: &str, value: &Binn) -> Result<(), &'static str> {
    let ptr = value.binn_ptr()?;
    let size = value.binn_size();
    self.binn_object_set(key, value.type_, ptr, size)
}

/// Sets an unsigned 8-bit integer value in a Binn object.
///
/// # Arguments
/// * `key` - The key to associate with the value.
/// * `value` - The unsigned 8-bit integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_object_set_uint8(&mut self, key: &str, value: u8) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_UINT8, &value, 0)
}

/// Sets a floating-point value in a Binn object with the specified key.
///
/// # Arguments
/// * `key` - The key to associate with the floating-point value.
/// * `value` - The floating-point value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_object_set_float(&mut self, key: &str, value: f32) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_FLOAT32, &value, 0)
}

/// Sets a double-precision floating-point value in a Binn object.
///
/// # Arguments
/// * `key` - The key associated with the value.
/// * `value` - The double-precision floating-point value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_object_set_double(&mut self, key: &str, value: f64) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_FLOAT64, &value, 0)
}

/// Sets a boolean value in a Binn object.
///
/// # Arguments
/// * `key` - The key associated with the boolean value.
/// * `value` - The boolean value to set.
///
/// # Returns
/// Returns `Ok(())` if the operation was successful, otherwise returns an error message.
pub fn binn_object_set_bool(&mut self, key: &str, value: bool) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_BOOL, &value, 0)
}

/// Sets a string value in a Binn object.
///
/// # Arguments
/// * `key` - The key to associate with the string value.
/// * `str` - The string value to set.
///
/// # Returns
/// Returns `Ok(())` if the operation was successful, otherwise returns an error.
pub fn binn_object_set_str(&mut self, key: &str, str: &str) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_STRING, str.as_bytes(), 0)
}

/// Sets an unsigned 32-bit integer value in a Binn object.
///
/// # Arguments
/// * `key` - The key to associate with the value.
/// * `value` - The unsigned 32-bit integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_object_set_uint32(&mut self, key: &str, value: u32) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_UINT32, &value, 0)
}

/// Sets a key-value pair in the binary object with the value set to `NULL`.
///
/// # Arguments
/// * `key` - The key to set.
///
/// # Returns
/// Returns `Ok(())` if the key-value pair was successfully set, otherwise returns an error message.
pub fn binn_object_set_null(&mut self, key: &str) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_NULL, &[], 0)
}

/// Sets a 16-bit signed integer value in a Binn object.
///
/// # Arguments
/// * `key` - The key to associate with the value.
/// * `value` - The 16-bit signed integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_object_set_int16(&mut self, key: &str, value: i16) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_INT16, &value, 0)
}

/// Sets an unsigned 16-bit integer value in a Binn object.
///
/// # Arguments
/// * `key` - The key to associate with the value.
/// * `value` - The unsigned 16-bit integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_object_set_uint16(&mut self, key: &str, value: u16) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_UINT16, &value, 0)
}

/// Sets a binary data block (BLOB) in a Binn object.
///
/// # Arguments
/// * `key` - The key associated with the BLOB data.
/// * `data` - The binary data to be stored.
///
/// # Returns
/// Returns `Ok(())` if the operation was successful, otherwise returns an error message.
pub fn binn_object_set_blob(&mut self, key: &str, data: &[u8]) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_BLOB, data)
}

/// Sets an 8-bit signed integer value in a Binn object.
///
/// # Arguments
/// * `key` - The key to associate with the value.
/// * `value` - The 8-bit signed integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_object_set_int8(&mut self, key: &str, value: i8) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_INT8, &value, 0)
}

/// Sets a binary object as a value in another binary object with the specified key.
///
/// # Arguments
/// * `key` - The key to associate with the binary object.
/// * `obj2` - The binary object to be set as the value.
///
/// # Returns
/// Returns `Ok(())` if the operation was successful, otherwise returns an error message.
pub fn binn_object_set_object(&mut self, key: &str, obj2: &Binn) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_OBJECT, binn_ptr(obj2)?, binn_size(obj2))
}

/// 将一个 32 位整数与指定的键关联，并将其存储在二进制对象中。
///
/// # 参数
/// - `key`: 键名，用于标识存储的值。
/// - `value`: 要存储的 32 位整数值。
///
/// # 返回值
/// 返回 `Ok(())` 如果操作成功，否则返回错误信息。
pub fn binn_object_set_int32(&mut self, key: &str, value: i32) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_INT32, &value, 0)
}

/// Sets a 64-bit unsigned integer value in a Binn object with the specified key.
///
/// # Arguments
/// * `key` - The key to associate with the value.
/// * `value` - The 64-bit unsigned integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_object_set_uint64(&mut self, key: &str, value: u64) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_UINT64, &value, 0)
}

/// Sets a 64-bit signed integer value in a Binn object with the specified key.
///
/// # Arguments
/// * `key` - The key to associate with the value.
/// * `value` - The 64-bit signed integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_object_set_int64(&mut self, key: &str, value: i64) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_INT64, &value, 0)
}

/// Sets a map in the binary object with the specified key.
///
/// # Arguments
/// * `key` - The key to associate with the map.
/// * `map` - A reference to the binary map to be set.
///
/// # Returns
/// Returns `Ok(())` if the map was successfully set, otherwise returns an error message.
pub fn binn_object_set_map(&mut self, key: &str, map: &Binn) -> Result<(), &'static str> {
    self.binn_object_set(key, BINN_MAP, map.binn_ptr()?, binn_size(Some(map)))
}

/// 从二进制列表中获取指定位置的元素值，并返回一个指向该值的指针。
///
/// # 参数
/// - `ptr`: 指向二进制数据的引用。
/// - `pos`: 元素的位置索引（从1开始）。
///
/// # 返回值
/// 返回 `Result<Box<Binn>, &'static str>`，表示成功时返回 `Box<Binn>`，失败时返回错误信息。
pub fn binn_list_value(ptr: &[u8], pos: usize) -> Result<Box<Binn>, &'static str> {
    let mut value = Box::new(Binn::default());
    binn_list_get_value(ptr, pos, &mut value)?;
    value.allocated = true;
    Ok(value)
}

/// 从二进制列表中读取指定位置的元素，并返回该元素的指针。
///
/// # 参数
/// - `list`: 指向二进制列表的引用。
/// - `pos`: 元素的位置索引（从1开始）。
/// - `ptype`: 可选参数，用于存储元素的类型。
/// - `psize`: 可选参数，用于存储元素的大小。
///
/// # 返回值
/// 返回 `Result<Option<&dyn std::any::Any>, &'static str>`，表示成功时返回元素的指针，失败时返回错误信息。
pub fn binn_list_read(list: &Binn, pos: usize, ptype: Option<&mut i32>, psize: Option<&mut usize>) -> Result<Option<&dyn std::any::Any>, &'static str> {
    let mut value = Binn::default();
    binn_list_get_value(list, pos, &mut value)?;

    if let Some(ptype) = ptype {
        *ptype = value.type_;
    }
    if let Some(psize) = psize {
        *psize = value.size;
    }

    if cfg!(target_endian = "little") {
        Ok(value.store_value())
    } else {
        Ok(value.ptr.as_ref().map(|ptr| ptr.as_ref()))
    }
}

/// 从二进制对象中提取与指定键关联的值，并返回一个指向该值的指针。
///
/// # 参数
/// - `ptr`: 指向二进制对象的引用。
/// - `key`: 要查找的键名。
///
/// # 返回值
/// 返回 `Result<Box<Binn>, &'static str>`，表示成功时返回 `Box<Binn>`，失败时返回错误信息。
pub fn binn_object_value(ptr: &Binn, key: &str) -> Result<Box<Binn>, &'static str> {
    let mut value = Box::new(Binn::default());
    binn_object_get_value(ptr, key, &mut value)?;
    value.allocated = true;
    Ok(value)
}

/// 从二进制列表中获取指定位置的元素值，并将其复制到提供的指针中。
///
/// # 参数
/// - `ptr`: 指向二进制数据的引用。
/// - `pos`: 元素的位置索引（从1开始）。
/// - `type_`: 期望的数据类型。
/// - `pvalue`: 用于存储值的可变切片。
/// - `psize`: 可选参数，用于返回数据的大小。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功获取并存储了值，否则返回错误信息。
pub fn binn_list_get(ptr: &[u8], pos: usize, type_: i32, pvalue: &mut [u8], psize: Option<&mut usize>) -> Result<(), &'static str> {
    let storage_type = binn_get_read_storage(type_)?;
    if storage_type != BINN_STORAGE_NOBYTES && pvalue.is_empty() {
        return Err("Invalid value pointer");
    }

    zero_value(pvalue, type_);

    let mut value = Binn::default();
    binn_list_get_value(ptr, pos, &mut value)?;

    copy_value(value.ptr.as_ref().unwrap(), pvalue, value.type_, type_, storage_type)?;

    if let Some(size) = psize {
        *size = value.size;
    }

    Ok(())
}

/// 从二进制对象中提取指定键的值，并将其转换为请求的数据类型。
///
/// # 参数
/// - `key`: 键名，用于查找对应的值。
/// - `type_`: 期望的数据类型。
/// - `pvalue`: 用于存储值的可变引用。
/// - `psize`: 可选参数，用于返回数据的大小。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功提取并存储了值，否则返回错误信息。
pub fn binn_object_get(&self, key: &str, type_: i32, pvalue: &mut [u8], psize: &mut Option<usize>) -> Result<(), &'static str> {
    let storage_type = binn_get_read_storage(type_)?;
    if storage_type != BINN_STORAGE_NOBYTES && pvalue.is_empty() {
        return Err("Invalid value pointer");
    }

    zero_value(pvalue, type_);

    let mut value = Binn::default();
    binn_object_get_value(self, key, &mut value)?;

    copy_value(value.ptr.as_ref().unwrap(), pvalue, value.type_, type_, storage_type)?;

    if let Some(size) = psize {
        *size = value.size;
    }

    Ok(())
}

/// Sets a floating-point value in a Binn map.
///
/// # Arguments
/// * `id` - The identifier of the key.
/// * `value` - The floating-point value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_map_set_float(&mut self, id: i32, value: f32) -> Result<(), &'static str> {
    self.binn_map_set(id, BINN_FLOAT32, &value, 0)
}

/// Sets a double-precision floating-point value in a Binn map.
///
/// # Arguments
/// * `id` - The identifier of the key.
/// * `value` - The double-precision floating-point value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_map_set_double(&mut self, id: i32, value: f64) -> Result<(), &'static str> {
    self.binn_map_set(id, BINN_FLOAT64, &value, 0)
}

/// Sets a list value in the Binn map with the specified ID.
///
/// # Arguments
/// * `map` - A mutable reference to the Binn map.
/// * `id` - The identifier for the list.
/// * `list` - A reference to the list to be stored in the map.
///
/// # Returns
/// Returns `Ok(())` if the operation was successful, otherwise returns an error message.
pub fn binn_map_set_list(map: &mut Binn, id: i32, list: &Binn) -> Result<(), &'static str> {
    let ptr = binn_ptr(list)?;
    let size = binn_size(Some(list));
    binn_map_set(map, id, BINN_LIST, ptr, size)
}

/// 从二进制对象中读取指定键的值，并返回该值的指针。
///
/// # 参数
/// - `key`: 要查找的键名。
/// - `ptype`: 可选参数，用于存储值的类型。
/// - `psize`: 可选参数，用于存储值的大小。
///
/// # 返回值
/// 返回 `Option<&dyn std::any::Any>`，表示成功时返回值的指针，失败时返回 `None`。
pub fn binn_object_read(&self, key: &str, ptype: Option<&mut i32>, psize: Option<&mut usize>) -> Option<&dyn std::any::Any> {
    let mut value = Binn::default();
    if self.binn_object_get_value(key, &mut value).is_err() {
        return None;
    }
    if let Some(ptype) = ptype {
        *ptype = value.type_;
    }
    if let Some(psize) = psize {
        *psize = value.size;
    }
    if cfg!(target_endian = "little") {
        value.store_value()
    } else {
        value.ptr.as_ref().map(|ptr| ptr.as_ref())
    }
}

/// Sets a value in the binary map with the specified ID.
///
/// # Arguments
/// * `id` - The identifier of the key.
/// * `value` - The value to be stored in the map.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_map_set_value(&mut self, id: i32, value: &Binn) -> Result<(), &'static str> {
    let ptr = value.binn_ptr()?;
    let size = value.binn_size();
    self.binn_map_set(id, value.type_, ptr, size)
}

/// Sets an object in the binary map with the specified ID.
///
/// # Arguments
/// * `id` - The identifier for the object.
/// * `obj` - A reference to the object to be set.
///
/// # Returns
/// Returns `Ok(())` if the object was successfully set, otherwise returns an error message.
pub fn binn_map_set_object(&mut self, id: i32, obj: &Binn) -> Result<(), &'static str> {
    let ptr = obj.binn_ptr()?;
    let size = binn_size(Some(obj));
    self.binn_map_set(id, BINN_OBJECT, ptr, size)
}

/// Sets a boolean value in a Binn map.
///
/// # Arguments
/// * `map` - A mutable reference to the Binn map.
/// * `id` - The identifier for the key.
/// * `value` - The boolean value to set.
///
/// # Returns
/// Returns `Ok(())` if the operation was successful, otherwise returns an error message.
pub fn binn_map_set_bool(map: &mut Binn, id: i32, value: bool) -> Result<(), &'static str> {
    binn_map_set(map, id, BINN_BOOL, &value, 0)
}

/// Sets a 64-bit unsigned integer value in a Binn map.
///
/// # Arguments
/// * `id` - The identifier of the key.
/// * `value` - The 64-bit unsigned integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_map_set_uint64(&mut self, id: i32, value: u64) -> Result<(), &'static str> {
    self.binn_map_set(id, BINN_UINT64, &value, 0)
}

/// Sets a string value in a Binn map with the specified key.
///
/// # Arguments
/// * `id` - The key to associate with the string.
/// * `str` - The string value to set.
///
/// # Returns
/// Returns `Ok(())` if the operation was successful, otherwise returns an error message.
pub fn binn_map_set_str(&mut self, id: i32, str: &str) -> Result<(), &'static str> {
    self.binn_map_set(id, BINN_STRING, str.as_bytes(), 0)
}

/// Sets a binary data block (BLOB) in a Binn map.
///
/// # Arguments
/// * `id` - The identifier of the key.
/// * `data` - The binary data to be stored.
///
/// # Returns
/// Returns `Ok(())` if the operation was successful, otherwise returns an error message.
#[inline]
pub fn binn_map_set_blob(&mut self, id: i32, data: &[u8]) -> Result<(), &'static str> {
    self.binn_map_set(id, BINN_BLOB, data)
}

/// Sets a 16-bit unsigned integer value in a Binn map.
///
/// # Arguments
/// * `id` - The identifier of the key.
/// * `value` - The 16-bit unsigned integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_map_set_uint16(&mut self, id: i32, value: u16) -> Result<(), &'static str> {
    self.binn_map_set(id, BINN_UINT16, &value, 0)
}

/// Sets a nested map in the binary map.
///
/// # Arguments
/// * `id` - The identifier for the nested map.
/// * `map2` - The nested map to be inserted.
///
/// # Returns
/// Returns `Ok(())` if the operation was successful, otherwise returns an error message.
pub fn binn_map_set_map(&mut self, id: i32, map2: &Binn) -> Result<(), &'static str> {
    let ptr = map2.binn_ptr()?;
    let size = binn_size(Some(map2));
    self.binn_map_set(id, BINN_MAP, ptr, size)
}

/// Sets a 64-bit signed integer value in a Binn map.
///
/// # Arguments
/// * `id` - The identifier of the key.
/// * `value` - The 64-bit signed integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_map_set_int64(&mut self, id: i32, value: i64) -> Result<(), &'static str> {
    self.binn_map_set(id, BINN_INT64, &value, 0)
}

/// Reads a key-value pair from a binary map at the specified position.
///
/// # Arguments
/// * `ptr` - A reference to the binary map.
/// * `pos` - The position index of the key-value pair.
/// * `pid` - A mutable reference to store the key identifier.
/// * `ptype` - A mutable reference to store the value type.
/// * `psize` - A mutable reference to store the value size.
///
/// # Returns
/// An `Option<&T>` where `T` is the type of the value. Returns `None` if the key-value pair cannot be read.
pub fn binn_map_read_pair<T>(ptr: &BinnMap, pos: usize, pid: &mut i32, ptype: &mut i32, psize: &mut i32) -> Option<&T> {
    let mut value = Binn::default();
    if binn_map_get_pair(ptr, pos, pid, &mut value).is_err() {
        return None;
    }
    *ptype = value.type_;
    *psize = value.size;
    #[cfg(target_endian = "little")]
    return Some(store_value(&value));
    #[cfg(not(target_endian = "little"))]
    return Some(value.ptr);
}

/// Gets a key-value pair from a binary map at the specified position.
///
/// # Arguments
/// * `ptr` - A reference to the binary map.
/// * `pos` - The position index of the key-value pair.
/// * `pid` - A mutable reference to store the key identifier.
/// * `value` - A mutable reference to store the value.
///
/// # Returns
/// A `Result<(), Error>` indicating whether the operation was successful.
pub fn binn_map_get_pair(ptr: &BinnMap, pos: usize, pid: &mut i32, value: &mut Binn) -> Result<(), Error> {
    binn_read_pair(BinnType::Map, ptr, pos, pid, None, value)
}

/// Sets an unsigned 8-bit integer value in a Binn map.
///
/// # Arguments
/// * `map` - A mutable reference to the Binn map.
/// * `id` - The identifier of the key.
/// * `value` - The unsigned 8-bit integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_map_set_uint8(map: &mut Binn, id: i32, value: u8) -> Result<(), &'static str> {
    binn_map_set(map, id, BINN_UINT8, &value, 0)
}

/// Sets a 16-bit signed integer value in a Binn map.
///
/// # Arguments
/// * `id` - The identifier of the key.
/// * `value` - The 16-bit signed integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_map_set_int16(&mut self, id: i32, value: i16) -> Result<(), &'static str> {
    self.binn_map_set(id, BINN_INT16, &value, 0)
}

/// Sets a 32-bit unsigned integer value in a Binn map.
///
/// # Arguments
/// * `id` - The identifier of the key.
/// * `value` - The 32-bit unsigned integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_map_set_uint32(&mut self, id: i32, value: u32) -> Result<(), &'static str> {
    self.binn_map_set(id, BINN_UINT32, &value, 0)
}

/// Sets a key-value pair in the binary map with the value set to `NULL`.
///
/// # Arguments
/// * `id` - The key to set.
///
/// # Returns
/// Returns `Ok(())` if the key-value pair was successfully set, otherwise returns an error message.
pub fn binn_map_set_null(&mut self, id: i32) -> Result<(), &'static str> {
    self.binn_map_set(id, BINN_NULL, &[], 0)
}

impl Binn {
    /// Adds a value to the binary data structure based on the type.
    ///
    /// # Arguments
    /// * `binn_type` - The type of the binary data structure (list, map, or object).
    /// * `id` - The identifier for the value (used in maps).
    /// * `name` - The key name for the value (used in objects).
    /// * `type_` - The type of the value to add.
    /// * `pvalue` - A slice containing the value data.
    /// * `size` - The size of the value data.
    ///
    /// # Returns
    /// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
    pub fn binn_add_value(
        &mut self,
        binn_type: BinnType,
        id: i32,
        name: &str,
        type_: i32,
        pvalue: &[u8],
        size: usize,
    ) -> Result<(), &'static str> {
        match binn_type {
            BinnType::List => self.binn_list_add(type_, pvalue, size),
            BinnType::Map => self.binn_map_set(id, type_, pvalue, size),
            BinnType::Object => self.binn_object_set(name, type_, pvalue, size),
            _ => Err("Invalid binn type"),
        }
    }
}

/// Sets an 8-bit signed integer value in a Binn map.
///
/// # Arguments
/// * `id` - The identifier of the key.
/// * `value` - The 8-bit signed integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_map_set_int8(&mut self, id: i32, value: i8) -> Result<(), &'static str> {
    self.binn_map_set(id, BINN_INT8, &value, 0)
}

/// Sets a 32-bit signed integer value in a Binn map.
///
/// # Arguments
/// * `id` - The identifier of the key.
/// * `value` - The 32-bit signed integer value to set.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully set, otherwise returns an error message.
pub fn binn_map_set_int32(&mut self, id: i32, value: i32) -> Result<(), &'static str> {
    self.binn_map_set(id, BINN_INT32, &value, 0)
}

/// Advances the iterator to the next key-value pair in the binary map.
///
/// # Arguments
/// * `pid` - A mutable reference to store the current key's identifier.
/// * `value` - A mutable reference to store the current value.
///
/// # Returns
/// Returns `Ok(())` if the next key-value pair was successfully retrieved, otherwise returns an error message.
pub fn binn_map_next(&mut self, pid: &mut i32, value: &mut Binn) -> Result<(), &'static str> {
    self.binn_read_next_pair(BINN_MAP, pid, None, value)
}

/// Advances the iterator to the next key-value pair in the binary object.
///
/// # Arguments
/// * `iter` - A mutable reference to the iterator.
/// * `pkey` - A mutable reference to store the current key.
/// * `value` - A mutable reference to store the current value.
///
/// # Returns
/// Returns `Ok(true)` if the next key-value pair was successfully read, otherwise returns `Err("error message")`.
pub fn binn_object_next(iter: &mut BinnIter, pkey: &mut String, value: &mut Binn) -> Result<bool, &'static str> {
    binn_read_next_pair(BINN_OBJECT, iter, None, pkey, value)
}

/// Extracts a key-value pair from a binary map at the specified position.
///
/// # Arguments
/// * `map` - A reference to the binary map.
/// * `pos` - The position index of the key-value pair to extract.
/// * `pid` - A mutable reference to store the key identifier.
///
/// # Returns
/// Returns `Ok(Box<Binn>)` containing the extracted value if successful, otherwise returns an error message.
pub fn binn_map_pair(map: &Binn, pos: i32, pid: &mut i32) -> Result<Box<Binn>, &'static str> {
    let mut value = Box::new(Binn::default());
    if binn_read_pair(BINN_MAP, map, pos, Some(pid), None, &mut value).is_err() {
        return Err("Failed to read key-value pair");
    }
    value.allocated = true;
    Ok(value)
}

/// 从二进制对象中提取指定位置的键值对，并返回一个指向该值的指针。
///
/// # 参数
/// - `obj`: 指向二进制对象的引用。
/// - `pos`: 元素的位置索引（从1开始）。
/// - `pkey`: 用于存储键名的可变引用。
///
/// # 返回值
/// 返回 `Result<Box<Binn>, &'static str>`，表示成功时返回 `Box<Binn>`，失败时返回错误信息。
pub fn binn_object_pair(obj: &Binn, pos: usize, pkey: &mut String) -> Result<Box<Binn>, &'static str> {
    let mut value = Box::new(Binn::default());
    if binn_read_pair(BINN_OBJECT, obj, pos, None, Some(pkey), &mut value).is_err() {
        return Err("Failed to read pair");
    }
    value.allocated = true;
    Ok(value)
}

/// 从二进制映射中提取一个 8 位无符号整数值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 标识符，用于查找对应的值。
///
/// # 返回值
/// 返回 `Result<u8, &'static str>`，表示成功时返回提取的 8 位无符号整数值，失败时返回错误信息。
pub fn binn_map_uint8(map: &Binn, id: i32) -> Result<u8, &'static str> {
    let mut value: u8 = 0;
    binn_map_get(map, id, BINN_UINT8, &mut value, None)?;
    Ok(value)
}

/// Adds a `Binn` value to the list and automatically frees the value object.
///
/// # Arguments
/// * `value` - The `Binn` value to be added and freed.
///
/// # Returns
/// Returns `Ok(())` if the value was successfully added, otherwise returns an error message.
pub fn binn_list_add_new(&mut self, value: Box<Binn>) -> Result<(), &'static str> {
    self.binn_list_add_value(&value)?;
    Ok(())
}

/// 从二进制映射中获取指定键的无符号16位整数值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 键的标识符。
/// - `pvalue`: 用于存储提取的无符号16位整数值的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功提取并存储了值，否则返回错误信息。
pub fn binn_map_get_uint16(map: &Binn, id: i32, pvalue: &mut u16) -> Result<(), &'static str> {
    binn_map_get(map, id, BINN_UINT16, pvalue, None)
}

/// 从二进制映射中获取指定键对应的嵌套映射数据。
///
/// # 参数
/// - `id`: 要查找的键的标识符。
/// - `pvalue`: 用于存储结果的指针。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功获取并存储了值，否则返回错误信息。
pub fn binn_map_get_map(&self, id: i32, pvalue: &mut Option<Box<BinnMap>>) -> Result<(), &'static str> {
    self.binn_map_get(id, BINN_MAP, pvalue, None)
}

/// 从二进制映射中提取一个 32 位无符号整数值。
///
/// # 参数
/// - `id`: 要查找的标识符。
///
/// # 返回值
/// 返回 `Result<u32, &'static str>`，表示成功时返回 32 位无符号整数值，失败时返回错误信息。
pub fn binn_map_uint32(&self, id: i32) -> Result<u32, &'static str> {
    let mut value: u32 = 0;
    self.binn_map_get(id, BINN_UINT32, &mut value, None)?;
    Ok(value)
}

/// 从二进制对象中读取指定位置的键值对，并返回值的指针。
///
/// # 参数
/// - `ptr`: 指向二进制对象的引用。
/// - `pos`: 键值对的位置索引。
/// - `pkey`: 用于存储键名的可变引用。
/// - `ptype`: 可选参数，用于存储值类型的可变引用。
/// - `psize`: 可选参数，用于存储值大小的可变引用。
///
/// # 返回值
/// 返回 `Option<&dyn std::any::Any>`，表示值的指针，如果失败则返回 `None`。
pub fn binn_object_read_pair(
    ptr: &Binn,
    pos: i32,
    pkey: &mut String,
    ptype: Option<&mut i32>,
    psize: Option<&mut usize>,
) -> Option<&dyn std::any::Any> {
    let mut value = Binn::default();
    if binn_object_get_pair(ptr, pos, pkey, &mut value).is_err() {
        return None;
    }
    if let Some(ptype) = ptype {
        *ptype = value.type_;
    }
    if let Some(psize) = psize {
        *psize = value.size;
    }
    if cfg!(target_endian = "little") {
        value.store_value()
    } else {
        value.ptr.as_ref().map(|ptr| ptr.as_ref())
    }
}

/// 从二进制对象中获取指定位置的键值对。
///
/// # 参数
/// - `ptr`: 指向二进制对象的引用。
/// - `pos`: 键值对的位置索引。
/// - `pkey`: 用于存储键名的可变引用。
/// - `value`: 用于存储值的可变引用。
///
/// # 返回值
/// 返回 `Result<(), &'static str>`，表示操作是否成功。
pub fn binn_object_get_pair(
    ptr: &Binn,
    pos: i32,
    pkey: &mut String,
    value: &mut Binn,
) -> Result<(), &'static str> {
    binn_read_pair(BINN_OBJECT, ptr, pos, None, pkey, value)
}

/// 从二进制映射中提取一个32位整数值。
///
/// # 参数
/// - `id`: 标识符，用于查找映射中的值。
///
/// # 返回值
/// 返回 `Result<i32, &'static str>`，表示成功时返回32位整数，失败时返回错误信息。
pub fn binn_map_int32(&self, id: i32) -> Result<i32, &'static str> {
    let mut value = 0;
    self.binn_map_get(id, BINN_INT32, &mut value, None)?;
    Ok(value)
}

/// 从二进制映射中获取指定键的64位整数值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 键的标识符。
/// - `pvalue`: 用于存储提取的64位整数值的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功提取并存储了值，否则返回错误信息。
pub fn binn_map_get_int64(map: &Binn, id: i32, pvalue: &mut i64) -> Result<(), &'static str> {
    binn_map_get(map, id, BINN_INT64, pvalue, None)
}

/// Extracts a double-precision floating-point value from a binary map.
///
/// # Arguments
/// * `id` - The identifier of the key.
///
/// # Returns
/// Returns `Ok(f64)` containing the extracted value if successful, otherwise returns an error message.
pub fn binn_map_double(&self, id: i32) -> Result<f64, &'static str> {
    let mut value = 0.0;
    self.binn_map_get(id, BINN_FLOAT64, &mut value, None)?;
    Ok(value)
}

/// 从二进制映射中获取指定键对应的双精度浮点数值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 键的标识符。
/// - `pvalue`: 用于存储提取的双精度浮点数值的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功提取并存储了值，否则返回错误信息。
pub fn binn_map_get_double(map: &BinnMap, id: i32, pvalue: &mut f64) -> Result<(), &'static str> {
    binn_map_get(map, id, BINN_FLOAT64, pvalue, None)
}

/// 从二进制映射中获取指定 ID 的 8 位有符号整数值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 要查找的标识符。
/// - `pvalue`: 用于存储结果的 8 位有符号整数的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功获取并存储了值，否则返回错误信息。
pub fn binn_map_get_int8(map: &Binn, id: i32, pvalue: &mut i8) -> Result<(), &'static str> {
    binn_map_get(map, id, BINN_INT8, pvalue, None)
}

/// 从二进制映射中获取指定键的32位整数值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 键的标识符。
/// - `pvalue`: 用于存储提取的32位整数值的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功获取并存储了值，否则返回错误信息。
pub fn binn_map_get_int32(map: &BinnMap, id: i32, pvalue: &mut i32) -> Result<(), &'static str> {
    binn_map_get(map, id, BINN_INT32, pvalue, None)
}

/// 从二进制映射中提取一个 64 位无符号整数值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 要查找的标识符。
///
/// # 返回值
/// 返回 `Result<u64, &'static str>`，表示成功时返回提取的整数值，失败时返回错误信息。
pub fn binn_map_uint64(map: &Binn, id: i32) -> Result<u64, &'static str> {
    let mut value = 0u64;
    binn_map_get(map, id, BINN_UINT64, &mut value, None)?;
    Ok(value)
}

/// 从二进制映射中提取与指定 ID 关联的列表类型数据。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 标识符，用于查找对应的列表数据。
///
/// # 返回值
/// 返回 `Result<Option<Box<Binn>>, &'static str>`，表示成功时返回 `Box<Binn>`，失败时返回错误信息。
pub fn binn_map_list(map: &Binn, id: i32) -> Result<Option<Box<Binn>>, &'static str> {
    let mut value = Box::new(Binn::default());
    binn_map_get_value(map, id, &mut value)?;
    value.allocated = true;
    Ok(Some(value))
}

/// 从二进制映射中提取一个16位整数（`i16` 类型）的值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 标识符，用于查找对应的值。
///
/// # 返回值
/// 返回 `Result<i16, &'static str>`，表示成功时返回16位整数，失败时返回错误信息。
pub fn binn_map_int16(map: &BinnMap, id: i32) -> Result<i16, &'static str> {
    let mut value: i16 = 0;
    binn_map_get(map, id, BINN_INT16, &mut value, None)?;
    Ok(value)
}

/// 从二进制映射中提取与指定标识符关联的二进制大对象（BLOB）数据。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 标识符，用于查找对应的 BLOB 数据。
/// - `psize`: 用于存储 BLOB 数据大小的可变引用。
///
/// # 返回值
/// 返回 `Option<&[u8]>`，表示成功时返回 BLOB 数据的引用，失败时返回 `None`。
pub fn binn_map_blob(map: &Binn, id: i32, psize: &mut usize) -> Option<&[u8]> {
    let mut value = Vec::new();
    if binn_map_get(map, id, BINN_BLOB, &mut value, psize).is_ok() {
        Some(&value)
    } else {
        None
    }
}

/// 从二进制映射中提取一个 16 位无符号整数值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 标识符，用于查找对应的值。
///
/// # 返回值
/// 返回 `Result<u16, &'static str>`，表示成功时返回提取的 16 位无符号整数值，失败时返回错误信息。
pub fn binn_map_uint16(map: &Binn, id: i32) -> Result<u16, &'static str> {
    let mut value: u16 = 0;
    binn_map_get(map, id, BINN_UINT16, &mut value, None)?;
    Ok(value)
}

/// 从二进制映射中获取指定键的无符号8位整数值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 键的标识符。
/// - `pvalue`: 用于存储提取值的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功提取并存储了值，否则返回错误信息。
pub fn binn_map_get_uint8(map: &Binn, id: i32, pvalue: &mut u8) -> Result<(), &'static str> {
    binn_map_get(map, id, BINN_UINT8, pvalue, None)
}

/// 从二进制映射中获取指定键的字符串值。
///
/// # 参数
/// - `id`: 键的标识符。
/// - `pvalue`: 用于存储字符串指针的指针。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功获取并存储了字符串值，否则返回错误信息。
pub fn binn_map_get_str(&self, id: i32, pvalue: &mut Option<String>) -> Result<(), &'static str> {
    self.binn_map_get(id, BINN_STRING, pvalue, None)
}

/// 从二进制映射中获取指定键的浮点数值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 键的标识符。
/// - `pvalue`: 用于存储提取的浮点数值的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功获取并存储了值，否则返回错误信息。
pub fn binn_map_get_float(map: &Binn, id: i32, pvalue: &mut f32) -> Result<(), &'static str> {
    binn_map_get(map, id, BINN_FLOAT32, pvalue, None)
}

/// 从二进制映射中提取与指定标识符关联的字符串值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 标识符，用于查找对应的字符串值。
///
/// # 返回值
/// 返回 `Option<String>`，表示成功时返回字符串值，失败时返回 `None`。
pub fn binn_map_str(map: &BinnMap, id: i32) -> Option<String> {
    let mut value: *mut i8 = std::ptr::null_mut();
    if binn_map_get(map, id, BINN_STRING, &mut value, None).is_ok() {
        if !value.is_null() {
            let c_str = unsafe { std::ffi::CStr::from_ptr(value) };
            return Some(c_str.to_string_lossy().into_owned());
        }
    }
    None
}

/// 从二进制映射中获取指定键对应的对象类型值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 键的标识符。
/// - `pvalue`: 用于存储结果的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功获取并存储了值，否则返回错误信息。
pub fn binn_map_get_object(map: &Binn, id: i32, pvalue: &mut Option<Box<Binn>>) -> Result<(), &'static str> {
    binn_map_get(map, id, BINN_OBJECT, pvalue, None)
}

/// 从二进制映射中提取一个 8 位有符号整数值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 标识符，用于查找对应的值。
///
/// # 返回值
/// 返回 `Result<i8, &'static str>`，表示成功时返回提取的 8 位有符号整数值，失败时返回错误信息。
pub fn binn_map_int8(map: &Binn, id: i32) -> Result<i8, &'static str> {
    let mut value: i8 = 0;
    binn_map_get(map, id, BINN_INT8, &mut value, None)?;
    Ok(value)
}

/// Checks if the value associated with the given ID in the binary map is `NULL`.
///
/// # Arguments
/// * `id` - The identifier of the key to check.
///
/// # Returns
/// Returns `Ok(true)` if the value is `NULL`, otherwise returns `Ok(false)`.
/// If the key does not exist or an error occurs, returns `Err(&'static str)`.
pub fn binn_map_null(&self, id: i32) -> Result<bool, &'static str> {
    self.binn_map_get(id, BINN_NULL, &mut [], None)
}

/// 从二进制映射中提取一个 64 位有符号整数值。
///
/// # 参数
/// - `id`: 标识符，用于查找对应的值。
///
/// # 返回值
/// 返回 `Result<i64, &'static str>`，表示成功时返回 64 位有符号整数值，失败时返回错误信息。
pub fn binn_map_int64(&self, id: i32) -> Result<i64, &'static str> {
    let mut value: i64 = 0;
    self.binn_map_get(id, BINN_INT64, &mut value, None)?;
    Ok(value)
}

/// 从二进制映射中提取与给定 ID 关联的对象类型数据，并返回该对象的指针。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 标识符，用于查找映射中的值。
///
/// # 返回值
/// 返回 `Result<Option<Box<Binn>>, &'static str>`，表示成功时返回 `Box<Binn>`，失败时返回错误信息。
pub fn binn_map_object(map: &Binn, id: i32) -> Result<Option<Box<Binn>>, &'static str> {
    let mut value = None;
    binn_map_get(map, id, BINN_OBJECT, &mut value, None)?;
    Ok(value)
}

/// 从二进制映射中获取指定键的布尔值。
///
/// # 参数
/// - `map`: 指向二进制映射的引用。
/// - `id`: 键的标识符。
/// - `pvalue`: 用于存储布尔值的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功获取并存储了布尔值，否则返回错误信息。
pub fn binn_map_get_bool(map: &Binn, id: i32, pvalue: &mut bool) -> Result<(), &'static str> {
    binn_map_get(map, id, BINN_BOOL, pvalue, None)
}

/// 从二进制映射中提取一个布尔值。
///
/// # 参数
/// - `id`: 标识符，用于查找映射中的布尔值。
///
/// # 返回值
/// 返回 `Result<bool, &'static str>`，表示成功时返回布尔值，失败时返回错误信息。
pub fn binn_map_bool(&self, id: i32) -> Result<bool, &'static str> {
    let mut value = false;
    self.binn_map_get(id, BINN_BOOL, &mut value, None)?;
    Ok(value)
}

/// Extracts a boolean value from a binary object.
///
/// # Arguments
/// * `key` - The key associated with the boolean value.
/// * `pvalue` - A mutable reference to store the extracted boolean value.
///
/// # Returns
/// Returns `Ok(())` if the boolean value was successfully extracted, otherwise returns an error message.
pub fn binn_object_get_bool(&self, key: &str, pvalue: &mut bool) -> Result<(), &'static str> {
    self.binn_object_get(key, BINN_BOOL, pvalue, None)
}

/// Extracts a floating-point value from a binary map.
///
/// # Arguments
/// * `id` - The identifier of the key.
///
/// # Returns
/// Returns `Ok(f32)` containing the extracted value if successful, otherwise returns an error message.
pub fn binn_map_float(&self, id: i32) -> Result<f32, &'static str> {
    let mut value = 0.0f32;
    self.binn_map_get(id, BINN_FLOAT32, &mut value, None)?;
    Ok(value)
}

/// 从二进制对象中提取一个无符号8位整数（`uint8`）值。
///
/// # 参数
/// - `key`: 键名，用于查找对应的值。
/// - `pvalue`: 用于存储提取值的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功提取并存储了值，否则返回错误信息。
pub fn binn_object_get_uint8(&self, key: &str, pvalue: &mut u8) -> Result<(), &'static str> {
    self.binn_object_get(key, BINN_UINT8, pvalue, None)
}

/// 从二进制对象中获取与指定键关联的字符串值。
///
/// # 参数
/// - `key`: 要查找的键名。
/// - `pvalue`: 用于存储字符串的可变引用。
///
/// # 返回值
/// 返回 `Ok(())` 如果成功获取并存储了字符串值，否则返回错误信息。
pub fn binn_object_get_str(&self, key: &str, pvalue: &mut String) -> Result<(), &'static str> {
    self.binn_object_get(key, BINN_STRING, pvalue, None)
}