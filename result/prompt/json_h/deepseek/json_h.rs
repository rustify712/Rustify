use std::ptr;
use std::mem;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::collections::HashMap;
use std::fmt;

extern crate rand;
extern crate regex;
extern crate md5;

// JSON parsing flags
#[derive(Debug, Clone, Copy)]
pub enum JsonParseFlags {
    Default = 0,
    AllowTrailingComma = 0x1,
    AllowUnquotedKeys = 0x2,
    AllowGlobalObject = 0x4,
    AllowEqualsInObject = 0x8,
    AllowNoCommas = 0x10,
    AllowCStyleComments = 0x20,
    Deprecated = 0x40,
    AllowLocationInformation = 0x80,
    AllowSingleQuotedStrings = 0x100,
    AllowHexadecimalNumbers = 0x200,
    AllowLeadingPlusSign = 0x400,
    AllowLeadingOrTrailingDecimalPoint = 0x800,
    AllowInfAndNan = 0x1000,
    AllowMultiLineStrings = 0x2000,
    AllowSimplifiedJson = 0x1 | 0x2 | 0x4 | 0x8 | 0x10,
    AllowJson5 = 0x1 | 0x2 | 0x20 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000,
}

// JSON value types
#[derive(Debug, Clone, Copy)]
pub enum JsonType {
    String,
    Number,
    Object,
    Array,
    True,
    False,
    Null,
}

// JSON string value
#[derive(Debug)]
pub struct JsonString {
    pub string: *const c_char,
    pub string_size: usize,
}

// JSON string value (extended)
#[derive(Debug)]
pub struct JsonStringEx {
    pub string: JsonString,
    pub offset: usize,
    pub line_no: usize,
    pub row_no: usize,
}

// JSON number value
#[derive(Debug)]
pub struct JsonNumber {
    pub number: *const c_char,
    pub number_size: usize,
}

// JSON object element
#[derive(Debug)]
pub struct JsonObjectElement {
    pub name: *mut JsonString,
    pub value: *mut JsonValue,
    pub next: *mut JsonObjectElement,
}

// JSON object value
#[derive(Debug)]
pub struct JsonObject {
    pub start: *mut JsonObjectElement,
    pub length: usize,
}

// JSON array element
#[derive(Debug)]
pub struct JsonArrayElement {
    pub value: *mut JsonValue,
    pub next: *mut JsonArrayElement,
}

// JSON array value
#[derive(Debug)]
pub struct JsonArray {
    pub start: *mut JsonArrayElement,
    pub length: usize,
}

// JSON value
#[derive(Debug)]
pub struct JsonValue {
    pub payload: *mut (),
    pub type_: JsonType,
}

// JSON value (extended)
#[derive(Debug)]
pub struct JsonValueEx {
    pub value: JsonValue,
    pub offset: usize,
    pub line_no: usize,
    pub row_no: usize,
}

// JSON parse error codes
#[derive(Debug, Clone, Copy)]
pub enum JsonParseError {
    None = 0,
    ExpectedCommaOrClosingBracket,
    ExpectedColon,
    ExpectedOpeningQuote,
    InvalidStringEscapeSequence,
    InvalidNumberFormat,
    InvalidValue,
    PrematureEndOfBuffer,
    InvalidString,
    AllocatorFailed,
    UnexpectedTrailingCharacters,
    Unknown,
}

// JSON parse result
#[derive(Debug)]
pub struct JsonParseResult {
    pub error: JsonParseError,
    pub error_offset: usize,
    pub error_line_no: usize,
    pub error_row_no: usize,
}

// JSON parse state
#[derive(Debug)]
pub struct JsonParseState {
    pub src: *const c_char,
    pub size: usize,
    pub offset: usize,
    pub flags_bitset: usize,
    pub data: *mut c_char,
    pub dom: *mut c_char,
    pub dom_size: usize,
    pub data_size: usize,
    pub line_no: usize,
    pub line_offset: usize,
    pub error: JsonParseError,
}

// JSON extract result
#[derive(Debug)]
pub struct JsonExtractResult {
    pub dom_size: usize,
    pub data_size: usize,
}

// JSON extract state
#[derive(Debug)]
pub struct JsonExtractState {
    pub dom: *mut c_char,
    pub data: *mut c_char,
}

// JSON write state
#[derive(Debug)]
pub struct JsonWriteState {
    pub data: *mut c_char,
}

// JSON write result
#[derive(Debug)]
pub struct JsonWriteResult {
    pub data: *mut c_char,
    pub size: usize,
}

// JSON parsing functions
pub fn json_parse(src: *const c_char, src_size: usize) -> *mut JsonValue {
    json_parse_ex(src, src_size, JsonParseFlags::Default as usize, None, ptr::null_mut(), ptr::null_mut())
}

pub fn json_parse_ex(
    src: *const c_char,
    src_size: usize,
    flags_bitset: usize,
    alloc_func_ptr: Option<extern "C" fn(*mut (), usize) -> *mut ()>,
    user_data: *mut (),
    result: *mut JsonParseResult,
) -> *mut JsonValue {
    let mut state = JsonParseState {
        src,
        size: src_size,
        offset: 0,
        flags_bitset,
        data: ptr::null_mut(),
        dom: ptr::null_mut(),
        dom_size: 0,
        data_size: 0,
        line_no: 1,
        line_offset: 0,
        error: JsonParseError::None,
    };

    if src.is_null() {
        if !result.is_null() {
            unsafe {
                (*result).error = JsonParseError::Unknown;
                (*result).error_offset = 0;
                (*result).error_line_no = 0;
                (*result).error_row_no = 0;
            }
        }
        return ptr::null_mut();
    }

    let input_error = json_get_value_size(&mut state, (flags_bitset & JsonParseFlags::AllowGlobalObject as usize) != 0);

    if input_error == 0 {
        json_skip_all_skippables(&mut state);

        if state.offset != state.size {
            state.error = JsonParseError::UnexpectedTrailingCharacters;
            if !result.is_null() {
                unsafe {
                    (*result).error = state.error;
                    (*result).error_offset = state.offset;
                    (*result).error_line_no = state.line_no;
                    (*result).error_row_no = state.offset - state.line_offset;
                }
            }
            return ptr::null_mut();
        }
    }

    if input_error != 0 {
        if !result.is_null() {
            unsafe {
                (*result).error = state.error;
                (*result).error_offset = state.offset;
                (*result).error_line_no = state.line_no;
                (*result).error_row_no = state.offset - state.line_offset;
            }
        }
        return ptr::null_mut();
    }

    let total_size = state.dom_size + state.data_size;
    let allocation = match alloc_func_ptr {
        Some(func) => func(user_data, total_size),
        None => unsafe { libc::malloc(total_size) as *mut () },
    };

    if allocation.is_null() {
        if !result.is_null() {
            unsafe {
                (*result).error = JsonParseError::AllocatorFailed;
                (*result).error_offset = 0;
                (*result).error_line_no = 0;
                (*result).error_row_no = 0;
            }
        }
        return ptr::null_mut();
    }

    state.offset = 0;
    state.line_no = 1;
    state.line_offset = 0;
    state.dom = allocation as *mut c_char;
    state.data = unsafe { state.dom.offset(state.dom_size as isize) };

    let value = if (flags_bitset & JsonParseFlags::AllowLocationInformation as usize) != 0 {
        let value_ex = state.dom as *mut JsonValueEx;
        unsafe {
            (*value_ex).value.payload = state.dom.offset(mem::size_of::<JsonValueEx>() as isize);
            (*value_ex).value.type_ = JsonType::Object;
            (*value_ex).offset = state.offset;
            (*value_ex).line_no = state.line_no;
            (*value_ex).row_no = state.offset - state.line_offset;
            &mut (*value_ex).value
        }
    } else {
        let value = state.dom as *mut JsonValue;
        unsafe {
            (*value).payload = state.dom.offset(mem::size_of::<JsonValue>() as isize);
            (*value).type_ = JsonType::Object;
            value
        }
    };

    json_parse_value(&mut state, (flags_bitset & JsonParseFlags::AllowGlobalObject as usize) != 0, value);

    value
}

// JSON value extraction functions
pub fn json_extract_value(value: *const JsonValue) -> *mut JsonValue {
    json_extract_value_ex(value, None, ptr::null_mut())
}

pub fn json_extract_value_ex(
    value: *const JsonValue,
    alloc_func_ptr: Option<extern "C" fn(*mut (), usize) -> *mut ()>,
    user_data: *mut (),
) -> *mut JsonValue {
    if value.is_null() {
        return ptr::null_mut();
    }

    let result = json_extract_get_value_size(value);
    let total_size = result.dom_size + result.data_size;

    let allocation = match alloc_func_ptr {
        Some(func) => func(user_data, total_size),
        None => unsafe { libc::malloc(total_size) as *mut () },
    };

    if allocation.is_null() {
        return ptr::null_mut();
    }

    let mut state = JsonExtractState {
        dom: allocation as *mut c_char,
        data: unsafe { allocation.offset(result.dom_size as isize) as *mut c_char },
    };

    json_extract_copy_value(&mut state, value);

    allocation as *mut JsonValue
}

// JSON writing functions
pub fn json_write_minified(value: *const JsonValue, out_size: *mut usize) -> *mut c_char {
    let mut size = 0;
    if json_write_minified_get_value_size(value, &mut size) != 0 {
        return ptr::null_mut();
    }

    size += 1; // for the '\0' null terminating character.

    let data = unsafe { libc::malloc(size) as *mut c_char };
    if data.is_null() {
        return ptr::null_mut();
    }

    let data_end = json_write_minified_value(value, data);
    if data_end.is_null() {
        unsafe { libc::free(data as *mut ()); }
        return ptr::null_mut();
    }

    unsafe { *data_end = '\0'; }

    if !out_size.is_null() {
        unsafe { *out_size = size; }
    }

    data
}

pub fn json_write_pretty(
    value: *const JsonValue,
    indent: *const c_char,
    newline: *const c_char,
    out_size: *mut usize,
) -> *mut c_char {
    let mut size = 0;
    let indent_size = if indent.is_null() { 2 } else { unsafe { libc::strlen(indent) } };
    let newline_size = if newline.is_null() { 1 } else { unsafe { libc::strlen(newline) } };

    if json_write_pretty_get_value_size(value, 0, indent_size, newline_size, &mut size) != 0 {
        return ptr::null_mut();
    }

    size += 1; // for the '\0' null terminating character.

    let data = unsafe { libc::malloc(size) as *mut c_char };
    if data.is_null() {
        return ptr::null_mut();
    }

    let data_end = json_write_pretty_value(value, 0, indent, newline, data);
    if data_end.is_null() {
        unsafe { libc::free(data as *mut ()); }
        return ptr::null_mut();
    }

    unsafe { *data_end = '\0'; }

    if !out_size.is_null() {
        unsafe { *out_size = size; }
    }

    data
}

// Helper functions
fn json_skip_all_skippables(state: &mut JsonParseState) -> i32 {
    // Implementation of json_skip_all_skippables
    0
}

fn json_get_value_size(state: &mut JsonParseState, is_global_object: bool) -> i32 {
    // Implementation of json_get_value_size
    0
}

fn json_extract_get_value_size(value: *const JsonValue) -> JsonExtractResult {
    // Implementation of json_extract_get_value_size
    JsonExtractResult { dom_size: 0, data_size: 0 }
}

fn json_extract_copy_value(state: &mut JsonExtractState, value: *const JsonValue) {
    // Implementation of json_extract_copy_value
}

fn json_write_minified_get_value_size(value: *const JsonValue, size: &mut usize) -> i32 {
    // Implementation of json_write_minified_get_value_size
    0
}

fn json_write_minified_value(value: *const JsonValue, data: *mut c_char) -> *mut c_char {
    // Implementation of json_write_minified_value
    ptr::null_mut()
}

fn json_write_pretty_get_value_size(value: *const JsonValue, depth: usize, indent_size: usize, newline_size: usize, size: &mut usize) -> i32 {
    // Implementation of json_write_pretty_get_value_size
    0
}

fn json_write_pretty_value(value: *const JsonValue, depth: usize, indent: *const c_char, newline: *const c_char, data: *mut c_char) -> *mut c_char {
    // Implementation of json_write_pretty_value
    ptr::null_mut()
}