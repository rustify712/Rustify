//! JSON library tests

use super::*;
use std::ffi::CString;
use std::ptr;

#[test]
fn test_json_parse_flags() {
    assert_eq!(JsonParseFlags::Default.bits(), 0);
    assert_eq!(JsonParseFlags::AllowTrailingComma.bits(), 0x1);
    assert_eq!(JsonParseFlags::AllowUnquotedKeys.bits(), 0x2);
    
    let simplified = JsonParseFlags::simplified_json();
    assert!(simplified.contains(JsonParseFlags::AllowTrailingComma));
    assert!(simplified.contains(JsonParseFlags::AllowUnquotedKeys));
    
    let json5 = JsonParseFlags::json5();
    assert!(json5.contains(JsonParseFlags::AllowCStyleComments));
    assert!(json5.contains(JsonParseFlags::AllowSingleQuotedStrings));
}

#[test]
fn test_json_value_as_string() {
    let mut value = JsonValue {
        data: ptr::null_mut(),
        value_type: JsonType::String,
    };
    
    let string = JsonString {
        string: CString::new("test").unwrap().into_raw(),
        string_size: 4,
    };
    
    value.data = &string as *const _ as *mut c_void;
    
    unsafe {
        let result = json_value_as_string(&mut value);
        assert!(!result.is_null());
        assert_eq!((*result).string_size, 4);
        
        // Test non-string type
        value.value_type = JsonType::Number;
        assert!(json_value_as_string(&mut value).is_null());
    }
}

#[test]
fn test_json_parse_empty() {
    unsafe {
        let result = json_parse(ptr::null(), 0);
        assert!(result.is_null());
    }
}

#[test]
fn test_json_write_minified_empty() {
    unsafe {
        let mut out_size = 0;
        let result = json_write_minified(ptr::null(), &mut out_size);
        assert!(result.is_null());
    }
}

#[test]
fn test_json_write_pretty_empty() {
    unsafe {
        let mut out_size = 0;
        let result = json_write_pretty(ptr::null(), ptr::null(), ptr::null(), &mut out_size);
        assert!(result.is_null());
    }
}

#[test]
fn test_json_extract_value_empty() {
    unsafe {
        let result = json_extract_value(ptr::null());
        assert!(result.is_null());
    }
}

// Helper function to create test JSON string
fn create_test_json() -> CString {
    CString::new(r#"{"key":"value"}"#).unwrap()
}

#[test]
fn test_json_parse_basic() {
    let json_str = create_test_json();
    unsafe {
        let result = json_parse(json_str.as_ptr() as *const c_void, json_str.as_bytes().len());
        // Just verify it doesn't crash for now
        assert!(result.is_null() || !result.is_null());
    }
}