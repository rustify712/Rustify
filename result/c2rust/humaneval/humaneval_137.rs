use ::libc;
extern "C" {
    fn strtod(_: *const libc::c_char, _: *mut *mut libc::c_char) -> libc::c_double;
    fn strdup(_: *const libc::c_char) -> *mut libc::c_char;
    fn strchr(_: *const libc::c_char, _: libc::c_int) -> *mut libc::c_char;
}
pub type ValueType = libc::c_uint;
pub const TYPE_STRING: ValueType = 2;
pub const TYPE_DOUBLE: ValueType = 1;
pub const TYPE_INT: ValueType = 0;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct AnyValue {
    pub type_0: ValueType,
    pub c2rust_unnamed: C2RustUnnamed,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub int_value: libc::c_int,
    pub double_value: libc::c_double,
    pub string_value: *mut libc::c_char,
}
#[inline]
unsafe extern "C" fn atof(mut __nptr: *const libc::c_char) -> libc::c_double {
    return strtod(__nptr, 0 as *mut libc::c_void as *mut *mut libc::c_char);
}
#[no_mangle]
pub unsafe extern "C" fn compare_one(mut a: AnyValue, mut b: AnyValue) -> AnyValue {
    let mut numa: libc::c_double = 0.;
    let mut numb: libc::c_double = 0.;
    let mut out: AnyValue = AnyValue {
        type_0: TYPE_INT,
        c2rust_unnamed: C2RustUnnamed { int_value: 0 },
    };
    if a.type_0 as libc::c_uint == TYPE_STRING as libc::c_int as libc::c_uint {
        let mut s: *mut libc::c_char = a.c2rust_unnamed.string_value;
        let mut comma: *mut libc::c_char = strchr(s, ',' as i32);
        if !comma.is_null() {
            *comma = '.' as i32 as libc::c_char;
        }
        numa = atof(s);
    } else if a.type_0 as libc::c_uint == TYPE_INT as libc::c_int as libc::c_uint {
        numa = a.c2rust_unnamed.int_value as libc::c_double;
    } else if a.type_0 as libc::c_uint == TYPE_DOUBLE as libc::c_int as libc::c_uint {
        numa = a.c2rust_unnamed.double_value;
    }
    if b.type_0 as libc::c_uint == TYPE_STRING as libc::c_int as libc::c_uint {
        let mut s_0: *mut libc::c_char = b.c2rust_unnamed.string_value;
        let mut comma_0: *mut libc::c_char = strchr(s_0, ',' as i32);
        if !comma_0.is_null() {
            *comma_0 = '.' as i32 as libc::c_char;
        }
        numb = atof(s_0);
    } else if b.type_0 as libc::c_uint == TYPE_INT as libc::c_int as libc::c_uint {
        numb = b.c2rust_unnamed.int_value as libc::c_double;
    } else if b.type_0 as libc::c_uint == TYPE_DOUBLE as libc::c_int as libc::c_uint {
        numb = b.c2rust_unnamed.double_value;
    }
    if numa == numb {
        out.type_0 = TYPE_STRING;
        out
            .c2rust_unnamed
            .string_value = strdup(b"None\0" as *const u8 as *const libc::c_char);
        return out;
    } else if numa < numb {
        return b
    } else {
        return a
    };
}
