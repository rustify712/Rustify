use ::libc;
extern "C" {
    fn tolower(_: libc::c_int) -> libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn string_hash(mut string: *mut libc::c_void) -> libc::c_uint {
    let mut result: libc::c_uint = 5381 as libc::c_int as libc::c_uint;
    let mut p: *mut libc::c_uchar = 0 as *mut libc::c_uchar;
    p = string as *mut libc::c_uchar;
    while *p as libc::c_int != '\0' as i32 {
        result = (result << 5 as libc::c_int)
            .wrapping_add(result)
            .wrapping_add(*p as libc::c_uint);
        p = p.offset(1);
        p;
    }
    return result;
}
#[no_mangle]
pub unsafe extern "C" fn string_nocase_hash(
    mut string: *mut libc::c_void,
) -> libc::c_uint {
    let mut result: libc::c_uint = 5381 as libc::c_int as libc::c_uint;
    let mut p: *mut libc::c_uchar = 0 as *mut libc::c_uchar;
    p = string as *mut libc::c_uchar;
    while *p as libc::c_int != '\0' as i32 {
        result = (result << 5 as libc::c_int)
            .wrapping_add(result)
            .wrapping_add(tolower(*p as libc::c_int) as libc::c_uint);
        p = p.offset(1);
        p;
    }
    return result;
}
