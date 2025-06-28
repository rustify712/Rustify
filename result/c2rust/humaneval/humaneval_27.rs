use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn flip_case(mut str: *const libc::c_char) -> *mut libc::c_char {
    let mut length: libc::c_int = strlen(str) as libc::c_int;
    let mut out: *mut libc::c_char = malloc((length + 1 as libc::c_int) as libc::c_ulong)
        as *mut libc::c_char;
    if out.is_null() {
        return 0 as *mut libc::c_char;
    }
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < length {
        let mut w: libc::c_char = *str.offset(i as isize);
        if w as libc::c_int >= 'a' as i32 && w as libc::c_int <= 'z' as i32 {
            w = (w as libc::c_int - 32 as libc::c_int) as libc::c_char;
        } else if w as libc::c_int >= 'A' as i32 && w as libc::c_int <= 'Z' as i32 {
            w = (w as libc::c_int + 32 as libc::c_int) as libc::c_char;
        }
        *out.offset(i as isize) = w;
        i += 1;
        i;
    }
    *out.offset(length as isize) = '\0' as i32 as libc::c_char;
    return out;
}
