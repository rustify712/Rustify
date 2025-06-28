use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn encrypt(mut s: *mut libc::c_char) -> *mut libc::c_char {
    let mut len: libc::c_int = strlen(s) as libc::c_int;
    let mut out: *mut libc::c_char = malloc((len + 1 as libc::c_int) as libc::c_ulong)
        as *mut libc::c_char;
    let mut i: libc::c_int = 0;
    i = 0 as libc::c_int;
    while i < len {
        let mut w: libc::c_int = (*s.offset(i as isize) as libc::c_int + 4 as libc::c_int
            - 'a' as i32) % 26 as libc::c_int + 'a' as i32;
        *out.offset(i as isize) = w as libc::c_char;
        i += 1;
        i;
    }
    *out.offset(len as isize) = '\0' as i32 as libc::c_char;
    return out;
}
