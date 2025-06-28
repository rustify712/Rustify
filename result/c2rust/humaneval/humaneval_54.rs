use ::libc;
extern "C" {
    fn strchr(_: *const libc::c_char, _: libc::c_int) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn same_chars(
    mut s0: *const libc::c_char,
    mut s1: *const libc::c_char,
) -> libc::c_int {
    let mut len0: libc::c_int = strlen(s0) as libc::c_int;
    let mut len1: libc::c_int = strlen(s1) as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < len0 {
        if (strchr(s1, *s0.offset(i as isize) as libc::c_int)).is_null() {
            return 0 as libc::c_int;
        }
        i += 1;
        i;
    }
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while i_0 < len1 {
        if (strchr(s0, *s1.offset(i_0 as isize) as libc::c_int)).is_null() {
            return 0 as libc::c_int;
        }
        i_0 += 1;
        i_0;
    }
    return 1 as libc::c_int;
}
