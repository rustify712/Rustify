use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn longest(
    mut strings: *mut *mut libc::c_char,
    mut size: libc::c_int,
) -> *mut libc::c_char {
    if size == 0 as libc::c_int {
        return 0 as *mut libc::c_char;
    }
    let mut out: *mut libc::c_char = *strings.offset(0 as libc::c_int as isize);
    let mut i: libc::c_int = 1 as libc::c_int;
    while i < size {
        if strlen(*strings.offset(i as isize)) > strlen(out) {
            out = *strings.offset(i as isize);
        }
        i += 1;
        i;
    }
    return out;
}
