use ::libc;
extern "C" {
    fn strchr(_: *const libc::c_char, _: libc::c_int) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn hex_key(mut num: *const libc::c_char) -> libc::c_int {
    let mut key: *const libc::c_char = b"2357BD\0" as *const u8 as *const libc::c_char;
    let mut out: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i as libc::c_ulong) < strlen(num) {
        if !(strchr(key, *num.offset(i as isize) as libc::c_int)).is_null() {
            out += 1;
            out;
        }
        i += 1;
        i;
    }
    return out;
}
