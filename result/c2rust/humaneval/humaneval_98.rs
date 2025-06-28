use ::libc;
extern "C" {
    fn strchr(_: *const libc::c_char, _: libc::c_int) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn count_upper(mut s: *const libc::c_char) -> libc::c_int {
    let mut uvowel: *const libc::c_char = b"AEIOU\0" as *const u8 as *const libc::c_char;
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut length: libc::c_int = strlen(s) as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i * 2 as libc::c_int) < length {
        if !(strchr(uvowel, *s.offset((i * 2 as libc::c_int) as isize) as libc::c_int))
            .is_null()
        {
            count += 1;
            count;
        }
        i += 1;
        i;
    }
    return count;
}
