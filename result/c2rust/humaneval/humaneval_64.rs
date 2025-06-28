use ::libc;
extern "C" {
    fn strchr(_: *const libc::c_char, _: libc::c_int) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn vowels_count(mut s: *const libc::c_char) -> libc::c_int {
    let mut vowels: *const libc::c_char = b"aeiouAEIOU\0" as *const u8
        as *const libc::c_char;
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut len: libc::c_int = strlen(s) as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < len {
        if !(strchr(vowels, *s.offset(i as isize) as libc::c_int)).is_null() {
            count += 1;
            count;
        }
        i += 1;
        i;
    }
    if len > 0 as libc::c_int
        && (*s.offset((len - 1 as libc::c_int) as isize) as libc::c_int == 'y' as i32
            || *s.offset((len - 1 as libc::c_int) as isize) as libc::c_int == 'Y' as i32)
    {
        count += 1;
        count;
    }
    return count;
}
