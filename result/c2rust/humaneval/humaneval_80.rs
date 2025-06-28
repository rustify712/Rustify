use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn is_happy(mut s: *const libc::c_char) -> bool {
    let mut length: libc::c_int = strlen(s) as libc::c_int;
    if length < 3 as libc::c_int {
        return 0 as libc::c_int != 0;
    }
    let mut i: libc::c_int = 2 as libc::c_int;
    while i < length {
        if *s.offset(i as isize) as libc::c_int
            == *s.offset((i - 1 as libc::c_int) as isize) as libc::c_int
            || *s.offset(i as isize) as libc::c_int
                == *s.offset((i - 2 as libc::c_int) as isize) as libc::c_int
        {
            return 0 as libc::c_int != 0;
        }
        i += 1;
        i;
    }
    return 1 as libc::c_int != 0;
}
