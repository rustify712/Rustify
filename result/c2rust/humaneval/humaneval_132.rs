use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn is_nested(mut str: *const libc::c_char) -> bool {
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut maxcount: libc::c_int = 0 as libc::c_int;
    let mut length: libc::c_int = strlen(str) as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < length {
        if *str.offset(i as isize) as libc::c_int == '[' as i32 {
            count += 1 as libc::c_int;
        }
        if *str.offset(i as isize) as libc::c_int == ']' as i32 {
            count -= 1 as libc::c_int;
        }
        if count < 0 as libc::c_int {
            count = 0 as libc::c_int;
        }
        if count > maxcount {
            maxcount = count;
        }
        if count <= maxcount - 2 as libc::c_int {
            return 1 as libc::c_int != 0;
        }
        i += 1;
        i;
    }
    return 0 as libc::c_int != 0;
}
