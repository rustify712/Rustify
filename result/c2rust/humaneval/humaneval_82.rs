use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn prime_length(mut str: *const libc::c_char) -> bool {
    let mut l: libc::c_int = 0;
    let mut i: libc::c_int = 0;
    l = strlen(str) as libc::c_int;
    if l < 2 as libc::c_int {
        return 0 as libc::c_int != 0;
    }
    i = 2 as libc::c_int;
    while i * i <= l {
        if l % i == 0 as libc::c_int {
            return 0 as libc::c_int != 0;
        }
        i += 1;
        i;
    }
    return 1 as libc::c_int != 0;
}
