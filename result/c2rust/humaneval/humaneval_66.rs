use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn digitSum(mut s: *const libc::c_char) -> libc::c_int {
    let mut sum: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i as libc::c_ulong) < strlen(s) {
        if *s.offset(i as isize) as libc::c_int >= 65 as libc::c_int
            && *s.offset(i as isize) as libc::c_int <= 90 as libc::c_int
        {
            sum += *s.offset(i as isize) as libc::c_int;
        }
        i += 1;
        i;
    }
    return sum;
}
