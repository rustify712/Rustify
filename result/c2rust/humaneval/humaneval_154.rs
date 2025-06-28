use ::libc;
extern "C" {
    fn strncpy(
        _: *mut libc::c_char,
        _: *const libc::c_char,
        _: libc::c_ulong,
    ) -> *mut libc::c_char;
    fn strstr(_: *const libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn cycpattern_check(
    mut a: *const libc::c_char,
    mut b: *const libc::c_char,
) -> libc::c_int {
    let mut len_a: libc::c_int = strlen(a) as libc::c_int;
    let mut len_b: libc::c_int = strlen(b) as libc::c_int;
    let vla = (len_b + 1 as libc::c_int) as usize;
    let mut rotate: Vec::<libc::c_char> = ::std::vec::from_elem(0, vla);
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < len_b {
        strncpy(rotate.as_mut_ptr(), b.offset(i as isize), (len_b - i) as libc::c_ulong);
        strncpy(
            rotate.as_mut_ptr().offset(len_b as isize).offset(-(i as isize)),
            b,
            i as libc::c_ulong,
        );
        *rotate.as_mut_ptr().offset(len_b as isize) = '\0' as i32 as libc::c_char;
        if !(strstr(a, rotate.as_mut_ptr())).is_null() {
            return 1 as libc::c_int;
        }
        i += 1;
        i;
    }
    return 0 as libc::c_int;
}
