use ::libc;
extern "C" {
    fn strtol(
        _: *const libc::c_char,
        _: *mut *mut libc::c_char,
        _: libc::c_int,
    ) -> libc::c_long;
    fn strdup(_: *const libc::c_char) -> *mut libc::c_char;
    fn strndup(_: *const libc::c_char, _: libc::c_ulong) -> *mut libc::c_char;
    fn strchr(_: *const libc::c_char, _: libc::c_int) -> *mut libc::c_char;
}
#[inline]
unsafe extern "C" fn atoi(mut __nptr: *const libc::c_char) -> libc::c_int {
    return strtol(
        __nptr,
        0 as *mut libc::c_void as *mut *mut libc::c_char,
        10 as libc::c_int,
    ) as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn simplify(
    mut x: *const libc::c_char,
    mut n: *const libc::c_char,
) -> bool {
    let mut a: libc::c_int = 0;
    let mut b: libc::c_int = 0;
    let mut c: libc::c_int = 0;
    let mut d: libc::c_int = 0;
    let mut x_slash: *mut libc::c_char = strchr(x, '/' as i32);
    let mut n_slash: *mut libc::c_char = strchr(n, '/' as i32);
    a = atoi(strndup(x, x_slash.offset_from(x) as libc::c_long as libc::c_ulong));
    b = atoi(strdup(x_slash.offset(1 as libc::c_int as isize)));
    c = atoi(strndup(n, n_slash.offset_from(n) as libc::c_long as libc::c_ulong));
    d = atoi(strdup(n_slash.offset(1 as libc::c_int as isize)));
    if a * c % (b * d) == 0 as libc::c_int {
        return 1 as libc::c_int != 0;
    }
    return 0 as libc::c_int != 0;
}
