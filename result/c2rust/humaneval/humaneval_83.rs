use ::libc;
#[no_mangle]
pub unsafe extern "C" fn starts_one_ends(mut n: libc::c_int) -> libc::c_int {
    if n < 1 as libc::c_int {
        return 0 as libc::c_int;
    }
    if n == 1 as libc::c_int {
        return 1 as libc::c_int;
    }
    let mut out: libc::c_int = 18 as libc::c_int;
    let mut i: libc::c_int = 2 as libc::c_int;
    while i < n {
        out = out * 10 as libc::c_int;
        i += 1;
        i;
    }
    return out;
}
