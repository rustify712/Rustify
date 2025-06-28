use ::libc;
#[no_mangle]
pub unsafe extern "C" fn modp(mut n: libc::c_int, mut p: libc::c_int) -> libc::c_int {
    let mut out: libc::c_int = 1 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < n {
        out = out * 2 as libc::c_int % p;
        i += 1;
        i;
    }
    return out;
}
