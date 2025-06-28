use ::libc;
#[no_mangle]
pub unsafe extern "C" fn is_prime(mut n: libc::c_longlong) -> bool {
    if n < 2 as libc::c_int as libc::c_longlong {
        return 0 as libc::c_int != 0;
    }
    let mut i: libc::c_longlong = 2 as libc::c_int as libc::c_longlong;
    while i * i <= n {
        if n % i == 0 as libc::c_int as libc::c_longlong {
            return 0 as libc::c_int != 0;
        }
        i += 1;
        i;
    }
    return 1 as libc::c_int != 0;
}
