use ::libc;
#[no_mangle]
pub unsafe extern "C" fn special_factorial(mut n: libc::c_int) -> libc::c_longlong {
    let mut fact: libc::c_longlong = 1 as libc::c_int as libc::c_longlong;
    let mut bfact: libc::c_longlong = 1 as libc::c_int as libc::c_longlong;
    let mut i: libc::c_int = 1 as libc::c_int;
    while i <= n {
        fact = fact * i as libc::c_longlong;
        bfact = bfact * fact;
        i += 1;
        i;
    }
    return bfact;
}
