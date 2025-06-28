use ::libc;
#[no_mangle]
pub unsafe extern "C" fn is_multiply_prime(mut a: libc::c_int) -> bool {
    let mut num: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 2 as libc::c_int;
    while i * i <= a {
        while a % i == 0 as libc::c_int && a > i {
            a = a / i;
            num += 1 as libc::c_int;
        }
        i += 1;
        i;
    }
    if num == 2 as libc::c_int {
        return 1 as libc::c_int != 0;
    }
    return 0 as libc::c_int != 0;
}
