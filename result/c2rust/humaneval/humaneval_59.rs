use ::libc;
#[no_mangle]
pub unsafe extern "C" fn largest_prime_factor(mut n: libc::c_int) -> libc::c_int {
    let mut i: libc::c_int = 2 as libc::c_int;
    while i * i <= n {
        while n % i == 0 as libc::c_int && n > i {
            n = n / i;
        }
        i += 1;
        i;
    }
    return n;
}
