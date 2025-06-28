use ::libc;
#[no_mangle]
pub unsafe extern "C" fn largest_divisor(mut n: libc::c_int) -> libc::c_int {
    let mut i: libc::c_int = 2 as libc::c_int;
    while i * i <= n {
        if n % i == 0 as libc::c_int {
            return n / i;
        }
        i += 1;
        i;
    }
    return 1 as libc::c_int;
}
