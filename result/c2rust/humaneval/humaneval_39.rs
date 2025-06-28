use ::libc;
#[no_mangle]
pub unsafe extern "C" fn prime_fib(mut n: libc::c_int) -> libc::c_int {
    let mut f1: libc::c_int = 0;
    let mut f2: libc::c_int = 0;
    let mut m: libc::c_int = 0;
    f1 = 1 as libc::c_int;
    f2 = 2 as libc::c_int;
    let mut count: libc::c_int = 0 as libc::c_int;
    while count < n {
        f1 = f1 + f2;
        m = f1;
        f1 = f2;
        f2 = m;
        let mut isprime: bool = 1 as libc::c_int != 0;
        let mut w: libc::c_int = 2 as libc::c_int;
        while w * w <= f1 {
            if f1 % w == 0 as libc::c_int {
                isprime = 0 as libc::c_int != 0;
                break;
            } else {
                w += 1;
                w;
            }
        }
        if isprime {
            count += 1 as libc::c_int;
        }
        if count == n {
            return f1;
        }
    }
    return -(1 as libc::c_int);
}
