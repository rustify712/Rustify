use ::libc;
extern "C" {
    fn pow(_: libc::c_double, _: libc::c_double) -> libc::c_double;
    fn fabs(_: libc::c_double) -> libc::c_double;
}
#[no_mangle]
pub unsafe extern "C" fn poly(
    mut xs: *mut libc::c_double,
    mut size: libc::c_int,
    mut x: libc::c_double,
) -> libc::c_double {
    let mut sum: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        sum += *xs.offset(i as isize) * pow(x, i as libc::c_double);
        i += 1;
        i;
    }
    return sum;
}
#[no_mangle]
pub unsafe extern "C" fn find_zero(
    mut xs: *mut libc::c_double,
    mut size: libc::c_int,
) -> libc::c_double {
    let mut ans: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut value: libc::c_double = poly(xs, size, ans);
    while fabs(value) > 1e-6f64 {
        let mut driv: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut i: libc::c_int = 1 as libc::c_int;
        while i < size {
            driv
                += *xs.offset(i as isize)
                    * pow(ans, (i - 1 as libc::c_int) as libc::c_double)
                    * i as libc::c_double;
            i += 1;
            i;
        }
        ans = ans - value / driv;
        value = poly(xs, size, ans);
    }
    return ans;
}
