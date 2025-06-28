use ::libc;
#[no_mangle]
pub unsafe extern "C" fn x_or_y(
    mut n: libc::c_int,
    mut x: libc::c_int,
    mut y: libc::c_int,
) -> libc::c_int {
    let mut isp: bool = 1 as libc::c_int != 0;
    if n < 2 as libc::c_int {
        isp = 0 as libc::c_int != 0;
    }
    let mut i: libc::c_int = 2 as libc::c_int;
    while i * i <= n {
        if n % i == 0 as libc::c_int {
            isp = 0 as libc::c_int != 0;
        }
        i += 1;
        i;
    }
    if isp {
        return x;
    }
    return y;
}
