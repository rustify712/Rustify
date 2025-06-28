use ::libc;
#[no_mangle]
pub unsafe extern "C" fn is_simple_power(
    mut x: libc::c_int,
    mut n: libc::c_int,
) -> libc::c_int {
    let mut p: libc::c_int = 1 as libc::c_int;
    let mut count: libc::c_int = 0 as libc::c_int;
    while p <= x && count < 100 as libc::c_int {
        if p == x {
            return 1 as libc::c_int;
        }
        p = p * n;
        count += 1 as libc::c_int;
    }
    return 0 as libc::c_int;
}
