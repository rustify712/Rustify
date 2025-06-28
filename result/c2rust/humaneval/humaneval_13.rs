use ::libc;
#[no_mangle]
pub unsafe extern "C" fn greatest_common_divisor(
    mut a: libc::c_int,
    mut b: libc::c_int,
) -> libc::c_int {
    let mut out: libc::c_int = 0;
    let mut m: libc::c_int = 0;
    loop {
        if a < b {
            m = a;
            a = b;
            b = m;
        }
        a = a % b;
        if a == 0 as libc::c_int {
            return b;
        }
    };
}
