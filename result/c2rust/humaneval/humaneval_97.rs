use ::libc;
extern "C" {
    fn abs(_: libc::c_int) -> libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn multiply(
    mut a: libc::c_int,
    mut b: libc::c_int,
) -> libc::c_int {
    return abs(a) % 10 as libc::c_int * (abs(b) % 10 as libc::c_int);
}
