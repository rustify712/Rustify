use ::libc;
#[no_mangle]
pub unsafe extern "C" fn add1(mut x: libc::c_int, mut y: libc::c_int) -> libc::c_int {
    return x + y;
}
