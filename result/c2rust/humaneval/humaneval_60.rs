use ::libc;
#[no_mangle]
pub unsafe extern "C" fn sum_to_n(mut n: libc::c_int) -> libc::c_int {
    return n * (n + 1 as libc::c_int) / 2 as libc::c_int;
}
