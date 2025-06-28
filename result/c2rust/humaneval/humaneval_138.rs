use ::libc;
#[no_mangle]
pub unsafe extern "C" fn is_equal_to_sum_even(mut n: libc::c_int) -> bool {
    if n % 2 as libc::c_int == 0 as libc::c_int && n >= 8 as libc::c_int {
        return 1 as libc::c_int != 0;
    }
    return 0 as libc::c_int != 0;
}
