use ::libc;
#[no_mangle]
pub unsafe extern "C" fn sum_product(
    mut numbers: *mut libc::c_int,
    mut size: libc::c_int,
    mut sum: *mut libc::c_int,
    mut product: *mut libc::c_int,
) {
    *sum = 0 as libc::c_int;
    *product = 1 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        *sum += *numbers.offset(i as isize);
        *product *= *numbers.offset(i as isize);
        i += 1;
        i;
    }
}
