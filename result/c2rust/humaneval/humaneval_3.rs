use ::libc;
#[no_mangle]
pub unsafe extern "C" fn below_zero(
    mut operations: *mut libc::c_int,
    mut size: libc::c_int,
) -> bool {
    let mut num: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        num += *operations.offset(i as isize);
        if num < 0 as libc::c_int {
            return 1 as libc::c_int != 0;
        }
        i += 1;
        i;
    }
    return 0 as libc::c_int != 0;
}
