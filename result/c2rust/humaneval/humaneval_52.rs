use ::libc;
#[no_mangle]
pub unsafe extern "C" fn below_threshold(
    mut l: *mut libc::c_int,
    mut size: libc::c_int,
    mut t: libc::c_int,
) -> bool {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if *l.offset(i as isize) >= t {
            return 0 as libc::c_int != 0;
        }
        i += 1;
        i;
    }
    return 1 as libc::c_int != 0;
}
