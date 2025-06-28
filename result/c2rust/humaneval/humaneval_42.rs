use ::libc;
#[no_mangle]
pub unsafe extern "C" fn incr_list(
    mut l: *mut libc::c_int,
    mut size: libc::c_int,
) -> *mut libc::c_int {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        *l.offset(i as isize) += 1 as libc::c_int;
        i += 1;
        i;
    }
    return l;
}
