use ::libc;
#[no_mangle]
pub unsafe extern "C" fn max_element(
    mut l: *mut libc::c_float,
    mut size: libc::c_int,
) -> libc::c_float {
    let mut max: libc::c_float = -(10000 as libc::c_int) as libc::c_float;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if max < *l.offset(i as isize) {
            max = *l.offset(i as isize);
        }
        i += 1;
        i;
    }
    return max;
}
