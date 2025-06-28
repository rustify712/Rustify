use ::libc;
#[no_mangle]
pub unsafe extern "C" fn is_sorted(
    mut lst: *mut libc::c_int,
    mut size: libc::c_int,
) -> bool {
    let mut i: libc::c_int = 1 as libc::c_int;
    while i < size {
        if *lst.offset(i as isize) < *lst.offset((i - 1 as libc::c_int) as isize) {
            return 0 as libc::c_int != 0;
        }
        if i >= 2 as libc::c_int
            && *lst.offset(i as isize) == *lst.offset((i - 1 as libc::c_int) as isize)
            && *lst.offset(i as isize) == *lst.offset((i - 2 as libc::c_int) as isize)
        {
            return 0 as libc::c_int != 0;
        }
        i += 1;
        i;
    }
    return 1 as libc::c_int != 0;
}
