use ::libc;
#[no_mangle]
pub unsafe extern "C" fn can_arrange(
    mut arr: *mut libc::c_int,
    mut size: libc::c_int,
) -> libc::c_int {
    let mut max: libc::c_int = -(1 as libc::c_int);
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if *arr.offset(i as isize) <= i {
            max = i;
        }
        i += 1;
        i;
    }
    return max;
}
