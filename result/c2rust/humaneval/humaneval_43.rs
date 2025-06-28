use ::libc;
#[no_mangle]
pub unsafe extern "C" fn pairs_sum_to_zero(
    mut l: *mut libc::c_int,
    mut size: libc::c_int,
) -> bool {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        let mut j: libc::c_int = i + 1 as libc::c_int;
        while j < size {
            if *l.offset(i as isize) + *l.offset(j as isize) == 0 as libc::c_int {
                return 1 as libc::c_int != 0;
            }
            j += 1;
            j;
        }
        i += 1;
        i;
    }
    return 0 as libc::c_int != 0;
}
