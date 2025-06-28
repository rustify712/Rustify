use ::libc;
extern "C" {
    fn fabs(_: libc::c_double) -> libc::c_double;
}
#[no_mangle]
pub unsafe extern "C" fn has_close_elements(
    mut numbers: *mut libc::c_float,
    mut size: libc::c_int,
    mut threshold: libc::c_float,
) -> bool {
    let mut i: libc::c_int = 0;
    let mut j: libc::c_int = 0;
    i = 0 as libc::c_int;
    while i < size {
        j = i + 1 as libc::c_int;
        while j < size {
            if fabs(
                (*numbers.offset(i as isize) - *numbers.offset(j as isize))
                    as libc::c_double,
            ) < threshold as libc::c_double
            {
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
