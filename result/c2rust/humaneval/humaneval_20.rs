use ::libc;
extern "C" {
    fn fabs(_: libc::c_double) -> libc::c_double;
}
#[no_mangle]
pub unsafe extern "C" fn find_closest_elements(
    mut numbers: *mut libc::c_float,
    mut size: libc::c_int,
    mut out: *mut libc::c_float,
) {
    let mut min_diff: libc::c_float = ::core::f32::INFINITY;
    let mut min_i: libc::c_int = 0 as libc::c_int;
    let mut min_j: libc::c_int = 1 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        let mut j: libc::c_int = i + 1 as libc::c_int;
        while j < size {
            let mut diff: libc::c_float = fabs(
                (*numbers.offset(i as isize) - *numbers.offset(j as isize))
                    as libc::c_double,
            ) as libc::c_float;
            if diff < min_diff {
                min_diff = diff;
                min_i = i;
                min_j = j;
            }
            j += 1;
            j;
        }
        i += 1;
        i;
    }
    if *numbers.offset(min_i as isize) > *numbers.offset(min_j as isize) {
        *out.offset(0 as libc::c_int as isize) = *numbers.offset(min_j as isize);
        *out.offset(1 as libc::c_int as isize) = *numbers.offset(min_i as isize);
    } else {
        *out.offset(0 as libc::c_int as isize) = *numbers.offset(min_i as isize);
        *out.offset(1 as libc::c_int as isize) = *numbers.offset(min_j as isize);
    };
}
