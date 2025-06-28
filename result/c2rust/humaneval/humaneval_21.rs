use ::libc;
#[no_mangle]
pub unsafe extern "C" fn rescale_to_unit(
    mut numbers: *mut libc::c_float,
    mut size: libc::c_int,
) -> *mut libc::c_float {
    let mut min: libc::c_float = 100000 as libc::c_int as libc::c_float;
    let mut max: libc::c_float = -(100000 as libc::c_int) as libc::c_float;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if *numbers.offset(i as isize) < min {
            min = *numbers.offset(i as isize);
        }
        if *numbers.offset(i as isize) > max {
            max = *numbers.offset(i as isize);
        }
        i += 1;
        i;
    }
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while i_0 < size {
        *numbers
            .offset(i_0 as isize) = (*numbers.offset(i_0 as isize) - min) / (max - min);
        i_0 += 1;
        i_0;
    }
    return numbers;
}
