use ::libc;
extern "C" {
    fn fabs(_: libc::c_double) -> libc::c_double;
}
#[no_mangle]
pub unsafe extern "C" fn mean_absolute_deviation(
    mut numbers: *mut libc::c_float,
    mut size: libc::c_int,
) -> libc::c_float {
    let mut sum: libc::c_float = 0 as libc::c_int as libc::c_float;
    let mut avg: libc::c_float = 0.;
    let mut msum: libc::c_float = 0.;
    let mut mavg: libc::c_float = 0.;
    let mut i: libc::c_int = 0 as libc::c_int;
    i = 0 as libc::c_int;
    while i < size {
        sum += *numbers.offset(i as isize);
        i += 1;
        i;
    }
    avg = sum / size as libc::c_float;
    msum = 0 as libc::c_int as libc::c_float;
    i = 0 as libc::c_int;
    while i < size {
        msum = (msum as libc::c_double
            + fabs((*numbers.offset(i as isize) - avg) as libc::c_double))
            as libc::c_float;
        i += 1;
        i;
    }
    return msum / size as libc::c_float;
}
