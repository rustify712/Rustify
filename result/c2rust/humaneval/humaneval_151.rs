use ::libc;
extern "C" {
    fn fabs(_: libc::c_double) -> libc::c_double;
    fn round(_: libc::c_double) -> libc::c_double;
}
#[no_mangle]
pub unsafe extern "C" fn double_the_difference(
    mut lst: *mut libc::c_float,
    mut size: libc::c_int,
) -> libc::c_longlong {
    let mut sum: libc::c_longlong = 0 as libc::c_int as libc::c_longlong;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if fabs(
            *lst.offset(i as isize) as libc::c_double
                - round(*lst.offset(i as isize) as libc::c_double),
        ) < 1e-4f64
        {
            if *lst.offset(i as isize) > 0 as libc::c_int as libc::c_float
                && round(*lst.offset(i as isize) as libc::c_double) as libc::c_int
                    % 2 as libc::c_int == 1 as libc::c_int
            {
                sum
                    += (round(*lst.offset(i as isize) as libc::c_double) as libc::c_int
                        * round(*lst.offset(i as isize) as libc::c_double)
                            as libc::c_int) as libc::c_longlong;
            }
        }
        i += 1;
        i;
    }
    return sum;
}
