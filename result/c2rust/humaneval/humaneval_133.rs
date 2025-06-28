use ::libc;
extern "C" {
    fn ceil(_: libc::c_double) -> libc::c_double;
}
#[no_mangle]
pub unsafe extern "C" fn sum_squares(
    mut lst: *mut libc::c_float,
    mut size: libc::c_int,
) -> libc::c_int {
    let mut sum: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        sum
            += ceil(*lst.offset(i as isize) as libc::c_double) as libc::c_int
                * ceil(*lst.offset(i as isize) as libc::c_double) as libc::c_int;
        i += 1;
        i;
    }
    return sum;
}
