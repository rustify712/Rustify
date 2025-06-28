use ::libc;
#[no_mangle]
pub unsafe extern "C" fn sum_squares1(
    mut lst: *mut libc::c_int,
    mut size: libc::c_int,
) -> libc::c_int {
    let mut sum: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if i % 3 as libc::c_int == 0 as libc::c_int {
            sum += *lst.offset(i as isize) * *lst.offset(i as isize);
        } else if i % 4 as libc::c_int == 0 as libc::c_int {
            sum
                += *lst.offset(i as isize) * *lst.offset(i as isize)
                    * *lst.offset(i as isize);
        } else {
            sum += *lst.offset(i as isize);
        }
        i += 1;
        i;
    }
    return sum;
}
