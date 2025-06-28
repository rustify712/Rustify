use ::libc;
#[no_mangle]
pub unsafe extern "C" fn solutions(
    mut lst: *mut libc::c_int,
    mut size: libc::c_int,
) -> libc::c_int {
    let mut sum: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i * 2 as libc::c_int) < size {
        if *lst.offset((i * 2 as libc::c_int) as isize) % 2 as libc::c_int
            == 1 as libc::c_int
        {
            sum += *lst.offset((i * 2 as libc::c_int) as isize);
        }
        i += 1;
        i;
    }
    return sum;
}
