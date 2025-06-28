use ::libc;
#[no_mangle]
pub unsafe extern "C" fn add_elements(
    mut arr: *mut libc::c_int,
    mut k: libc::c_int,
    mut len: libc::c_int,
) -> libc::c_int {
    let mut sum: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < k && i < len {
        if *arr.offset(i as isize) >= -(99 as libc::c_int)
            && *arr.offset(i as isize) <= 99 as libc::c_int
        {
            sum += *arr.offset(i as isize);
        }
        i += 1;
        i;
    }
    return sum;
}
