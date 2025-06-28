use ::libc;
#[no_mangle]
pub unsafe extern "C" fn smallest_change(
    mut arr: *mut libc::c_int,
    mut size: libc::c_int,
) -> libc::c_int {
    let mut out: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size - 1 as libc::c_int - i {
        if *arr.offset(i as isize) != *arr.offset((size - 1 as libc::c_int - i) as isize)
        {
            out += 1 as libc::c_int;
        }
        i += 1;
        i;
    }
    return out;
}
