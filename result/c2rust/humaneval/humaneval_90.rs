use ::libc;
#[no_mangle]
pub unsafe extern "C" fn next_smallest(
    mut lst: *mut libc::c_int,
    mut size: libc::c_int,
) -> libc::c_int {
    if size <= 1 as libc::c_int {
        return -(1 as libc::c_int);
    }
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size - 1 as libc::c_int {
        let mut j: libc::c_int = i + 1 as libc::c_int;
        while j < size {
            if *lst.offset(i as isize) > *lst.offset(j as isize) {
                let mut temp: libc::c_int = *lst.offset(i as isize);
                *lst.offset(i as isize) = *lst.offset(j as isize);
                *lst.offset(j as isize) = temp;
            }
            j += 1;
            j;
        }
        i += 1;
        i;
    }
    let mut i_0: libc::c_int = 1 as libc::c_int;
    while i_0 < size {
        if *lst.offset(i_0 as isize) != *lst.offset((i_0 - 1 as libc::c_int) as isize) {
            return *lst.offset(i_0 as isize);
        }
        i_0 += 1;
        i_0;
    }
    return -(1 as libc::c_int);
}
