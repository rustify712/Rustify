use ::libc;
#[no_mangle]
pub unsafe extern "C" fn largest_smallest_integers(
    mut lst: *mut libc::c_int,
    mut size: libc::c_int,
    mut result: *mut libc::c_int,
) {
    let mut maxneg: libc::c_int = 0 as libc::c_int;
    let mut minpos: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if *lst.offset(i as isize) < 0 as libc::c_int
            && (maxneg == 0 as libc::c_int || *lst.offset(i as isize) > maxneg)
        {
            maxneg = *lst.offset(i as isize);
        }
        if *lst.offset(i as isize) > 0 as libc::c_int
            && (minpos == 0 as libc::c_int || *lst.offset(i as isize) < minpos)
        {
            minpos = *lst.offset(i as isize);
        }
        i += 1;
        i;
    }
    *result.offset(0 as libc::c_int as isize) = maxneg;
    *result.offset(1 as libc::c_int as isize) = minpos;
}
