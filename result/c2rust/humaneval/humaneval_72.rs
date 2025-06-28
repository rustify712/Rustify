use ::libc;
#[no_mangle]
pub unsafe extern "C" fn will_it_fly(
    mut q: *mut libc::c_int,
    mut size: libc::c_int,
    mut w: libc::c_int,
) -> bool {
    let mut sum: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if *q.offset(i as isize) != *q.offset((size - 1 as libc::c_int - i) as isize) {
            return 0 as libc::c_int != 0;
        }
        sum += *q.offset(i as isize);
        i += 1;
        i;
    }
    if sum > w {
        return 0 as libc::c_int != 0;
    }
    return 1 as libc::c_int != 0;
}
