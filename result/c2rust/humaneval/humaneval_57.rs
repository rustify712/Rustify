use ::libc;
#[no_mangle]
pub unsafe extern "C" fn monotonic(
    mut l: *mut libc::c_float,
    mut size: libc::c_int,
) -> bool {
    let mut incr: libc::c_int = 0 as libc::c_int;
    let mut decr: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 1 as libc::c_int;
    while i < size {
        if *l.offset(i as isize) > *l.offset((i - 1 as libc::c_int) as isize) {
            incr = 1 as libc::c_int;
        }
        if *l.offset(i as isize) < *l.offset((i - 1 as libc::c_int) as isize) {
            decr = 1 as libc::c_int;
        }
        i += 1;
        i;
    }
    if incr + decr == 2 as libc::c_int {
        return 0 as libc::c_int != 0;
    }
    return 1 as libc::c_int != 0;
}
