use ::libc;
#[no_mangle]
pub unsafe extern "C" fn move_one_ball(
    mut arr: *mut libc::c_int,
    mut size: libc::c_int,
) -> bool {
    let mut num: libc::c_int = 0 as libc::c_int;
    if size == 0 as libc::c_int {
        return 1 as libc::c_int != 0;
    }
    let mut i: libc::c_int = 1 as libc::c_int;
    while i < size {
        if *arr.offset(i as isize) < *arr.offset((i - 1 as libc::c_int) as isize) {
            num += 1;
            num;
        }
        i += 1;
        i;
    }
    if *arr.offset((size - 1 as libc::c_int) as isize)
        > *arr.offset(0 as libc::c_int as isize)
    {
        num += 1;
        num;
    }
    if num < 2 as libc::c_int {
        return 1 as libc::c_int != 0;
    }
    return 0 as libc::c_int != 0;
}
