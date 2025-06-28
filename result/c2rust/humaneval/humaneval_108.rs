use ::libc;
extern "C" {
    fn abs(_: libc::c_int) -> libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn count_nums(
    mut n: *mut libc::c_int,
    mut size: libc::c_int,
) -> libc::c_int {
    let mut num: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if *n.offset(i as isize) > 0 as libc::c_int {
            num += 1 as libc::c_int;
        } else {
            let mut sum: libc::c_int = 0 as libc::c_int;
            let mut w: libc::c_int = 0;
            w = abs(*n.offset(i as isize));
            while w >= 10 as libc::c_int {
                sum += w % 10 as libc::c_int;
                w = w / 10 as libc::c_int;
            }
            sum -= w;
            if sum > 0 as libc::c_int {
                num += 1 as libc::c_int;
            }
        }
        i += 1;
        i;
    }
    return num;
}
