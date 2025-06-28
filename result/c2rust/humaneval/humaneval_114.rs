use ::libc;
#[no_mangle]
pub unsafe extern "C" fn minSubArraySum(
    mut nums: *mut libc::c_longlong,
    mut size: libc::c_int,
) -> libc::c_longlong {
    let mut current: libc::c_longlong = 0;
    let mut min: libc::c_longlong = 0;
    current = *nums.offset(0 as libc::c_int as isize);
    min = *nums.offset(0 as libc::c_int as isize);
    let mut i: libc::c_int = 1 as libc::c_int;
    while i < size {
        if current < 0 as libc::c_int as libc::c_longlong {
            current = current + *nums.offset(i as isize);
        } else {
            current = *nums.offset(i as isize);
        }
        if current < min {
            min = current;
        }
        i += 1;
        i;
    }
    return min;
}
