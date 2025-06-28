use ::libc;
#[no_mangle]
pub unsafe extern "C" fn pointer_equal(
    mut location1: *mut libc::c_void,
    mut location2: *mut libc::c_void,
) -> libc::c_int {
    return (location1 == location2) as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn pointer_compare(
    mut location1: *mut libc::c_void,
    mut location2: *mut libc::c_void,
) -> libc::c_int {
    if location1 < location2 {
        return -(1 as libc::c_int)
    } else if location1 > location2 {
        return 1 as libc::c_int
    } else {
        return 0 as libc::c_int
    };
}
