use ::libc;
#[no_mangle]
pub unsafe extern "C" fn int_equal(
    mut vlocation1: *mut libc::c_void,
    mut vlocation2: *mut libc::c_void,
) -> libc::c_int {
    let mut location1: *mut libc::c_int = 0 as *mut libc::c_int;
    let mut location2: *mut libc::c_int = 0 as *mut libc::c_int;
    location1 = vlocation1 as *mut libc::c_int;
    location2 = vlocation2 as *mut libc::c_int;
    return (*location1 == *location2) as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn int_compare(
    mut vlocation1: *mut libc::c_void,
    mut vlocation2: *mut libc::c_void,
) -> libc::c_int {
    let mut location1: *mut libc::c_int = 0 as *mut libc::c_int;
    let mut location2: *mut libc::c_int = 0 as *mut libc::c_int;
    location1 = vlocation1 as *mut libc::c_int;
    location2 = vlocation2 as *mut libc::c_int;
    if *location1 < *location2 {
        return -(1 as libc::c_int)
    } else if *location1 > *location2 {
        return 1 as libc::c_int
    } else {
        return 0 as libc::c_int
    };
}
