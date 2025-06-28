use ::libc;
#[no_mangle]
pub unsafe extern "C" fn int_hash(mut vlocation: *mut libc::c_void) -> libc::c_uint {
    let mut location: *mut libc::c_int = 0 as *mut libc::c_int;
    location = vlocation as *mut libc::c_int;
    return *location as libc::c_uint;
}
