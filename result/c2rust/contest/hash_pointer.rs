use ::libc;
#[no_mangle]
pub unsafe extern "C" fn pointer_hash(mut location: *mut libc::c_void) -> libc::c_uint {
    return location as libc::c_ulong as libc::c_uint;
}
