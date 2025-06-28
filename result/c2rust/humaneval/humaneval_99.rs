use ::libc;
extern "C" {
    fn strtod(_: *const libc::c_char, _: *mut *mut libc::c_char) -> libc::c_double;
    fn round(_: libc::c_double) -> libc::c_double;
}
#[inline]
unsafe extern "C" fn atof(mut __nptr: *const libc::c_char) -> libc::c_double {
    return strtod(__nptr, 0 as *mut libc::c_void as *mut *mut libc::c_char);
}
#[no_mangle]
pub unsafe extern "C" fn closest_integer(mut value: *const libc::c_char) -> libc::c_int {
    let mut w: libc::c_double = atof(value);
    return round(w) as libc::c_int;
}
