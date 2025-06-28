use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn strlen1(mut str: *const libc::c_char) -> libc::c_int {
    return strlen(str) as libc::c_int;
}
