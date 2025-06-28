use ::libc;
extern "C" {
    fn floor(_: libc::c_double) -> libc::c_double;
}
#[no_mangle]
pub unsafe extern "C" fn truncate_number(mut number: libc::c_float) -> libc::c_float {
    return (number as libc::c_double - floor(number as libc::c_double)) as libc::c_float;
}
