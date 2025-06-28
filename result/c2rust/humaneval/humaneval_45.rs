use ::libc;
#[no_mangle]
pub unsafe extern "C" fn triangle_area1(
    mut a: libc::c_float,
    mut h: libc::c_float,
) -> libc::c_float {
    return ((a * h) as libc::c_double * 0.5f64) as libc::c_float;
}
