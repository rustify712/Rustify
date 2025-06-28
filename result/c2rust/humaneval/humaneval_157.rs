use ::libc;
extern "C" {
    fn fabs(_: libc::c_double) -> libc::c_double;
}
#[no_mangle]
pub unsafe extern "C" fn right_angle_triangle(
    mut a: libc::c_float,
    mut b: libc::c_float,
    mut c: libc::c_float,
) -> libc::c_int {
    if fabs((a * a + b * b - c * c) as libc::c_double) < 1e-4f64
        || fabs((a * a + c * c - b * b) as libc::c_double) < 1e-4f64
        || fabs((b * b + c * c - a * a) as libc::c_double) < 1e-4f64
    {
        return 1 as libc::c_int;
    }
    return 0 as libc::c_int;
}
