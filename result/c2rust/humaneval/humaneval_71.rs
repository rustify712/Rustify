use ::libc;
extern "C" {
    fn sqrt(_: libc::c_double) -> libc::c_double;
}
#[no_mangle]
pub unsafe extern "C" fn triangle_area2(
    mut a: libc::c_float,
    mut b: libc::c_float,
    mut c: libc::c_float,
) -> libc::c_float {
    if a + b <= c || a + c <= b || b + c <= a {
        return -(1 as libc::c_int) as libc::c_float;
    }
    let mut h: libc::c_float = (a + b + c) / 2 as libc::c_int as libc::c_float;
    let mut area: libc::c_float = sqrt(
        (h * (h - a) * (h - b) * (h - c)) as libc::c_double,
    ) as libc::c_float;
    return area;
}
