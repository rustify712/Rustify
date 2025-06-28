use ::libc;
extern "C" {
    fn round(_: libc::c_double) -> libc::c_double;
}
#[no_mangle]
pub unsafe extern "C" fn any_int(
    mut a: libc::c_float,
    mut b: libc::c_float,
    mut c: libc::c_float,
) -> libc::c_int {
    if round(a as libc::c_double) != a as libc::c_double {
        return 0 as libc::c_int;
    }
    if round(b as libc::c_double) != b as libc::c_double {
        return 0 as libc::c_int;
    }
    if round(c as libc::c_double) != c as libc::c_double {
        return 0 as libc::c_int;
    }
    if a + b == c || a + c == b || b + c == a {
        return 1 as libc::c_int;
    }
    return 0 as libc::c_int;
}
