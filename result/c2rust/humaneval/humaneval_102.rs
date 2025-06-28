use ::libc;
#[no_mangle]
pub unsafe extern "C" fn choose_num(
    mut x: libc::c_int,
    mut y: libc::c_int,
) -> libc::c_int {
    if y < x {
        return -(1 as libc::c_int);
    }
    if y == x && y % 2 as libc::c_int == 1 as libc::c_int {
        return -(1 as libc::c_int);
    }
    if y % 2 as libc::c_int == 1 as libc::c_int {
        return y - 1 as libc::c_int;
    }
    return y;
}
