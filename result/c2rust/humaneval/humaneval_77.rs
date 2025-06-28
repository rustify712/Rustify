use ::libc;
extern "C" {
    fn abs(_: libc::c_int) -> libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn iscuber(mut a: libc::c_int) -> libc::c_int {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i * i * i <= abs(a) {
        if i * i * i == abs(a) {
            return 1 as libc::c_int;
        }
        i += 1;
        i;
    }
    return 0 as libc::c_int;
}
