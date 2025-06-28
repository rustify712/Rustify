use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn correct_bracketing2(mut brackets: *const libc::c_char) -> bool {
    let mut level: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i as libc::c_ulong) < strlen(brackets) {
        if *brackets.offset(i as isize) as libc::c_int == '(' as i32 {
            level += 1 as libc::c_int;
        }
        if *brackets.offset(i as isize) as libc::c_int == ')' as i32 {
            level -= 1 as libc::c_int;
        }
        if level < 0 as libc::c_int {
            return 0 as libc::c_int != 0;
        }
        i += 1;
        i;
    }
    if level != 0 as libc::c_int {
        return 0 as libc::c_int != 0;
    }
    return 1 as libc::c_int != 0;
}
