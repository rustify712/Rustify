use ::libc;
extern "C" {
    fn sprintf(_: *mut libc::c_char, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn digits(mut n: libc::c_int) -> libc::c_int {
    let mut prod: libc::c_int = 1 as libc::c_int;
    let mut has: libc::c_int = 0 as libc::c_int;
    let mut s: [libc::c_char; 20] = [0; 20];
    sprintf(s.as_mut_ptr(), b"%d\0" as *const u8 as *const libc::c_char, n);
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i as libc::c_ulong) < strlen(s.as_mut_ptr()) {
        if (s[i as usize] as libc::c_int - '0' as i32) % 2 as libc::c_int
            == 1 as libc::c_int
        {
            has = 1 as libc::c_int;
            prod = prod * (s[i as usize] as libc::c_int - '0' as i32);
        }
        i += 1;
        i;
    }
    if has == 0 as libc::c_int {
        return 0 as libc::c_int;
    }
    return prod;
}
