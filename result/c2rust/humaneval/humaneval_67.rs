use ::libc;
extern "C" {
    fn strtol(
        _: *const libc::c_char,
        _: *mut *mut libc::c_char,
        _: libc::c_int,
    ) -> libc::c_long;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[inline]
unsafe extern "C" fn atoi(mut __nptr: *const libc::c_char) -> libc::c_int {
    return strtol(
        __nptr,
        0 as *mut libc::c_void as *mut *mut libc::c_char,
        10 as libc::c_int,
    ) as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn fruit_distribution(
    mut s: *const libc::c_char,
    mut n: libc::c_int,
) -> libc::c_int {
    let mut num1: [libc::c_char; 32] = [
        0 as libc::c_int as libc::c_char,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];
    let mut num2: [libc::c_char; 32] = [
        0 as libc::c_int as libc::c_char,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];
    let mut is12: libc::c_int = 0 as libc::c_int;
    let mut j: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i as libc::c_ulong) < strlen(s) {
        if *s.offset(i as isize) as libc::c_int >= '0' as i32
            && *s.offset(i as isize) as libc::c_int <= '9' as i32
        {
            if is12 == 0 as libc::c_int {
                let fresh0 = j;
                j = j + 1;
                num1[fresh0 as usize] = *s.offset(i as isize);
            } else {
                let fresh1 = j;
                j = j + 1;
                num2[fresh1 as usize] = *s.offset(i as isize);
            }
        } else if is12 == 0 as libc::c_int && j > 0 as libc::c_int {
            is12 = 1 as libc::c_int;
            j = 0 as libc::c_int;
        }
        i += 1;
        i;
    }
    return n - atoi(num1.as_mut_ptr()) - atoi(num2.as_mut_ptr());
}
