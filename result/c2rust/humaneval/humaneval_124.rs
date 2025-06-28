use ::libc;
extern "C" {
    fn strtol(
        _: *const libc::c_char,
        _: *mut *mut libc::c_char,
        _: libc::c_int,
    ) -> libc::c_long;
    fn strncpy(
        _: *mut libc::c_char,
        _: *const libc::c_char,
        _: libc::c_ulong,
    ) -> *mut libc::c_char;
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
pub unsafe extern "C" fn valid_date(mut date: *const libc::c_char) -> bool {
    let mut mm: libc::c_int = 0;
    let mut dd: libc::c_int = 0;
    let mut yy: libc::c_int = 0;
    if strlen(date) != 10 as libc::c_int as libc::c_ulong {
        return 0 as libc::c_int != 0;
    }
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < 10 as libc::c_int {
        if i == 2 as libc::c_int || i == 5 as libc::c_int {
            if *date.offset(i as isize) as libc::c_int != '-' as i32 {
                return 0 as libc::c_int != 0;
            }
        } else if (*date.offset(i as isize) as libc::c_int) < '0' as i32
            || *date.offset(i as isize) as libc::c_int > '9' as i32
        {
            return 0 as libc::c_int != 0
        }
        i += 1;
        i;
    }
    let mut mm_str: [libc::c_char; 3] = [0; 3];
    let mut dd_str: [libc::c_char; 3] = [0; 3];
    let mut yy_str: [libc::c_char; 5] = [0; 5];
    strncpy(mm_str.as_mut_ptr(), date, 2 as libc::c_int as libc::c_ulong);
    mm_str[2 as libc::c_int as usize] = '\0' as i32 as libc::c_char;
    strncpy(
        dd_str.as_mut_ptr(),
        date.offset(3 as libc::c_int as isize),
        2 as libc::c_int as libc::c_ulong,
    );
    dd_str[2 as libc::c_int as usize] = '\0' as i32 as libc::c_char;
    strncpy(
        yy_str.as_mut_ptr(),
        date.offset(6 as libc::c_int as isize),
        4 as libc::c_int as libc::c_ulong,
    );
    yy_str[4 as libc::c_int as usize] = '\0' as i32 as libc::c_char;
    mm = atoi(mm_str.as_mut_ptr());
    dd = atoi(dd_str.as_mut_ptr());
    yy = atoi(yy_str.as_mut_ptr());
    if mm < 1 as libc::c_int || mm > 12 as libc::c_int {
        return 0 as libc::c_int != 0;
    }
    if dd < 1 as libc::c_int || dd > 31 as libc::c_int {
        return 0 as libc::c_int != 0;
    }
    if dd == 31 as libc::c_int
        && (mm == 4 as libc::c_int || mm == 6 as libc::c_int || mm == 9 as libc::c_int
            || mm == 11 as libc::c_int || mm == 2 as libc::c_int)
    {
        return 0 as libc::c_int != 0;
    }
    if dd == 30 as libc::c_int && mm == 2 as libc::c_int {
        return 0 as libc::c_int != 0;
    }
    return 1 as libc::c_int != 0;
}
