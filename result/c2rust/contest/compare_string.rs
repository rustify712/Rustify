use ::libc;
extern "C" {
    fn tolower(_: libc::c_int) -> libc::c_int;
    fn strcmp(_: *const libc::c_char, _: *const libc::c_char) -> libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn string_equal(
    mut string1: *mut libc::c_void,
    mut string2: *mut libc::c_void,
) -> libc::c_int {
    return (strcmp(string1 as *mut libc::c_char, string2 as *mut libc::c_char)
        == 0 as libc::c_int) as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn string_compare(
    mut string1: *mut libc::c_void,
    mut string2: *mut libc::c_void,
) -> libc::c_int {
    let mut result: libc::c_int = 0;
    result = strcmp(string1 as *mut libc::c_char, string2 as *mut libc::c_char);
    if result < 0 as libc::c_int {
        return -(1 as libc::c_int)
    } else if result > 0 as libc::c_int {
        return 1 as libc::c_int
    } else {
        return 0 as libc::c_int
    };
}
#[no_mangle]
pub unsafe extern "C" fn string_nocase_equal(
    mut string1: *mut libc::c_void,
    mut string2: *mut libc::c_void,
) -> libc::c_int {
    return (string_nocase_compare(
        string1 as *mut libc::c_char as *mut libc::c_void,
        string2 as *mut libc::c_char as *mut libc::c_void,
    ) == 0 as libc::c_int) as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn string_nocase_compare(
    mut string1: *mut libc::c_void,
    mut string2: *mut libc::c_void,
) -> libc::c_int {
    let mut p1: *mut libc::c_char = 0 as *mut libc::c_char;
    let mut p2: *mut libc::c_char = 0 as *mut libc::c_char;
    let mut c1: libc::c_int = 0;
    let mut c2: libc::c_int = 0;
    p1 = string1 as *mut libc::c_char;
    p2 = string2 as *mut libc::c_char;
    loop {
        c1 = tolower(*p1 as libc::c_int);
        c2 = tolower(*p2 as libc::c_int);
        if c1 != c2 {
            if c1 < c2 { return -(1 as libc::c_int) } else { return 1 as libc::c_int }
        }
        if c1 == '\0' as i32 {
            break;
        }
        p1 = p1.offset(1);
        p1;
        p2 = p2.offset(1);
        p2;
    }
    return 0 as libc::c_int;
}
