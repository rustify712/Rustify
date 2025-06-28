use ::libc;
extern "C" {
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn is_palindrome1(mut str: *const libc::c_char) -> bool {
    let mut len: libc::c_int = strlen(str) as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < len / 2 as libc::c_int {
        if *str.offset(i as isize) as libc::c_int
            != *str.offset((len - 1 as libc::c_int - i) as isize) as libc::c_int
        {
            return 0 as libc::c_int != 0;
        }
        i += 1;
        i;
    }
    return 1 as libc::c_int != 0;
}
#[no_mangle]
pub unsafe extern "C" fn make_palindrome(
    mut str: *const libc::c_char,
    mut result: *mut libc::c_char,
) {
    let mut len: libc::c_int = strlen(str) as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < len {
        if is_palindrome1(str.offset(i as isize)) {
            strcpy(result, str);
            let mut j: libc::c_int = i - 1 as libc::c_int;
            while j >= 0 as libc::c_int {
                *result
                    .offset(
                        (len + (i - 1 as libc::c_int - j)) as isize,
                    ) = *str.offset(j as isize);
                j -= 1;
                j;
            }
            *result.offset((len + i) as isize) = '\0' as i32 as libc::c_char;
            return;
        }
        i += 1;
        i;
    }
    strcpy(result, str);
    let mut i_0: libc::c_int = len - 1 as libc::c_int;
    while i_0 >= 0 as libc::c_int {
        *result
            .offset(
                (len + (len - 1 as libc::c_int - i_0)) as isize,
            ) = *str.offset(i_0 as isize);
        i_0 -= 1;
        i_0;
    }
    *result.offset((2 as libc::c_int * len) as isize) = '\0' as i32 as libc::c_char;
}
