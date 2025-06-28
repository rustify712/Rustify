use ::libc;
extern "C" {
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strcmp(_: *const libc::c_char, _: *const libc::c_char) -> libc::c_int;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn char_in_string(
    mut ch: libc::c_char,
    mut str: *const libc::c_char,
) -> libc::c_int {
    let mut i: libc::c_int = 0 as libc::c_int;
    while *str.offset(i as isize) as libc::c_int != '\0' as i32 {
        if *str.offset(i as isize) as libc::c_int == ch as libc::c_int {
            return 1 as libc::c_int;
        }
        i += 1;
        i;
    }
    return 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn reverse_string(mut str: *mut libc::c_char) {
    let mut len: libc::c_int = strlen(str) as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < len / 2 as libc::c_int {
        let mut temp: libc::c_char = *str.offset(i as isize);
        *str.offset(i as isize) = *str.offset((len - i - 1 as libc::c_int) as isize);
        *str.offset((len - i - 1 as libc::c_int) as isize) = temp;
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn reverse_delete(
    mut s: *const libc::c_char,
    mut c: *const libc::c_char,
) -> *mut *mut libc::c_char {
    let mut n: *mut libc::c_char = malloc(
        (strlen(s)).wrapping_add(1 as libc::c_int as libc::c_ulong),
    ) as *mut libc::c_char;
    let mut j: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while *s.offset(i as isize) as libc::c_int != '\0' as i32 {
        if char_in_string(*s.offset(i as isize), c) == 0 {
            let fresh0 = j;
            j = j + 1;
            *n.offset(fresh0 as isize) = *s.offset(i as isize);
        }
        i += 1;
        i;
    }
    *n.offset(j as isize) = '\0' as i32 as libc::c_char;
    let mut result: *mut *mut libc::c_char = malloc(
        (2 as libc::c_int as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong),
    ) as *mut *mut libc::c_char;
    let ref mut fresh1 = *result.offset(0 as libc::c_int as isize);
    *fresh1 = malloc((strlen(n)).wrapping_add(1 as libc::c_int as libc::c_ulong))
        as *mut libc::c_char;
    strcpy(*result.offset(0 as libc::c_int as isize), n);
    if strlen(n) == 0 as libc::c_int as libc::c_ulong {
        let ref mut fresh2 = *result.offset(1 as libc::c_int as isize);
        *fresh2 = malloc(6 as libc::c_int as libc::c_ulong) as *mut libc::c_char;
        strcpy(
            *result.offset(1 as libc::c_int as isize),
            b"True\0" as *const u8 as *const libc::c_char,
        );
        free(n as *mut libc::c_void);
        return result;
    }
    let mut w: *mut libc::c_char = malloc(
        (strlen(n)).wrapping_add(1 as libc::c_int as libc::c_ulong),
    ) as *mut libc::c_char;
    strcpy(w, n);
    reverse_string(w);
    if strcmp(w, n) == 0 as libc::c_int {
        let ref mut fresh3 = *result.offset(1 as libc::c_int as isize);
        *fresh3 = malloc(6 as libc::c_int as libc::c_ulong) as *mut libc::c_char;
        strcpy(
            *result.offset(1 as libc::c_int as isize),
            b"True\0" as *const u8 as *const libc::c_char,
        );
    } else {
        let ref mut fresh4 = *result.offset(1 as libc::c_int as isize);
        *fresh4 = malloc(6 as libc::c_int as libc::c_ulong) as *mut libc::c_char;
        strcpy(
            *result.offset(1 as libc::c_int as isize),
            b"False\0" as *const u8 as *const libc::c_char,
        );
    }
    free(n as *mut libc::c_void);
    free(w as *mut libc::c_void);
    return result;
}
