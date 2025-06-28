use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn all_prefixes(
    mut str: *const libc::c_char,
    mut out_size: *mut libc::c_int,
) -> *mut *mut libc::c_char {
    let mut len: libc::c_int = strlen(str) as libc::c_int;
    let mut out: *mut *mut libc::c_char = malloc(
        (len as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong),
    ) as *mut *mut libc::c_char;
    let mut current: *mut libc::c_char = malloc(
        ((len + 1 as libc::c_int) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    *current.offset(0 as libc::c_int as isize) = '\0' as i32 as libc::c_char;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < len {
        *current.offset(i as isize) = *str.offset(i as isize);
        *current.offset((i + 1 as libc::c_int) as isize) = '\0' as i32 as libc::c_char;
        let ref mut fresh0 = *out.offset(i as isize);
        *fresh0 = malloc(
            ((i + 2 as libc::c_int) as libc::c_ulong)
                .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
        ) as *mut libc::c_char;
        strcpy(*out.offset(i as isize), current);
        i += 1;
        i;
    }
    *out_size = len;
    free(current as *mut libc::c_void);
    return out;
}
