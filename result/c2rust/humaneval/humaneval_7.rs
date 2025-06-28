use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn strstr(_: *const libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
}
#[no_mangle]
pub unsafe extern "C" fn filter_by_substring(
    mut strings: *mut *mut libc::c_char,
    mut num_strings: libc::c_int,
    mut substring: *const libc::c_char,
    mut out_num_strings: *mut libc::c_int,
) -> *mut *mut libc::c_char {
    let mut out: *mut *mut libc::c_char = malloc(
        (num_strings as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong),
    ) as *mut *mut libc::c_char;
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < num_strings {
        if !(strstr(*strings.offset(i as isize), substring)).is_null() {
            let ref mut fresh0 = *out.offset(count as isize);
            *fresh0 = *strings.offset(i as isize);
            count += 1;
            count;
        }
        i += 1;
        i;
    }
    *out_num_strings = count;
    return out;
}
