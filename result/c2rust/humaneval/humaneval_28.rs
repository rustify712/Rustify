use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn strcat(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn concatenate(
    mut strings: *mut *mut libc::c_char,
    mut count: libc::c_int,
) -> *mut libc::c_char {
    let mut total_length: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < count {
        total_length = (total_length as libc::c_ulong)
            .wrapping_add(strlen(*strings.offset(i as isize))) as libc::c_int
            as libc::c_int;
        i += 1;
        i;
    }
    let mut out: *mut libc::c_char = malloc(
        (total_length + 1 as libc::c_int) as libc::c_ulong,
    ) as *mut libc::c_char;
    if out.is_null() {
        return 0 as *mut libc::c_char;
    }
    *out.offset(0 as libc::c_int as isize) = '\0' as i32 as libc::c_char;
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while i_0 < count {
        strcat(out, *strings.offset(i_0 as isize));
        i_0 += 1;
        i_0;
    }
    return out;
}
