use ::libc;
extern "C" {
    fn sprintf(_: *mut libc::c_char, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn strncpy(
        _: *mut libc::c_char,
        _: *const libc::c_char,
        _: libc::c_ulong,
    ) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn circular_shift(
    mut x: libc::c_int,
    mut shift: libc::c_int,
) -> *mut libc::c_char {
    let mut xs: [libc::c_char; 20] = [0; 20];
    sprintf(xs.as_mut_ptr(), b"%d\0" as *const u8 as *const libc::c_char, x);
    let mut len: libc::c_int = strlen(xs.as_mut_ptr()) as libc::c_int;
    let mut result: *mut libc::c_char = malloc(
        ((len + 1 as libc::c_int) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    if result.is_null() {
        return 0 as *mut libc::c_char;
    }
    if len < shift {
        let mut i: libc::c_int = 0 as libc::c_int;
        while i < len {
            *result.offset(i as isize) = xs[(len - 1 as libc::c_int - i) as usize];
            i += 1;
            i;
        }
        *result.offset(len as isize) = '\0' as i32 as libc::c_char;
    } else {
        strncpy(
            result,
            xs.as_mut_ptr().offset(len as isize).offset(-(shift as isize)),
            shift as libc::c_ulong,
        );
        strncpy(
            result.offset(shift as isize),
            xs.as_mut_ptr(),
            (len - shift) as libc::c_ulong,
        );
        *result.offset(len as isize) = '\0' as i32 as libc::c_char;
    }
    return result;
}
