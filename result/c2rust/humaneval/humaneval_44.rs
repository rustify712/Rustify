use ::libc;
extern "C" {
    fn sprintf(_: *mut libc::c_char, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strcat(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
}
#[no_mangle]
pub unsafe extern "C" fn change_base(
    mut x: libc::c_int,
    mut base: libc::c_int,
) -> *mut libc::c_char {
    let mut out: *mut libc::c_char = malloc(
        (32 as libc::c_int as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    *out.offset(0 as libc::c_int as isize) = '\0' as i32 as libc::c_char;
    while x > 0 as libc::c_int {
        let mut temp: [libc::c_char; 2] = [0; 2];
        sprintf(
            temp.as_mut_ptr(),
            b"%d\0" as *const u8 as *const libc::c_char,
            x % base,
        );
        strcat(temp.as_mut_ptr(), out);
        strcpy(out, temp.as_mut_ptr());
        x = x / base;
    }
    return out;
}
