use ::libc;
extern "C" {
    fn sprintf(_: *mut libc::c_char, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strcat(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
}
#[no_mangle]
pub unsafe extern "C" fn rounded_avg(
    mut n: libc::c_int,
    mut m: libc::c_int,
) -> *mut libc::c_char {
    if n > m {
        let mut result: *mut libc::c_char = malloc(
            (3 as libc::c_int as libc::c_ulong)
                .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
        ) as *mut libc::c_char;
        strcpy(result, b"-1\0" as *const u8 as *const libc::c_char);
        return result;
    }
    let mut num: libc::c_int = (m + n) / 2 as libc::c_int;
    let mut out: *mut libc::c_char = malloc(
        (32 as libc::c_int as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    *out.offset(0 as libc::c_int as isize) = '\0' as i32 as libc::c_char;
    if num == 0 as libc::c_int {
        strcpy(out, b"0\0" as *const u8 as *const libc::c_char);
        return out;
    }
    while num > 0 as libc::c_int {
        let mut temp: [libc::c_char; 2] = [0; 2];
        sprintf(
            temp.as_mut_ptr(),
            b"%d\0" as *const u8 as *const libc::c_char,
            num % 2 as libc::c_int,
        );
        strcat(temp.as_mut_ptr(), out);
        strcpy(out, temp.as_mut_ptr());
        num = num / 2 as libc::c_int;
    }
    return out;
}
