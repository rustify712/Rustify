use ::libc;
extern "C" {
    fn sprintf(_: *mut libc::c_char, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strcat(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn solve2(mut N: libc::c_int) -> *mut libc::c_char {
    let mut str: [libc::c_char; 6] = [0; 6];
    sprintf(str.as_mut_ptr(), b"%d\0" as *const u8 as *const libc::c_char, N);
    let mut sum: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i as libc::c_ulong) < strlen(str.as_mut_ptr()) {
        sum += str[i as usize] as libc::c_int - '0' as i32;
        i += 1;
        i;
    }
    let mut bi: *mut libc::c_char = malloc(
        (20 as libc::c_int as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    *bi.offset(0 as libc::c_int as isize) = '\0' as i32 as libc::c_char;
    if sum == 0 as libc::c_int {
        strcat(bi, b"0\0" as *const u8 as *const libc::c_char);
    } else {
        while sum > 0 as libc::c_int {
            let mut temp: [libc::c_char; 2] = [0; 2];
            sprintf(
                temp.as_mut_ptr(),
                b"%d\0" as *const u8 as *const libc::c_char,
                sum % 2 as libc::c_int,
            );
            strcat(temp.as_mut_ptr(), bi);
            strcpy(bi, temp.as_mut_ptr());
            sum /= 2 as libc::c_int;
        }
    }
    return bi;
}
