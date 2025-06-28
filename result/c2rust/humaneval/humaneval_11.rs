use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn string_xor(
    mut a: *const libc::c_char,
    mut b: *const libc::c_char,
) -> *mut libc::c_char {
    let mut len_a: libc::c_int = strlen(a) as libc::c_int;
    let mut len_b: libc::c_int = strlen(b) as libc::c_int;
    let mut max_len: libc::c_int = if len_a > len_b { len_a } else { len_b };
    let mut output: *mut libc::c_char = malloc(
        (max_len + 1 as libc::c_int) as libc::c_ulong,
    ) as *mut libc::c_char;
    if output.is_null() {
        return 0 as *mut libc::c_char;
    }
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < max_len {
        let mut char_a: libc::c_char = (if i < len_a {
            *a.offset(i as isize) as libc::c_int
        } else {
            '0' as i32
        }) as libc::c_char;
        let mut char_b: libc::c_char = (if i < len_b {
            *b.offset(i as isize) as libc::c_int
        } else {
            '0' as i32
        }) as libc::c_char;
        if char_a as libc::c_int == char_b as libc::c_int {
            *output.offset(i as isize) = '0' as i32 as libc::c_char;
        } else {
            *output.offset(i as isize) = '1' as i32 as libc::c_char;
        }
        i += 1;
        i;
    }
    *output.offset(max_len as isize) = '\0' as i32 as libc::c_char;
    return output;
}
