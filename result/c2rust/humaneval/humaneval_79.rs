use ::libc;
extern "C" {
    fn sprintf(_: *mut libc::c_char, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strncat(
        _: *mut libc::c_char,
        _: *const libc::c_char,
        _: libc::c_ulong,
    ) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn decimal_to_binary(
    mut decimal: libc::c_int,
) -> *mut libc::c_char {
    let mut out: *mut libc::c_char = malloc(
        (32 as libc::c_int as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    *out.offset(0 as libc::c_int as isize) = '\0' as i32 as libc::c_char;
    if decimal == 0 as libc::c_int {
        strcpy(out, b"db0db\0" as *const u8 as *const libc::c_char);
        return out;
    }
    let mut binary: [libc::c_char; 32] = [0; 32];
    let mut index: libc::c_int = 0 as libc::c_int;
    while decimal > 0 as libc::c_int {
        let fresh0 = index;
        index = index + 1;
        binary[fresh0
            as usize] = (decimal % 2 as libc::c_int + '0' as i32) as libc::c_char;
        decimal = decimal / 2 as libc::c_int;
    }
    let mut i: libc::c_int = index - 1 as libc::c_int;
    while i >= 0 as libc::c_int {
        strncat(
            out,
            &mut *binary.as_mut_ptr().offset(i as isize),
            1 as libc::c_int as libc::c_ulong,
        );
        i -= 1;
        i;
    }
    let mut result: *mut libc::c_char = malloc(
        (strlen(out))
            .wrapping_add(5 as libc::c_int as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    sprintf(result, b"db%sdb\0" as *const u8 as *const libc::c_char, out);
    free(out as *mut libc::c_void);
    return result;
}
