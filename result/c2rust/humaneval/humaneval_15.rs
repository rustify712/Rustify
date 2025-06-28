use ::libc;
extern "C" {
    fn snprintf(
        _: *mut libc::c_char,
        _: libc::c_ulong,
        _: *const libc::c_char,
        _: ...
    ) -> libc::c_int;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strcat(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
}
#[no_mangle]
pub unsafe extern "C" fn string_sequence(mut n: libc::c_int) -> *mut libc::c_char {
    let mut max_length: libc::c_int = snprintf(
        0 as *mut libc::c_char,
        0 as libc::c_int as libc::c_ulong,
        b"%d\0" as *const u8 as *const libc::c_char,
        n,
    ) * (n + 1 as libc::c_int) + n + 1 as libc::c_int;
    let mut out: *mut libc::c_char = malloc(
        (max_length as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    if out.is_null() {
        return 0 as *mut libc::c_char;
    }
    strcpy(out, b"0\0" as *const u8 as *const libc::c_char);
    let mut i: libc::c_int = 1 as libc::c_int;
    while i <= n {
        let mut buffer: [libc::c_char; 20] = [0; 20];
        snprintf(
            buffer.as_mut_ptr(),
            ::core::mem::size_of::<[libc::c_char; 20]>() as libc::c_ulong,
            b" %d\0" as *const u8 as *const libc::c_char,
            i,
        );
        strcat(out, buffer.as_mut_ptr());
        i += 1;
        i;
    }
    return out;
}
