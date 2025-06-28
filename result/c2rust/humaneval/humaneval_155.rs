use ::libc;
extern "C" {
    fn snprintf(
        _: *mut libc::c_char,
        _: libc::c_ulong,
        _: *const libc::c_char,
        _: ...
    ) -> libc::c_int;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn abs(_: libc::c_int) -> libc::c_int;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn even_odd_count(mut num: libc::c_int) -> *mut libc::c_int {
    let mut result: *mut libc::c_int = malloc(
        (2 as libc::c_int as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    if result.is_null() {
        return 0 as *mut libc::c_int;
    }
    let mut buffer: [libc::c_char; 20] = [0; 20];
    snprintf(
        buffer.as_mut_ptr(),
        ::core::mem::size_of::<[libc::c_char; 20]>() as libc::c_ulong,
        b"%d\0" as *const u8 as *const libc::c_char,
        abs(num),
    );
    let mut n1: libc::c_int = 0 as libc::c_int;
    let mut n2: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i as libc::c_ulong) < strlen(buffer.as_mut_ptr()) {
        if (buffer[i as usize] as libc::c_int - '0' as i32) % 2 as libc::c_int
            == 1 as libc::c_int
        {
            n1 += 1;
            n1;
        } else {
            n2 += 1;
            n2;
        }
        i += 1;
        i;
    }
    *result.offset(0 as libc::c_int as isize) = n2;
    *result.offset(1 as libc::c_int as isize) = n1;
    return result;
}
