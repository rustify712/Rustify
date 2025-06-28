use ::libc;
extern "C" {
    fn snprintf(
        _: *mut libc::c_char,
        _: libc::c_ulong,
        _: *const libc::c_char,
        _: ...
    ) -> libc::c_int;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn specialFilter(
    mut nums: *mut libc::c_int,
    mut size: libc::c_int,
) -> libc::c_int {
    let mut num: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if *nums.offset(i as isize) > 10 as libc::c_int {
            let mut buffer: [libc::c_char; 20] = [0; 20];
            snprintf(
                buffer.as_mut_ptr(),
                ::core::mem::size_of::<[libc::c_char; 20]>() as libc::c_ulong,
                b"%d\0" as *const u8 as *const libc::c_char,
                *nums.offset(i as isize),
            );
            let mut len: libc::c_int = strlen(buffer.as_mut_ptr()) as libc::c_int;
            if (buffer[0 as libc::c_int as usize] as libc::c_int - '0' as i32)
                % 2 as libc::c_int == 1 as libc::c_int
                && (buffer[(len - 1 as libc::c_int) as usize] as libc::c_int
                    - '0' as i32) % 2 as libc::c_int == 1 as libc::c_int
            {
                num += 1;
                num;
            }
        }
        i += 1;
        i;
    }
    return num;
}
