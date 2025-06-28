use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn realloc(_: *mut libc::c_void, _: libc::c_ulong) -> *mut libc::c_void;
    fn strncat(
        _: *mut libc::c_char,
        _: *const libc::c_char,
        _: libc::c_ulong,
    ) -> *mut libc::c_char;
    fn strcmp(_: *const libc::c_char, _: *const libc::c_char) -> libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn parse_music(
    mut music_string: *const libc::c_char,
    mut out_size: *mut libc::c_int,
) -> *mut libc::c_int {
    let mut capacity: libc::c_int = 10 as libc::c_int;
    let mut out: *mut libc::c_int = malloc(
        (capacity as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut current: [libc::c_char; 3] = [0 as libc::c_int as libc::c_char, 0, 0];
    let mut i: libc::c_int = 0 as libc::c_int;
    while *music_string.offset(i as isize) as libc::c_int != '\0' as i32 {
        if *music_string.offset(i as isize) as libc::c_int == ' ' as i32 {
            if strcmp(current.as_mut_ptr(), b"o\0" as *const u8 as *const libc::c_char)
                == 0 as libc::c_int
            {
                if count >= capacity {
                    capacity *= 2 as libc::c_int;
                    out = realloc(
                        out as *mut libc::c_void,
                        (capacity as libc::c_ulong)
                            .wrapping_mul(
                                ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
                            ),
                    ) as *mut libc::c_int;
                }
                let fresh0 = count;
                count = count + 1;
                *out.offset(fresh0 as isize) = 4 as libc::c_int;
            } else if strcmp(
                current.as_mut_ptr(),
                b"o|\0" as *const u8 as *const libc::c_char,
            ) == 0 as libc::c_int
            {
                if count >= capacity {
                    capacity *= 2 as libc::c_int;
                    out = realloc(
                        out as *mut libc::c_void,
                        (capacity as libc::c_ulong)
                            .wrapping_mul(
                                ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
                            ),
                    ) as *mut libc::c_int;
                }
                let fresh1 = count;
                count = count + 1;
                *out.offset(fresh1 as isize) = 2 as libc::c_int;
            } else if strcmp(
                current.as_mut_ptr(),
                b".|\0" as *const u8 as *const libc::c_char,
            ) == 0 as libc::c_int
            {
                if count >= capacity {
                    capacity *= 2 as libc::c_int;
                    out = realloc(
                        out as *mut libc::c_void,
                        (capacity as libc::c_ulong)
                            .wrapping_mul(
                                ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
                            ),
                    ) as *mut libc::c_int;
                }
                let fresh2 = count;
                count = count + 1;
                *out.offset(fresh2 as isize) = 1 as libc::c_int;
            }
            current[0 as libc::c_int as usize] = '\0' as i32 as libc::c_char;
        } else {
            strncat(
                current.as_mut_ptr(),
                &*music_string.offset(i as isize),
                1 as libc::c_int as libc::c_ulong,
            );
        }
        i += 1;
        i;
    }
    *out_size = count;
    return out;
}
