use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn realloc(_: *mut libc::c_void, _: libc::c_ulong) -> *mut libc::c_void;
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn is_consonant(mut c: libc::c_char) -> libc::c_int {
    let mut vowels: [libc::c_char; 11] = *::core::mem::transmute::<
        &[u8; 11],
        &mut [libc::c_char; 11],
    >(b"aeiouAEIOU\0");
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i as libc::c_ulong) < strlen(vowels.as_mut_ptr()) {
        if c as libc::c_int == vowels[i as usize] as libc::c_int {
            return 0 as libc::c_int;
        }
        i += 1;
        i;
    }
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn select_words(
    mut s: *const libc::c_char,
    mut n: libc::c_int,
    mut result_size: *mut libc::c_int,
) -> *mut *mut libc::c_char {
    let mut capacity: libc::c_int = 10 as libc::c_int;
    let mut out: *mut *mut libc::c_char = malloc(
        (capacity as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong),
    ) as *mut *mut libc::c_char;
    *result_size = 0 as libc::c_int;
    let mut current: [libc::c_char; 100] = [0; 100];
    let mut current_len: libc::c_int = 0 as libc::c_int;
    let mut numc: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i as libc::c_ulong <= strlen(s) {
        if *s.offset(i as isize) as libc::c_int == ' ' as i32
            || *s.offset(i as isize) as libc::c_int == '\0' as i32
        {
            if numc == n {
                current[current_len as usize] = '\0' as i32 as libc::c_char;
                if *result_size >= capacity {
                    capacity *= 2 as libc::c_int;
                    out = realloc(
                        out as *mut libc::c_void,
                        (capacity as libc::c_ulong)
                            .wrapping_mul(
                                ::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong,
                            ),
                    ) as *mut *mut libc::c_char;
                }
                let ref mut fresh0 = *out.offset(*result_size as isize);
                *fresh0 = malloc(
                    ((current_len + 1 as libc::c_int) as libc::c_ulong)
                        .wrapping_mul(
                            ::core::mem::size_of::<libc::c_char>() as libc::c_ulong,
                        ),
                ) as *mut libc::c_char;
                strcpy(*out.offset(*result_size as isize), current.as_mut_ptr());
                *result_size += 1;
                *result_size;
            }
            current_len = 0 as libc::c_int;
            numc = 0 as libc::c_int;
        } else {
            let fresh1 = current_len;
            current_len = current_len + 1;
            current[fresh1 as usize] = *s.offset(i as isize);
            if *s.offset(i as isize) as libc::c_int >= 'A' as i32
                && *s.offset(i as isize) as libc::c_int <= 'Z' as i32
                || *s.offset(i as isize) as libc::c_int >= 'a' as i32
                    && *s.offset(i as isize) as libc::c_int <= 'z' as i32
            {
                if is_consonant(*s.offset(i as isize)) != 0 {
                    numc += 1;
                    numc;
                }
            }
        }
        i += 1;
        i;
    }
    return out;
}
