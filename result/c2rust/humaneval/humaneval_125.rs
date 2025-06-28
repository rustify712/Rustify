use ::libc;
extern "C" {
    fn sprintf(_: *mut libc::c_char, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn realloc(_: *mut libc::c_void, _: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strcat(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strncat(
        _: *mut libc::c_char,
        _: *const libc::c_char,
        _: libc::c_ulong,
    ) -> *mut libc::c_char;
    fn strchr(_: *const libc::c_char, _: libc::c_int) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn split_words(
    mut txt: *const libc::c_char,
    mut out_size: *mut libc::c_int,
) -> *mut *mut libc::c_char {
    let mut i: libc::c_int = 0;
    let mut current: *mut libc::c_char = malloc(
        (strlen(txt)).wrapping_add(1 as libc::c_int as libc::c_ulong),
    ) as *mut libc::c_char;
    strcpy(current, b"\0" as *const u8 as *const libc::c_char);
    let mut out: *mut *mut libc::c_char = malloc(0 as libc::c_int as libc::c_ulong)
        as *mut *mut libc::c_char;
    *out_size = 0 as libc::c_int;
    if !(strchr(txt, ' ' as i32)).is_null() {
        let mut temp: *mut libc::c_char = malloc(
            (strlen(txt)).wrapping_add(2 as libc::c_int as libc::c_ulong),
        ) as *mut libc::c_char;
        strcpy(temp, txt);
        strcat(temp, b" \0" as *const u8 as *const libc::c_char);
        i = 0 as libc::c_int;
        while (i as libc::c_ulong) < strlen(temp) {
            if *temp.offset(i as isize) as libc::c_int == ' ' as i32 {
                if strlen(current) > 0 as libc::c_int as libc::c_ulong {
                    out = realloc(
                        out as *mut libc::c_void,
                        ((*out_size + 1 as libc::c_int) as libc::c_ulong)
                            .wrapping_mul(
                                ::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong,
                            ),
                    ) as *mut *mut libc::c_char;
                    let ref mut fresh0 = *out.offset(*out_size as isize);
                    *fresh0 = malloc(
                        (strlen(current)).wrapping_add(1 as libc::c_int as libc::c_ulong),
                    ) as *mut libc::c_char;
                    strcpy(*out.offset(*out_size as isize), current);
                    *out_size += 1;
                    *out_size;
                }
                strcpy(current, b"\0" as *const u8 as *const libc::c_char);
            } else {
                strncat(
                    current,
                    &mut *temp.offset(i as isize),
                    1 as libc::c_int as libc::c_ulong,
                );
            }
            i += 1;
            i;
        }
        free(temp as *mut libc::c_void);
        free(current as *mut libc::c_void);
        return out;
    }
    if !(strchr(txt, ',' as i32)).is_null() {
        let mut temp_0: *mut libc::c_char = malloc(
            (strlen(txt)).wrapping_add(2 as libc::c_int as libc::c_ulong),
        ) as *mut libc::c_char;
        strcpy(temp_0, txt);
        strcat(temp_0, b",\0" as *const u8 as *const libc::c_char);
        i = 0 as libc::c_int;
        while (i as libc::c_ulong) < strlen(temp_0) {
            if *temp_0.offset(i as isize) as libc::c_int == ',' as i32 {
                if strlen(current) > 0 as libc::c_int as libc::c_ulong {
                    out = realloc(
                        out as *mut libc::c_void,
                        ((*out_size + 1 as libc::c_int) as libc::c_ulong)
                            .wrapping_mul(
                                ::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong,
                            ),
                    ) as *mut *mut libc::c_char;
                    let ref mut fresh1 = *out.offset(*out_size as isize);
                    *fresh1 = malloc(
                        (strlen(current)).wrapping_add(1 as libc::c_int as libc::c_ulong),
                    ) as *mut libc::c_char;
                    strcpy(*out.offset(*out_size as isize), current);
                    *out_size += 1;
                    *out_size;
                }
                strcpy(current, b"\0" as *const u8 as *const libc::c_char);
            } else {
                strncat(
                    current,
                    &mut *temp_0.offset(i as isize),
                    1 as libc::c_int as libc::c_ulong,
                );
            }
            i += 1;
            i;
        }
        free(temp_0 as *mut libc::c_void);
        free(current as *mut libc::c_void);
        return out;
    }
    let mut num: libc::c_int = 0 as libc::c_int;
    i = 0 as libc::c_int;
    while (i as libc::c_ulong) < strlen(txt) {
        if *txt.offset(i as isize) as libc::c_int >= 'a' as i32
            && *txt.offset(i as isize) as libc::c_int <= 'z' as i32
            && (*txt.offset(i as isize) as libc::c_int - 'a' as i32) % 2 as libc::c_int
                == 0 as libc::c_int
        {
            num += 1;
            num;
        }
        i += 1;
        i;
    }
    out = realloc(
        out as *mut libc::c_void,
        ((*out_size + 1 as libc::c_int) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong),
    ) as *mut *mut libc::c_char;
    let ref mut fresh2 = *out.offset(*out_size as isize);
    *fresh2 = malloc(12 as libc::c_int as libc::c_ulong) as *mut libc::c_char;
    sprintf(
        *out.offset(*out_size as isize),
        b"%d\0" as *const u8 as *const libc::c_char,
        num,
    );
    *out_size += 1;
    *out_size;
    free(current as *mut libc::c_void);
    return out;
}
