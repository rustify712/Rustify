use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strcat(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn words_string(
    mut s: *const libc::c_char,
    mut word_count: *mut libc::c_int,
) -> *mut *mut libc::c_char {
    let mut len: libc::c_int = strlen(s) as libc::c_int;
    let mut str: *mut libc::c_char = malloc(
        ((len + 1 as libc::c_int) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    strcpy(str, s);
    strcat(str, b" \0" as *const u8 as *const libc::c_char);
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < len + 1 as libc::c_int {
        if *str.offset(i as isize) as libc::c_int == ' ' as i32
            || *str.offset(i as isize) as libc::c_int == ',' as i32
        {
            count += 1;
            count;
        }
        i += 1;
        i;
    }
    let mut words: *mut *mut libc::c_char = malloc(
        (count as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong),
    ) as *mut *mut libc::c_char;
    let mut word_index: libc::c_int = 0 as libc::c_int;
    let mut current: *mut libc::c_char = malloc(
        ((len + 1 as libc::c_int) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    let mut current_index: libc::c_int = 0 as libc::c_int;
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while i_0 < len + 1 as libc::c_int {
        if *str.offset(i_0 as isize) as libc::c_int == ' ' as i32
            || *str.offset(i_0 as isize) as libc::c_int == ',' as i32
        {
            if current_index > 0 as libc::c_int {
                *current.offset(current_index as isize) = '\0' as i32 as libc::c_char;
                let ref mut fresh0 = *words.offset(word_index as isize);
                *fresh0 = malloc(
                    ((current_index + 1 as libc::c_int) as libc::c_ulong)
                        .wrapping_mul(
                            ::core::mem::size_of::<libc::c_char>() as libc::c_ulong,
                        ),
                ) as *mut libc::c_char;
                strcpy(*words.offset(word_index as isize), current);
                word_index += 1;
                word_index;
                current_index = 0 as libc::c_int;
            }
        } else {
            let fresh1 = current_index;
            current_index = current_index + 1;
            *current.offset(fresh1 as isize) = *str.offset(i_0 as isize);
        }
        i_0 += 1;
        i_0;
    }
    *word_count = word_index;
    free(current as *mut libc::c_void);
    free(str as *mut libc::c_void);
    return words;
}
