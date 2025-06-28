use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn sort2(mut arr: *mut libc::c_int, mut size: libc::c_int) {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size - 1 as libc::c_int {
        let mut j: libc::c_int = 0 as libc::c_int;
        while j < size - i - 1 as libc::c_int {
            if *arr.offset(j as isize) > *arr.offset((j + 1 as libc::c_int) as isize) {
                let mut temp: libc::c_int = *arr.offset(j as isize);
                *arr.offset(j as isize) = *arr.offset((j + 1 as libc::c_int) as isize);
                *arr.offset((j + 1 as libc::c_int) as isize) = temp;
            }
            j += 1;
            j;
        }
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn reverse(mut arr: *mut libc::c_int, mut size: libc::c_int) {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size / 2 as libc::c_int {
        let mut temp: libc::c_int = *arr.offset(i as isize);
        *arr.offset(i as isize) = *arr.offset((size - i - 1 as libc::c_int) as isize);
        *arr.offset((size - i - 1 as libc::c_int) as isize) = temp;
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn numToWord(mut num: libc::c_int) -> *const libc::c_char {
    match num {
        0 => return b"Zero\0" as *const u8 as *const libc::c_char,
        1 => return b"One\0" as *const u8 as *const libc::c_char,
        2 => return b"Two\0" as *const u8 as *const libc::c_char,
        3 => return b"Three\0" as *const u8 as *const libc::c_char,
        4 => return b"Four\0" as *const u8 as *const libc::c_char,
        5 => return b"Five\0" as *const u8 as *const libc::c_char,
        6 => return b"Six\0" as *const u8 as *const libc::c_char,
        7 => return b"Seven\0" as *const u8 as *const libc::c_char,
        8 => return b"Eight\0" as *const u8 as *const libc::c_char,
        9 => return b"Nine\0" as *const u8 as *const libc::c_char,
        _ => return 0 as *const libc::c_char,
    };
}
#[no_mangle]
pub unsafe extern "C" fn by_length(
    mut arr: *mut libc::c_int,
    mut size: libc::c_int,
    mut result_size: *mut libc::c_int,
) -> *mut *mut libc::c_char {
    sort2(arr, size);
    reverse(arr, size);
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if *arr.offset(i as isize) >= 1 as libc::c_int
            && *arr.offset(i as isize) <= 9 as libc::c_int
        {
            count += 1;
            count;
        }
        i += 1;
        i;
    }
    let mut result: *mut *mut libc::c_char = malloc(
        (count as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong),
    ) as *mut *mut libc::c_char;
    if result.is_null() {
        *result_size = 0 as libc::c_int;
        return 0 as *mut *mut libc::c_char;
    }
    let mut index: libc::c_int = 0 as libc::c_int;
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while i_0 < size {
        if *arr.offset(i_0 as isize) >= 1 as libc::c_int
            && *arr.offset(i_0 as isize) <= 9 as libc::c_int
        {
            let mut word: *const libc::c_char = numToWord(*arr.offset(i_0 as isize));
            let ref mut fresh0 = *result.offset(index as isize);
            *fresh0 = malloc(
                (strlen(word))
                    .wrapping_add(1 as libc::c_int as libc::c_ulong)
                    .wrapping_mul(
                        ::core::mem::size_of::<libc::c_char>() as libc::c_ulong,
                    ),
            ) as *mut libc::c_char;
            if (*result.offset(index as isize)).is_null() {
                let mut j: libc::c_int = 0 as libc::c_int;
                while j < index {
                    free(*result.offset(j as isize) as *mut libc::c_void);
                    j += 1;
                    j;
                }
                free(result as *mut libc::c_void);
                *result_size = 0 as libc::c_int;
                return 0 as *mut *mut libc::c_char;
            }
            strcpy(*result.offset(index as isize), word);
            index += 1;
            index;
        }
        i_0 += 1;
        i_0;
    }
    *result_size = count;
    return result;
}
