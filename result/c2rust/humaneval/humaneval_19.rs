use ::libc;
extern "C" {
    fn strcat(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strcmp(_: *const libc::c_char, _: *const libc::c_char) -> libc::c_int;
    fn strdup(_: *const libc::c_char) -> *mut libc::c_char;
    fn strtok(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn sort_numbers(
    mut numbers: *const libc::c_char,
) -> *mut libc::c_char {
    let mut tonum_keys: [*const libc::c_char; 10] = [
        b"zero\0" as *const u8 as *const libc::c_char,
        b"one\0" as *const u8 as *const libc::c_char,
        b"two\0" as *const u8 as *const libc::c_char,
        b"three\0" as *const u8 as *const libc::c_char,
        b"four\0" as *const u8 as *const libc::c_char,
        b"five\0" as *const u8 as *const libc::c_char,
        b"six\0" as *const u8 as *const libc::c_char,
        b"seven\0" as *const u8 as *const libc::c_char,
        b"eight\0" as *const u8 as *const libc::c_char,
        b"nine\0" as *const u8 as *const libc::c_char,
    ];
    let mut tonum_values: [libc::c_int; 10] = [
        0 as libc::c_int,
        1 as libc::c_int,
        2 as libc::c_int,
        3 as libc::c_int,
        4 as libc::c_int,
        5 as libc::c_int,
        6 as libc::c_int,
        7 as libc::c_int,
        8 as libc::c_int,
        9 as libc::c_int,
    ];
    let mut numto_keys: [*const libc::c_char; 10] = [
        b"zero\0" as *const u8 as *const libc::c_char,
        b"one\0" as *const u8 as *const libc::c_char,
        b"two\0" as *const u8 as *const libc::c_char,
        b"three\0" as *const u8 as *const libc::c_char,
        b"four\0" as *const u8 as *const libc::c_char,
        b"five\0" as *const u8 as *const libc::c_char,
        b"six\0" as *const u8 as *const libc::c_char,
        b"seven\0" as *const u8 as *const libc::c_char,
        b"eight\0" as *const u8 as *const libc::c_char,
        b"nine\0" as *const u8 as *const libc::c_char,
    ];
    let mut count: [libc::c_int; 10] = [0 as libc::c_int, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut input: *mut libc::c_char = strdup(numbers);
    let mut token: *mut libc::c_char = strtok(
        input,
        b" \0" as *const u8 as *const libc::c_char,
    );
    while !token.is_null() {
        let mut i: libc::c_int = 0 as libc::c_int;
        while i < 10 as libc::c_int {
            if strcmp(token, tonum_keys[i as usize]) == 0 as libc::c_int {
                count[i as usize] += 1;
                count[i as usize];
                break;
            } else {
                i += 1;
                i;
            }
        }
        token = strtok(
            0 as *mut libc::c_char,
            b" \0" as *const u8 as *const libc::c_char,
        );
    }
    let mut out: *mut libc::c_char = malloc(
        (1000 as libc::c_int as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    *out.offset(0 as libc::c_int as isize) = '\0' as i32 as libc::c_char;
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while i_0 < 10 as libc::c_int {
        let mut j: libc::c_int = 0 as libc::c_int;
        while j < count[i_0 as usize] {
            strcat(out, numto_keys[i_0 as usize]);
            strcat(out, b" \0" as *const u8 as *const libc::c_char);
            j += 1;
            j;
        }
        i_0 += 1;
        i_0;
    }
    if strlen(out) > 0 as libc::c_int as libc::c_ulong {
        *out
            .offset(
                (strlen(out)).wrapping_sub(1 as libc::c_int as libc::c_ulong) as isize,
            ) = '\0' as i32 as libc::c_char;
    }
    free(input as *mut libc::c_void);
    return out;
}
