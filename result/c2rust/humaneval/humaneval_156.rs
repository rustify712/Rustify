use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn strcat(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
}
#[no_mangle]
pub unsafe extern "C" fn int_to_mini_roman(
    mut number: libc::c_int,
) -> *mut libc::c_char {
    let mut current: *mut libc::c_char = malloc(
        (100 as libc::c_int as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    *current.offset(0 as libc::c_int as isize) = '\0' as i32 as libc::c_char;
    let mut rep: [*const libc::c_char; 13] = [
        b"m\0" as *const u8 as *const libc::c_char,
        b"cm\0" as *const u8 as *const libc::c_char,
        b"d\0" as *const u8 as *const libc::c_char,
        b"cd\0" as *const u8 as *const libc::c_char,
        b"c\0" as *const u8 as *const libc::c_char,
        b"xc\0" as *const u8 as *const libc::c_char,
        b"l\0" as *const u8 as *const libc::c_char,
        b"xl\0" as *const u8 as *const libc::c_char,
        b"x\0" as *const u8 as *const libc::c_char,
        b"ix\0" as *const u8 as *const libc::c_char,
        b"v\0" as *const u8 as *const libc::c_char,
        b"iv\0" as *const u8 as *const libc::c_char,
        b"i\0" as *const u8 as *const libc::c_char,
    ];
    let mut num: [libc::c_int; 13] = [
        1000 as libc::c_int,
        900 as libc::c_int,
        500 as libc::c_int,
        400 as libc::c_int,
        100 as libc::c_int,
        90 as libc::c_int,
        50 as libc::c_int,
        40 as libc::c_int,
        10 as libc::c_int,
        9 as libc::c_int,
        5 as libc::c_int,
        4 as libc::c_int,
        1 as libc::c_int,
    ];
    let mut pos: libc::c_int = 0 as libc::c_int;
    while number > 0 as libc::c_int {
        while number >= num[pos as usize] {
            strcat(current, rep[pos as usize]);
            number -= num[pos as usize];
        }
        if number > 0 as libc::c_int {
            pos += 1 as libc::c_int;
        }
    }
    return current;
}
