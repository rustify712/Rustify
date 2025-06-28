use ::libc;
extern "C" {
    fn strchr(_: *const libc::c_char, _: libc::c_int) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn get_closest_vowel(
    mut word: *const libc::c_char,
) -> *mut libc::c_char {
    static mut out: [libc::c_char; 2] = unsafe {
        *::core::mem::transmute::<&[u8; 2], &mut [libc::c_char; 2]>(b"\0\0")
    };
    let mut vowels: *const libc::c_char = b"AEIOUaeiou\0" as *const u8
        as *const libc::c_char;
    let mut len: libc::c_int = strlen(word) as libc::c_int;
    let mut i: libc::c_int = len - 2 as libc::c_int;
    while i >= 1 as libc::c_int {
        if !(strchr(vowels, *word.offset(i as isize) as libc::c_int)).is_null() {
            if (strchr(
                vowels,
                *word.offset((i + 1 as libc::c_int) as isize) as libc::c_int,
            ))
                .is_null()
                && (strchr(
                    vowels,
                    *word.offset((i - 1 as libc::c_int) as isize) as libc::c_int,
                ))
                    .is_null()
            {
                out[0 as libc::c_int as usize] = *word.offset(i as isize);
                out[1 as libc::c_int as usize] = '\0' as i32 as libc::c_char;
                return out.as_mut_ptr();
            }
        }
        i -= 1;
        i;
    }
    return out.as_mut_ptr();
}
