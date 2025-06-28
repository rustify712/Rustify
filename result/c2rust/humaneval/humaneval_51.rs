use ::libc;
extern "C" {
    fn strchr(_: *const libc::c_char, _: libc::c_int) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn remove_vowels(
    mut text: *const libc::c_char,
) -> *mut libc::c_char {
    let mut len: libc::c_int = strlen(text) as libc::c_int;
    let mut out: *mut libc::c_char = malloc((len + 1 as libc::c_int) as libc::c_ulong)
        as *mut libc::c_char;
    if out.is_null() {
        return 0 as *mut libc::c_char;
    }
    let mut vowels: *const libc::c_char = b"AEIOUaeiou\0" as *const u8
        as *const libc::c_char;
    let mut out_index: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < len {
        if (strchr(vowels, *text.offset(i as isize) as libc::c_int)).is_null() {
            let fresh0 = out_index;
            out_index = out_index + 1;
            *out.offset(fresh0 as isize) = *text.offset(i as isize);
        }
        i += 1;
        i;
    }
    *out.offset(out_index as isize) = '\0' as i32 as libc::c_char;
    return out;
}
