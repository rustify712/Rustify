use ::libc;
extern "C" {
    fn strcmp(_: *const libc::c_char, _: *const libc::c_char) -> libc::c_int;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn is_palindrome2(mut text: *const libc::c_char) -> bool {
    let mut length: libc::c_int = strlen(text) as libc::c_int;
    let vla = (length + 1 as libc::c_int) as usize;
    let mut reversed: Vec::<libc::c_char> = ::std::vec::from_elem(0, vla);
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < length {
        *reversed
            .as_mut_ptr()
            .offset(i as isize) = *text.offset((length - 1 as libc::c_int - i) as isize);
        i += 1;
        i;
    }
    *reversed.as_mut_ptr().offset(length as isize) = '\0' as i32 as libc::c_char;
    return strcmp(reversed.as_mut_ptr(), text) == 0 as libc::c_int;
}
