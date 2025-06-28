use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn check_if_last_char_is_a_letter(
    mut txt: *const libc::c_char,
) -> bool {
    let mut length: libc::c_int = strlen(txt) as libc::c_int;
    if length == 0 as libc::c_int {
        return 0 as libc::c_int != 0;
    }
    let mut chr: libc::c_char = *txt.offset((length - 1 as libc::c_int) as isize);
    if (chr as libc::c_int) < 65 as libc::c_int
        || chr as libc::c_int > 90 as libc::c_int
            && (chr as libc::c_int) < 97 as libc::c_int
        || chr as libc::c_int > 122 as libc::c_int
    {
        return 0 as libc::c_int != 0;
    }
    if length == 1 as libc::c_int {
        return 1 as libc::c_int != 0;
    }
    chr = *txt.offset((length - 2 as libc::c_int) as isize);
    if chr as libc::c_int >= 65 as libc::c_int && chr as libc::c_int <= 90 as libc::c_int
        || chr as libc::c_int >= 97 as libc::c_int
            && chr as libc::c_int <= 122 as libc::c_int
    {
        return 0 as libc::c_int != 0;
    }
    return 1 as libc::c_int != 0;
}
