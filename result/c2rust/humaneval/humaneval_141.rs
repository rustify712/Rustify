use ::libc;
extern "C" {
    fn strcmp(_: *const libc::c_char, _: *const libc::c_char) -> libc::c_int;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn file_name_check(
    mut file_name: *const libc::c_char,
) -> *const libc::c_char {
    let mut numdigit: libc::c_int = 0 as libc::c_int;
    let mut numdot: libc::c_int = 0 as libc::c_int;
    let mut length: libc::c_int = strlen(file_name) as libc::c_int;
    if length < 5 as libc::c_int {
        return b"No\0" as *const u8 as *const libc::c_char;
    }
    let mut w: libc::c_char = *file_name.offset(0 as libc::c_int as isize);
    if !(w as libc::c_int >= 'A' as i32 && w as libc::c_int <= 'Z' as i32
        || w as libc::c_int >= 'a' as i32 && w as libc::c_int <= 'z' as i32)
    {
        return b"No\0" as *const u8 as *const libc::c_char;
    }
    let mut last: *const libc::c_char = file_name
        .offset(length as isize)
        .offset(-(4 as libc::c_int as isize));
    if strcmp(last, b".txt\0" as *const u8 as *const libc::c_char) != 0 as libc::c_int
        && strcmp(last, b".exe\0" as *const u8 as *const libc::c_char)
            != 0 as libc::c_int
        && strcmp(last, b".dll\0" as *const u8 as *const libc::c_char)
            != 0 as libc::c_int
    {
        return b"No\0" as *const u8 as *const libc::c_char;
    }
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < length {
        if *file_name.offset(i as isize) as libc::c_int >= '0' as i32
            && *file_name.offset(i as isize) as libc::c_int <= '9' as i32
        {
            numdigit += 1;
            numdigit;
        }
        if *file_name.offset(i as isize) as libc::c_int == '.' as i32 {
            numdot += 1;
            numdot;
        }
        i += 1;
        i;
    }
    if numdigit > 3 as libc::c_int || numdot != 1 as libc::c_int {
        return b"No\0" as *const u8 as *const libc::c_char;
    }
    return b"Yes\0" as *const u8 as *const libc::c_char;
}
