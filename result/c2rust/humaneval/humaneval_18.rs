use ::libc;
extern "C" {
    fn strncmp(
        _: *const libc::c_char,
        _: *const libc::c_char,
        _: libc::c_ulong,
    ) -> libc::c_int;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn how_many_times(
    mut str: *const libc::c_char,
    mut substring: *const libc::c_char,
) -> libc::c_int {
    let mut out: libc::c_int = 0 as libc::c_int;
    let mut str_len: libc::c_int = strlen(str) as libc::c_int;
    let mut sub_len: libc::c_int = strlen(substring) as libc::c_int;
    if str_len == 0 as libc::c_int {
        return 0 as libc::c_int;
    }
    let mut i: libc::c_int = 0 as libc::c_int;
    while i <= str_len - sub_len {
        if strncmp(str.offset(i as isize), substring, sub_len as libc::c_ulong)
            == 0 as libc::c_int
        {
            out += 1 as libc::c_int;
        }
        i += 1;
        i;
    }
    return out;
}
