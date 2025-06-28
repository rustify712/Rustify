use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn is_bored(mut S: *const libc::c_char) -> libc::c_int {
    let mut isstart: libc::c_int = 1 as libc::c_int;
    let mut isi: libc::c_int = 0 as libc::c_int;
    let mut sum: libc::c_int = 0 as libc::c_int;
    let mut len: libc::c_int = strlen(S) as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < len {
        if *S.offset(i as isize) as libc::c_int == ' ' as i32 && isi != 0 {
            isi = 0 as libc::c_int;
            sum += 1 as libc::c_int;
        }
        if *S.offset(i as isize) as libc::c_int == 'I' as i32 && isstart != 0 {
            isi = 1 as libc::c_int;
        } else {
            isi = 0 as libc::c_int;
        }
        if *S.offset(i as isize) as libc::c_int != ' ' as i32 {
            isstart = 0 as libc::c_int;
        }
        if *S.offset(i as isize) as libc::c_int == '.' as i32
            || *S.offset(i as isize) as libc::c_int == '?' as i32
            || *S.offset(i as isize) as libc::c_int == '!' as i32
        {
            isstart = 1 as libc::c_int;
        }
        i += 1;
        i;
    }
    return sum;
}
