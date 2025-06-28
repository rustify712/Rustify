use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn abs(_: libc::c_int) -> libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn compare3(
    mut game: *mut libc::c_int,
    mut guess: *mut libc::c_int,
    mut size: libc::c_int,
) -> *mut libc::c_int {
    let mut out: *mut libc::c_int = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        *out
            .offset(
                i as isize,
            ) = abs(*game.offset(i as isize) - *guess.offset(i as isize));
        i += 1;
        i;
    }
    return out;
}
