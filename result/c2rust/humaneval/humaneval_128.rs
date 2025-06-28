use ::libc;
extern "C" {
    fn abs(_: libc::c_int) -> libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn prod_signs(
    mut arr: *mut libc::c_int,
    mut size: libc::c_int,
) -> libc::c_int {
    if size == 0 as libc::c_int {
        return -(32768 as libc::c_int);
    }
    let mut i: libc::c_int = 0;
    let mut sum: libc::c_int = 0 as libc::c_int;
    let mut prods: libc::c_int = 1 as libc::c_int;
    i = 0 as libc::c_int;
    while i < size {
        sum += abs(*arr.offset(i as isize));
        if *arr.offset(i as isize) == 0 as libc::c_int {
            prods = 0 as libc::c_int;
        }
        if *arr.offset(i as isize) < 0 as libc::c_int {
            prods = -prods;
        }
        i += 1;
        i;
    }
    return sum * prods;
}
