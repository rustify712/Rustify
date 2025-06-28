use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn find(
    mut arr: *mut libc::c_int,
    mut size: libc::c_int,
    mut element: libc::c_int,
) -> libc::c_int {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if *arr.offset(i as isize) == element {
            return 1 as libc::c_int;
        }
        i += 1;
        i;
    }
    return 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn sort1(mut arr: *mut libc::c_int, mut size: libc::c_int) {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size - 1 as libc::c_int {
        let mut j: libc::c_int = 0 as libc::c_int;
        while j < size - i - 1 as libc::c_int {
            if *arr.offset(j as isize) > *arr.offset((j + 1 as libc::c_int) as isize) {
                let mut temp: libc::c_int = *arr.offset(j as isize);
                *arr.offset(j as isize) = *arr.offset((j + 1 as libc::c_int) as isize);
                *arr.offset((j + 1 as libc::c_int) as isize) = temp;
            }
            j += 1;
            j;
        }
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn common(
    mut l1: *mut libc::c_int,
    mut size1: libc::c_int,
    mut l2: *mut libc::c_int,
    mut size2: libc::c_int,
    mut outSize: *mut libc::c_int,
) -> *mut libc::c_int {
    let mut out: *mut libc::c_int = malloc(
        (size1 as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size1 {
        if find(out, count, *l1.offset(i as isize)) == 0 {
            if find(l2, size2, *l1.offset(i as isize)) != 0 {
                let fresh0 = count;
                count = count + 1;
                *out.offset(fresh0 as isize) = *l1.offset(i as isize);
            }
        }
        i += 1;
        i;
    }
    sort1(out, count);
    *outSize = count;
    return out;
}
