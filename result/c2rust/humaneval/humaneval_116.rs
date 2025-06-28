use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
    fn abs(_: libc::c_int) -> libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn sort_array1(
    mut arr: *mut libc::c_int,
    mut size: libc::c_int,
) -> *mut libc::c_int {
    let mut bin: *mut libc::c_int = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut m: libc::c_int = 0;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        let mut b: libc::c_int = 0 as libc::c_int;
        let mut n: libc::c_int = abs(*arr.offset(i as isize));
        while n > 0 as libc::c_int {
            b += n % 2 as libc::c_int;
            n = n / 2 as libc::c_int;
        }
        *bin.offset(i as isize) = b;
        i += 1;
        i;
    }
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while i_0 < size {
        let mut j: libc::c_int = 1 as libc::c_int;
        while j < size {
            if *bin.offset(j as isize) < *bin.offset((j - 1 as libc::c_int) as isize)
                || *bin.offset(j as isize)
                    == *bin.offset((j - 1 as libc::c_int) as isize)
                    && *arr.offset(j as isize)
                        < *arr.offset((j - 1 as libc::c_int) as isize)
            {
                m = *arr.offset(j as isize);
                *arr.offset(j as isize) = *arr.offset((j - 1 as libc::c_int) as isize);
                *arr.offset((j - 1 as libc::c_int) as isize) = m;
                m = *bin.offset(j as isize);
                *bin.offset(j as isize) = *bin.offset((j - 1 as libc::c_int) as isize);
                *bin.offset((j - 1 as libc::c_int) as isize) = m;
            }
            j += 1;
            j;
        }
        i_0 += 1;
        i_0;
    }
    free(bin as *mut libc::c_void);
    return arr;
}
