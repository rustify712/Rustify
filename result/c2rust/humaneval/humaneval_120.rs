use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn qsort(
        __base: *mut libc::c_void,
        __nmemb: size_t,
        __size: size_t,
        __compar: __compar_fn_t,
    );
}
pub type size_t = libc::c_ulong;
pub type __compar_fn_t = Option::<
    unsafe extern "C" fn(*const libc::c_void, *const libc::c_void) -> libc::c_int,
>;
#[no_mangle]
pub unsafe extern "C" fn compare1(
    mut a: *const libc::c_void,
    mut b: *const libc::c_void,
) -> libc::c_int {
    return *(a as *mut libc::c_int) - *(b as *mut libc::c_int);
}
#[no_mangle]
pub unsafe extern "C" fn maximum(
    mut arr: *mut libc::c_int,
    mut arrSize: libc::c_int,
    mut k: libc::c_int,
    mut returnSize: *mut libc::c_int,
) -> *mut libc::c_int {
    qsort(
        arr as *mut libc::c_void,
        arrSize as size_t,
        ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
        Some(
            compare1
                as unsafe extern "C" fn(
                    *const libc::c_void,
                    *const libc::c_void,
                ) -> libc::c_int,
        ),
    );
    let mut out: *mut libc::c_int = malloc(
        (k as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < k {
        *out.offset(i as isize) = *arr.offset((arrSize - k + i) as isize);
        i += 1;
        i;
    }
    *returnSize = k;
    return out;
}
