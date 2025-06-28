use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn realloc(_: *mut libc::c_void, _: libc::c_ulong) -> *mut libc::c_void;
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
pub unsafe extern "C" fn compare4(
    mut a: *const libc::c_void,
    mut b: *const libc::c_void,
) -> libc::c_int {
    return *(a as *mut libc::c_int) - *(b as *mut libc::c_int);
}
#[no_mangle]
pub unsafe extern "C" fn unique(
    mut arr: *mut libc::c_int,
    mut size: libc::c_int,
    mut result_size: *mut libc::c_int,
) -> *mut libc::c_int {
    if size == 0 as libc::c_int {
        *result_size = 0 as libc::c_int;
        return 0 as *mut libc::c_int;
    }
    qsort(
        arr as *mut libc::c_void,
        size as size_t,
        ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
        Some(
            compare4
                as unsafe extern "C" fn(
                    *const libc::c_void,
                    *const libc::c_void,
                ) -> libc::c_int,
        ),
    );
    let mut result: *mut libc::c_int = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    if result.is_null() {
        *result_size = 0 as libc::c_int;
        return 0 as *mut libc::c_int;
    }
    *result.offset(0 as libc::c_int as isize) = *arr.offset(0 as libc::c_int as isize);
    let mut j: libc::c_int = 1 as libc::c_int;
    let mut i: libc::c_int = 1 as libc::c_int;
    while i < size {
        if *arr.offset(i as isize) != *arr.offset((i - 1 as libc::c_int) as isize) {
            let fresh0 = j;
            j = j + 1;
            *result.offset(fresh0 as isize) = *arr.offset(i as isize);
        }
        i += 1;
        i;
    }
    result = realloc(
        result as *mut libc::c_void,
        (j as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    *result_size = j;
    return result;
}
