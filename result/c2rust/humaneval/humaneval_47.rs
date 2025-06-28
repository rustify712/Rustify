use ::libc;
extern "C" {
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
pub unsafe extern "C" fn compare_floats(
    mut a: *const libc::c_void,
    mut b: *const libc::c_void,
) -> libc::c_int {
    let mut fa: libc::c_float = *(a as *const libc::c_float);
    let mut fb: libc::c_float = *(b as *const libc::c_float);
    return (fa > fb) as libc::c_int - (fa < fb) as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn median(
    mut l: *mut libc::c_float,
    mut size: libc::c_int,
) -> libc::c_float {
    qsort(
        l as *mut libc::c_void,
        size as size_t,
        ::core::mem::size_of::<libc::c_float>() as libc::c_ulong,
        Some(
            compare_floats
                as unsafe extern "C" fn(
                    *const libc::c_void,
                    *const libc::c_void,
                ) -> libc::c_int,
        ),
    );
    if size % 2 as libc::c_int == 1 as libc::c_int {
        return *l.offset((size / 2 as libc::c_int) as isize)
    } else {
        return (0.5f64
            * (*l.offset((size / 2 as libc::c_int) as isize)
                + *l.offset((size / 2 as libc::c_int - 1 as libc::c_int) as isize))
                as libc::c_double) as libc::c_float
    };
}
