use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn realloc(_: *mut libc::c_void, _: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct FloatArray {
    pub data: *mut libc::c_float,
    pub size: libc::c_int,
}
#[no_mangle]
pub unsafe extern "C" fn createFloatArray(mut size: libc::c_int) -> FloatArray {
    let mut arr: FloatArray = FloatArray {
        data: 0 as *mut libc::c_float,
        size: 0,
    };
    arr
        .data = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_float>() as libc::c_ulong),
    ) as *mut libc::c_float;
    arr.size = size;
    return arr;
}
#[no_mangle]
pub unsafe extern "C" fn freeFloatArray(mut arr: *mut FloatArray) {
    free((*arr).data as *mut libc::c_void);
    (*arr).data = 0 as *mut libc::c_float;
    (*arr).size = 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn get_positive(mut l: FloatArray) -> FloatArray {
    let mut out: FloatArray = createFloatArray(0 as libc::c_int);
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < l.size {
        if *(l.data).offset(i as isize) > 0 as libc::c_int as libc::c_float {
            out.size += 1;
            out.size;
            out
                .data = realloc(
                out.data as *mut libc::c_void,
                (out.size as libc::c_ulong)
                    .wrapping_mul(
                        ::core::mem::size_of::<libc::c_float>() as libc::c_ulong,
                    ),
            ) as *mut libc::c_float;
            *(out.data)
                .offset(
                    (out.size - 1 as libc::c_int) as isize,
                ) = *(l.data).offset(i as isize);
        }
        i += 1;
        i;
    }
    return out;
}
