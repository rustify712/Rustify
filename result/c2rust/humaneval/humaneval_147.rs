use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn calloc(_: libc::c_ulong, _: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn get_matrix_triples(mut n: libc::c_int) -> libc::c_int {
    let mut a: *mut libc::c_int = malloc(
        (n as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut sum: *mut *mut libc::c_int = malloc(
        ((n + 1 as libc::c_int) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_int>() as libc::c_ulong),
    ) as *mut *mut libc::c_int;
    let mut sum2: *mut *mut libc::c_int = malloc(
        ((n + 1 as libc::c_int) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_int>() as libc::c_ulong),
    ) as *mut *mut libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i <= n {
        let ref mut fresh0 = *sum.offset(i as isize);
        *fresh0 = calloc(
            3 as libc::c_int as libc::c_ulong,
            ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
        ) as *mut libc::c_int;
        let ref mut fresh1 = *sum2.offset(i as isize);
        *fresh1 = calloc(
            3 as libc::c_int as libc::c_ulong,
            ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
        ) as *mut libc::c_int;
        i += 1;
        i;
    }
    let mut i_0: libc::c_int = 1 as libc::c_int;
    while i_0 <= n {
        *a
            .offset(
                (i_0 - 1 as libc::c_int) as isize,
            ) = (i_0 * i_0 - i_0 + 1 as libc::c_int) % 3 as libc::c_int;
        let mut j: libc::c_int = 0 as libc::c_int;
        while j < 3 as libc::c_int {
            *(*sum.offset(i_0 as isize))
                .offset(
                    j as isize,
                ) = *(*sum.offset((i_0 - 1 as libc::c_int) as isize)).offset(j as isize);
            j += 1;
            j;
        }
        *(*sum.offset(i_0 as isize))
            .offset(*a.offset((i_0 - 1 as libc::c_int) as isize) as isize)
            += 1 as libc::c_int;
        i_0 += 1;
        i_0;
    }
    let mut times: libc::c_int = 1 as libc::c_int;
    while times < 3 as libc::c_int {
        let mut i_1: libc::c_int = 1 as libc::c_int;
        while i_1 <= n {
            let mut j_0: libc::c_int = 0 as libc::c_int;
            while j_0 < 3 as libc::c_int {
                *(*sum2.offset(i_1 as isize))
                    .offset(
                        j_0 as isize,
                    ) = *(*sum2.offset((i_1 - 1 as libc::c_int) as isize))
                    .offset(j_0 as isize);
                j_0 += 1;
                j_0;
            }
            if i_1 >= 1 as libc::c_int {
                let mut j_1: libc::c_int = 0 as libc::c_int;
                while j_1 < 3 as libc::c_int {
                    *(*sum2.offset(i_1 as isize))
                        .offset(
                            ((*a.offset((i_1 - 1 as libc::c_int) as isize) + j_1)
                                % 3 as libc::c_int) as isize,
                        )
                        += *(*sum.offset((i_1 - 1 as libc::c_int) as isize))
                            .offset(j_1 as isize);
                    j_1 += 1;
                    j_1;
                }
            }
            i_1 += 1;
            i_1;
        }
        let mut i_2: libc::c_int = 0 as libc::c_int;
        while i_2 <= n {
            let mut j_2: libc::c_int = 0 as libc::c_int;
            while j_2 < 3 as libc::c_int {
                *(*sum.offset(i_2 as isize))
                    .offset(
                        j_2 as isize,
                    ) = *(*sum2.offset(i_2 as isize)).offset(j_2 as isize);
                *(*sum2.offset(i_2 as isize)).offset(j_2 as isize) = 0 as libc::c_int;
                j_2 += 1;
                j_2;
            }
            i_2 += 1;
            i_2;
        }
        times += 1;
        times;
    }
    let mut result: libc::c_int = *(*sum.offset(n as isize))
        .offset(0 as libc::c_int as isize);
    let mut i_3: libc::c_int = 0 as libc::c_int;
    while i_3 <= n {
        free(*sum.offset(i_3 as isize) as *mut libc::c_void);
        free(*sum2.offset(i_3 as isize) as *mut libc::c_void);
        i_3 += 1;
        i_3;
    }
    free(sum as *mut libc::c_void);
    free(sum2 as *mut libc::c_void);
    free(a as *mut libc::c_void);
    return result;
}
