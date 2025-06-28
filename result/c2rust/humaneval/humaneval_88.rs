use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn sort_array2(
    mut array: *mut libc::c_int,
    mut size: libc::c_int,
) -> *mut libc::c_int {
    if size == 0 as libc::c_int {
        let mut empty_array: *mut libc::c_int = malloc(
            ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
        ) as *mut libc::c_int;
        *empty_array.offset(0 as libc::c_int as isize) = 0 as libc::c_int;
        return empty_array;
    }
    let mut sorted_array: *mut libc::c_int = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        *sorted_array.offset(i as isize) = *array.offset(i as isize);
        i += 1;
        i;
    }
    if (*array.offset(0 as libc::c_int as isize)
        + *array.offset((size - 1 as libc::c_int) as isize)) % 2 as libc::c_int
        == 1 as libc::c_int
    {
        let mut i_0: libc::c_int = 0 as libc::c_int;
        while i_0 < size - 1 as libc::c_int {
            let mut j: libc::c_int = i_0 + 1 as libc::c_int;
            while j < size {
                if *sorted_array.offset(i_0 as isize) > *sorted_array.offset(j as isize)
                {
                    let mut temp: libc::c_int = *sorted_array.offset(i_0 as isize);
                    *sorted_array
                        .offset(i_0 as isize) = *sorted_array.offset(j as isize);
                    *sorted_array.offset(j as isize) = temp;
                }
                j += 1;
                j;
            }
            i_0 += 1;
            i_0;
        }
    } else {
        let mut i_1: libc::c_int = 0 as libc::c_int;
        while i_1 < size - 1 as libc::c_int {
            let mut j_0: libc::c_int = i_1 + 1 as libc::c_int;
            while j_0 < size {
                if *sorted_array.offset(i_1 as isize)
                    < *sorted_array.offset(j_0 as isize)
                {
                    let mut temp_0: libc::c_int = *sorted_array.offset(i_1 as isize);
                    *sorted_array
                        .offset(i_1 as isize) = *sorted_array.offset(j_0 as isize);
                    *sorted_array.offset(j_0 as isize) = temp_0;
                }
                j_0 += 1;
                j_0;
            }
            i_1 += 1;
            i_1;
        }
    }
    return sorted_array;
}
