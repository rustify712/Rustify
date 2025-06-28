use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn realloc(_: *mut libc::c_void, _: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn get_odd_collatz(
    mut n: libc::c_int,
    mut returnSize: *mut libc::c_int,
) -> *mut libc::c_int {
    let mut capacity: libc::c_int = 10 as libc::c_int;
    let mut out: *mut libc::c_int = malloc(
        (capacity as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut size: libc::c_int = 0 as libc::c_int;
    let fresh0 = size;
    size = size + 1;
    *out.offset(fresh0 as isize) = 1 as libc::c_int;
    while n != 1 as libc::c_int {
        if n % 2 as libc::c_int == 1 as libc::c_int {
            if size >= capacity {
                capacity *= 2 as libc::c_int;
                out = realloc(
                    out as *mut libc::c_void,
                    (capacity as libc::c_ulong)
                        .wrapping_mul(
                            ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
                        ),
                ) as *mut libc::c_int;
            }
            let fresh1 = size;
            size = size + 1;
            *out.offset(fresh1 as isize) = n;
            n = n * 3 as libc::c_int + 1 as libc::c_int;
        } else {
            n = n / 2 as libc::c_int;
        }
    }
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size - 1 as libc::c_int {
        let mut j: libc::c_int = i + 1 as libc::c_int;
        while j < size {
            if *out.offset(i as isize) > *out.offset(j as isize) {
                let mut temp: libc::c_int = *out.offset(i as isize);
                *out.offset(i as isize) = *out.offset(j as isize);
                *out.offset(j as isize) = temp;
            }
            j += 1;
            j;
        }
        i += 1;
        i;
    }
    *returnSize = size;
    return out;
}
