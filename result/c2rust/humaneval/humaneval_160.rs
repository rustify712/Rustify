use ::libc;
extern "C" {
    fn pow(_: libc::c_double, _: libc::c_double) -> libc::c_double;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
    fn strcmp(_: *const libc::c_char, _: *const libc::c_char) -> libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn do_algebra(
    mut operato: *mut *mut libc::c_char,
    mut operand: *mut libc::c_int,
    mut operato_size: libc::c_int,
    mut operand_size: libc::c_int,
) -> libc::c_int {
    let mut posto: *mut libc::c_int = malloc(
        (operand_size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < operand_size {
        *posto.offset(i as isize) = i;
        i += 1;
        i;
    }
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while i_0 < operato_size {
        if strcmp(
            *operato.offset(i_0 as isize),
            b"**\0" as *const u8 as *const libc::c_char,
        ) == 0 as libc::c_int
        {
            while *posto.offset(*posto.offset(i_0 as isize) as isize)
                != *posto.offset(i_0 as isize)
            {
                *posto
                    .offset(
                        i_0 as isize,
                    ) = *posto.offset(*posto.offset(i_0 as isize) as isize);
            }
            while *posto
                .offset(*posto.offset((i_0 + 1 as libc::c_int) as isize) as isize)
                != *posto.offset((i_0 + 1 as libc::c_int) as isize)
            {
                *posto
                    .offset(
                        (i_0 + 1 as libc::c_int) as isize,
                    ) = *posto
                    .offset(*posto.offset((i_0 + 1 as libc::c_int) as isize) as isize);
            }
            *operand
                .offset(
                    *posto.offset(i_0 as isize) as isize,
                ) = pow(
                *operand.offset(*posto.offset(i_0 as isize) as isize) as libc::c_double,
                *operand
                    .offset(*posto.offset((i_0 + 1 as libc::c_int) as isize) as isize)
                    as libc::c_double,
            ) as libc::c_int;
            *posto
                .offset((i_0 + 1 as libc::c_int) as isize) = *posto.offset(i_0 as isize);
        }
        i_0 += 1;
        i_0;
    }
    let mut i_1: libc::c_int = 0 as libc::c_int;
    while i_1 < operato_size {
        if strcmp(
            *operato.offset(i_1 as isize),
            b"*\0" as *const u8 as *const libc::c_char,
        ) == 0 as libc::c_int
            || strcmp(
                *operato.offset(i_1 as isize),
                b"//\0" as *const u8 as *const libc::c_char,
            ) == 0 as libc::c_int
        {
            while *posto.offset(*posto.offset(i_1 as isize) as isize)
                != *posto.offset(i_1 as isize)
            {
                *posto
                    .offset(
                        i_1 as isize,
                    ) = *posto.offset(*posto.offset(i_1 as isize) as isize);
            }
            while *posto
                .offset(*posto.offset((i_1 + 1 as libc::c_int) as isize) as isize)
                != *posto.offset((i_1 + 1 as libc::c_int) as isize)
            {
                *posto
                    .offset(
                        (i_1 + 1 as libc::c_int) as isize,
                    ) = *posto
                    .offset(*posto.offset((i_1 + 1 as libc::c_int) as isize) as isize);
            }
            if strcmp(
                *operato.offset(i_1 as isize),
                b"*\0" as *const u8 as *const libc::c_char,
            ) == 0 as libc::c_int
            {
                *operand
                    .offset(
                        *posto.offset(i_1 as isize) as isize,
                    ) = *operand.offset(*posto.offset(i_1 as isize) as isize)
                    * *operand
                        .offset(
                            *posto.offset((i_1 + 1 as libc::c_int) as isize) as isize,
                        );
            } else {
                *operand
                    .offset(
                        *posto.offset(i_1 as isize) as isize,
                    ) = *operand.offset(*posto.offset(i_1 as isize) as isize)
                    / *operand
                        .offset(
                            *posto.offset((i_1 + 1 as libc::c_int) as isize) as isize,
                        );
            }
            *posto
                .offset((i_1 + 1 as libc::c_int) as isize) = *posto.offset(i_1 as isize);
        }
        i_1 += 1;
        i_1;
    }
    let mut i_2: libc::c_int = 0 as libc::c_int;
    while i_2 < operato_size {
        if strcmp(
            *operato.offset(i_2 as isize),
            b"+\0" as *const u8 as *const libc::c_char,
        ) == 0 as libc::c_int
            || strcmp(
                *operato.offset(i_2 as isize),
                b"-\0" as *const u8 as *const libc::c_char,
            ) == 0 as libc::c_int
        {
            while *posto.offset(*posto.offset(i_2 as isize) as isize)
                != *posto.offset(i_2 as isize)
            {
                *posto
                    .offset(
                        i_2 as isize,
                    ) = *posto.offset(*posto.offset(i_2 as isize) as isize);
            }
            while *posto
                .offset(*posto.offset((i_2 + 1 as libc::c_int) as isize) as isize)
                != *posto.offset((i_2 + 1 as libc::c_int) as isize)
            {
                *posto
                    .offset(
                        (i_2 + 1 as libc::c_int) as isize,
                    ) = *posto
                    .offset(*posto.offset((i_2 + 1 as libc::c_int) as isize) as isize);
            }
            if strcmp(
                *operato.offset(i_2 as isize),
                b"+\0" as *const u8 as *const libc::c_char,
            ) == 0 as libc::c_int
            {
                *operand
                    .offset(
                        *posto.offset(i_2 as isize) as isize,
                    ) = *operand.offset(*posto.offset(i_2 as isize) as isize)
                    + *operand
                        .offset(
                            *posto.offset((i_2 + 1 as libc::c_int) as isize) as isize,
                        );
            } else {
                *operand
                    .offset(
                        *posto.offset(i_2 as isize) as isize,
                    ) = *operand.offset(*posto.offset(i_2 as isize) as isize)
                    - *operand
                        .offset(
                            *posto.offset((i_2 + 1 as libc::c_int) as isize) as isize,
                        );
            }
            *posto
                .offset((i_2 + 1 as libc::c_int) as isize) = *posto.offset(i_2 as isize);
        }
        i_2 += 1;
        i_2;
    }
    let mut result: libc::c_int = *operand.offset(0 as libc::c_int as isize);
    free(posto as *mut libc::c_void);
    return result;
}
