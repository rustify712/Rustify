use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn minPath(
    mut grid: *mut *mut libc::c_int,
    mut gridSize: libc::c_int,
    mut gridColSize: *mut libc::c_int,
    mut k: libc::c_int,
    mut returnSize: *mut libc::c_int,
) -> *mut libc::c_int {
    let mut i: libc::c_int = 0;
    let mut j: libc::c_int = 0;
    let mut x: libc::c_int = 0;
    let mut y: libc::c_int = 0;
    let mut min: libc::c_int = 0;
    i = 0 as libc::c_int;
    while i < gridSize {
        j = 0 as libc::c_int;
        while j < *gridColSize.offset(i as isize) {
            if *(*grid.offset(i as isize)).offset(j as isize) == 1 as libc::c_int {
                x = i;
                y = j;
            }
            j += 1;
            j;
        }
        i += 1;
        i;
    }
    min = gridSize * gridSize;
    if x > 0 as libc::c_int
        && *(*grid.offset((x - 1 as libc::c_int) as isize)).offset(y as isize) < min
    {
        min = *(*grid.offset((x - 1 as libc::c_int) as isize)).offset(y as isize);
    }
    if x < gridSize - 1 as libc::c_int
        && *(*grid.offset((x + 1 as libc::c_int) as isize)).offset(y as isize) < min
    {
        min = *(*grid.offset((x + 1 as libc::c_int) as isize)).offset(y as isize);
    }
    if y > 0 as libc::c_int
        && *(*grid.offset(x as isize)).offset((y - 1 as libc::c_int) as isize) < min
    {
        min = *(*grid.offset(x as isize)).offset((y - 1 as libc::c_int) as isize);
    }
    if y < gridSize - 1 as libc::c_int
        && *(*grid.offset(x as isize)).offset((y + 1 as libc::c_int) as isize) < min
    {
        min = *(*grid.offset(x as isize)).offset((y + 1 as libc::c_int) as isize);
    }
    let mut out: *mut libc::c_int = malloc(
        (k as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    *returnSize = k;
    i = 0 as libc::c_int;
    while i < k {
        if i % 2 as libc::c_int == 0 as libc::c_int {
            *out.offset(i as isize) = 1 as libc::c_int;
        } else {
            *out.offset(i as isize) = min;
        }
        i += 1;
        i;
    }
    return out;
}
