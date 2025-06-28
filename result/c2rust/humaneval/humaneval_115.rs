use ::libc;
#[no_mangle]
pub unsafe extern "C" fn max_fill(
    mut grid: *mut *mut libc::c_int,
    mut gridSize: libc::c_int,
    mut gridColSize: *mut libc::c_int,
    mut capacity: libc::c_int,
) -> libc::c_int {
    let mut out: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < gridSize {
        let mut sum: libc::c_int = 0 as libc::c_int;
        let mut j: libc::c_int = 0 as libc::c_int;
        while j < *gridColSize.offset(i as isize) {
            sum += *(*grid.offset(i as isize)).offset(j as isize);
            j += 1;
            j;
        }
        if sum > 0 as libc::c_int {
            out += (sum - 1 as libc::c_int) / capacity + 1 as libc::c_int;
        }
        i += 1;
        i;
    }
    return out;
}
