use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn realloc(_: *mut libc::c_void, _: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Coordinate {
    pub row: libc::c_int,
    pub col: libc::c_int,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CoordinateArray {
    pub data: *mut Coordinate,
    pub size: libc::c_int,
    pub capacity: libc::c_int,
}
#[no_mangle]
pub unsafe extern "C" fn initCoordinateArray(
    mut arr: *mut CoordinateArray,
    mut capacity: libc::c_int,
) {
    (*arr)
        .data = malloc(
        (capacity as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<Coordinate>() as libc::c_ulong),
    ) as *mut Coordinate;
    (*arr).size = 0 as libc::c_int;
    (*arr).capacity = capacity;
}
#[no_mangle]
pub unsafe extern "C" fn pushBack(mut arr: *mut CoordinateArray, mut coord: Coordinate) {
    if (*arr).size == (*arr).capacity {
        (*arr).capacity *= 2 as libc::c_int;
        (*arr)
            .data = realloc(
            (*arr).data as *mut libc::c_void,
            ((*arr).capacity as libc::c_ulong)
                .wrapping_mul(::core::mem::size_of::<Coordinate>() as libc::c_ulong),
        ) as *mut Coordinate;
    }
    let fresh0 = (*arr).size;
    (*arr).size = (*arr).size + 1;
    *((*arr).data).offset(fresh0 as isize) = coord;
}
#[no_mangle]
pub unsafe extern "C" fn freeCoordinateArray(mut arr: *mut CoordinateArray) {
    free((*arr).data as *mut libc::c_void);
    (*arr).data = 0 as *mut Coordinate;
    (*arr).capacity = 0 as libc::c_int;
    (*arr).size = (*arr).capacity;
}
#[no_mangle]
pub unsafe extern "C" fn get_row(
    mut lst: *mut *mut libc::c_int,
    mut row_sizes: *mut libc::c_int,
    mut num_rows: libc::c_int,
    mut x: libc::c_int,
) -> CoordinateArray {
    let mut out: CoordinateArray = CoordinateArray {
        data: 0 as *mut Coordinate,
        size: 0,
        capacity: 0,
    };
    initCoordinateArray(&mut out, 10 as libc::c_int);
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < num_rows {
        let mut j: libc::c_int = *row_sizes.offset(i as isize) - 1 as libc::c_int;
        while j >= 0 as libc::c_int {
            if *(*lst.offset(i as isize)).offset(j as isize) == x {
                let mut coord: Coordinate = {
                    let mut init = Coordinate { row: i, col: j };
                    init
                };
                pushBack(&mut out, coord);
            }
            j -= 1;
            j;
        }
        i += 1;
        i;
    }
    return out;
}
