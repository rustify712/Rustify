use ::libc;
extern "C" {
    fn strcmp(_: *const libc::c_char, _: *const libc::c_char) -> libc::c_int;
    fn strdup(_: *const libc::c_char) -> *mut libc::c_char;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
}
#[no_mangle]
pub static mut planets: [*const libc::c_char; 8] = [
    b"Mercury\0" as *const u8 as *const libc::c_char,
    b"Venus\0" as *const u8 as *const libc::c_char,
    b"Earth\0" as *const u8 as *const libc::c_char,
    b"Mars\0" as *const u8 as *const libc::c_char,
    b"Jupiter\0" as *const u8 as *const libc::c_char,
    b"Saturn\0" as *const u8 as *const libc::c_char,
    b"Uranus\0" as *const u8 as *const libc::c_char,
    b"Neptune\0" as *const u8 as *const libc::c_char,
];
#[no_mangle]
pub static mut num_planets: libc::c_int = 0;
#[no_mangle]
pub unsafe extern "C" fn find_planet_index(
    mut planet: *const libc::c_char,
) -> libc::c_int {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < num_planets {
        if strcmp(planets[i as usize], planet) == 0 as libc::c_int {
            return i;
        }
        i += 1;
        i;
    }
    return -(1 as libc::c_int);
}
#[no_mangle]
pub unsafe extern "C" fn bf(
    mut planet1: *const libc::c_char,
    mut planet2: *const libc::c_char,
    mut out_size: *mut libc::c_int,
) -> *mut *mut libc::c_char {
    let mut pos1: libc::c_int = find_planet_index(planet1);
    let mut pos2: libc::c_int = find_planet_index(planet2);
    if pos1 == -(1 as libc::c_int) || pos2 == -(1 as libc::c_int) {
        *out_size = 0 as libc::c_int;
        return 0 as *mut *mut libc::c_char;
    }
    if pos1 > pos2 {
        let mut temp: libc::c_int = pos1;
        pos1 = pos2;
        pos2 = temp;
    }
    *out_size = pos2 - pos1 - 1 as libc::c_int;
    if *out_size <= 0 as libc::c_int {
        *out_size = 0 as libc::c_int;
        return 0 as *mut *mut libc::c_char;
    }
    let mut out: *mut *mut libc::c_char = malloc(
        (*out_size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong),
    ) as *mut *mut libc::c_char;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < *out_size {
        let ref mut fresh0 = *out.offset(i as isize);
        *fresh0 = strdup(planets[(pos1 + 1 as libc::c_int + i) as usize]);
        i += 1;
        i;
    }
    return out;
}
#[no_mangle]
pub unsafe extern "C" fn free_bf_result(
    mut result: *mut *mut libc::c_char,
    mut size: libc::c_int,
) {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        free(*result.offset(i as isize) as *mut libc::c_void);
        i += 1;
        i;
    }
    free(result as *mut libc::c_void);
}
unsafe extern "C" fn run_static_initializers() {
    num_planets = (::core::mem::size_of::<[*const libc::c_char; 8]>() as libc::c_ulong)
        .wrapping_div(::core::mem::size_of::<*const libc::c_char>() as libc::c_ulong)
        as libc::c_int;
}
#[used]
#[cfg_attr(target_os = "linux", link_section = ".init_array")]
#[cfg_attr(target_os = "windows", link_section = ".CRT$XIB")]
#[cfg_attr(target_os = "macos", link_section = "__DATA,__mod_init_func")]
static INIT_ARRAY: [unsafe extern "C" fn(); 1] = [run_static_initializers];
