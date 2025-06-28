use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn strdup(_: *const libc::c_char) -> *mut libc::c_char;
}
#[no_mangle]
pub unsafe extern "C" fn numerical_letter_grade(
    mut grades: *mut libc::c_float,
    mut size: libc::c_int,
) -> *mut *mut libc::c_char {
    let mut out: *mut *mut libc::c_char = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong),
    ) as *mut *mut libc::c_char;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if *grades.offset(i as isize) as libc::c_double >= 3.9999f64 {
            let ref mut fresh0 = *out.offset(i as isize);
            *fresh0 = strdup(b"A+\0" as *const u8 as *const libc::c_char);
        } else if *grades.offset(i as isize) as libc::c_double > 3.7001f64
            && (*grades.offset(i as isize) as libc::c_double) < 3.9999f64
        {
            let ref mut fresh1 = *out.offset(i as isize);
            *fresh1 = strdup(b"A\0" as *const u8 as *const libc::c_char);
        } else if *grades.offset(i as isize) as libc::c_double > 3.3001f64
            && *grades.offset(i as isize) as libc::c_double <= 3.7001f64
        {
            let ref mut fresh2 = *out.offset(i as isize);
            *fresh2 = strdup(b"A-\0" as *const u8 as *const libc::c_char);
        } else if *grades.offset(i as isize) as libc::c_double > 3.0001f64
            && *grades.offset(i as isize) as libc::c_double <= 3.3001f64
        {
            let ref mut fresh3 = *out.offset(i as isize);
            *fresh3 = strdup(b"B+\0" as *const u8 as *const libc::c_char);
        } else if *grades.offset(i as isize) as libc::c_double > 2.7001f64
            && *grades.offset(i as isize) as libc::c_double <= 3.0001f64
        {
            let ref mut fresh4 = *out.offset(i as isize);
            *fresh4 = strdup(b"B\0" as *const u8 as *const libc::c_char);
        } else if *grades.offset(i as isize) as libc::c_double > 2.3001f64
            && *grades.offset(i as isize) as libc::c_double <= 2.7001f64
        {
            let ref mut fresh5 = *out.offset(i as isize);
            *fresh5 = strdup(b"B-\0" as *const u8 as *const libc::c_char);
        } else if *grades.offset(i as isize) as libc::c_double > 2.0001f64
            && *grades.offset(i as isize) as libc::c_double <= 2.3001f64
        {
            let ref mut fresh6 = *out.offset(i as isize);
            *fresh6 = strdup(b"C+\0" as *const u8 as *const libc::c_char);
        } else if *grades.offset(i as isize) as libc::c_double > 1.7001f64
            && *grades.offset(i as isize) as libc::c_double <= 2.0001f64
        {
            let ref mut fresh7 = *out.offset(i as isize);
            *fresh7 = strdup(b"C\0" as *const u8 as *const libc::c_char);
        } else if *grades.offset(i as isize) as libc::c_double > 1.3001f64
            && *grades.offset(i as isize) as libc::c_double <= 1.7001f64
        {
            let ref mut fresh8 = *out.offset(i as isize);
            *fresh8 = strdup(b"C-\0" as *const u8 as *const libc::c_char);
        } else if *grades.offset(i as isize) as libc::c_double > 1.0001f64
            && *grades.offset(i as isize) as libc::c_double <= 1.3001f64
        {
            let ref mut fresh9 = *out.offset(i as isize);
            *fresh9 = strdup(b"D+\0" as *const u8 as *const libc::c_char);
        } else if *grades.offset(i as isize) as libc::c_double > 0.7001f64
            && *grades.offset(i as isize) as libc::c_double <= 1.0001f64
        {
            let ref mut fresh10 = *out.offset(i as isize);
            *fresh10 = strdup(b"D\0" as *const u8 as *const libc::c_char);
        } else if *grades.offset(i as isize) as libc::c_double > 0.0001f64
            && *grades.offset(i as isize) as libc::c_double <= 0.7001f64
        {
            let ref mut fresh11 = *out.offset(i as isize);
            *fresh11 = strdup(b"D-\0" as *const u8 as *const libc::c_char);
        } else if *grades.offset(i as isize) as libc::c_double <= 0.0001f64 {
            let ref mut fresh12 = *out.offset(i as isize);
            *fresh12 = strdup(b"E\0" as *const u8 as *const libc::c_char);
        }
        i += 1;
        i;
    }
    return out;
}
