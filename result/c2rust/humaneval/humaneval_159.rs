use ::libc;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Result_0 {
    pub total_eaten: libc::c_int,
    pub carrots_left: libc::c_int,
}
#[no_mangle]
pub unsafe extern "C" fn eat(
    mut number: libc::c_int,
    mut need: libc::c_int,
    mut remaining: libc::c_int,
) -> Result_0 {
    let mut result: Result_0 = Result_0 {
        total_eaten: 0,
        carrots_left: 0,
    };
    if need > remaining {
        result.total_eaten = number + remaining;
        result.carrots_left = 0 as libc::c_int;
    } else {
        result.total_eaten = number + need;
        result.carrots_left = remaining - need;
    }
    return result;
}
