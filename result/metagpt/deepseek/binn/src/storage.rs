//! 内存管理模块

use std::alloc::{alloc, dealloc, Layout};
use std::ffi::c_void;
use std::ptr;

use crate::types::*;

// 全局内存分配函数指针
static mut MALLOC_FN: Option<unsafe extern "C" fn(usize) -> *mut c_void> = None;
static mut REALLOC_FN: Option<unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void> = None;
static mut FREE_FN: Option<unsafe extern "C" fn(*mut c_void)> = None;

/// 设置自定义内存分配函数
pub unsafe fn binn_set_alloc_functions(
    new_malloc: unsafe extern "C" fn(usize) -> *mut c_void,
    new_realloc: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void,
    new_free: unsafe extern "C" fn(*mut c_void),
) {
    MALLOC_FN = Some(new_malloc);
    REALLOC_FN = Some(new_realloc);
    FREE_FN = Some(new_free);
}