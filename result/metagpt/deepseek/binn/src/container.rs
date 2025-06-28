//! 容器操作模块

use std::ptr;
use std::mem;
use std::ffi::c_void;

use crate::types::*;
use crate::storage::*;

/// 创建新的binn容器
pub unsafe fn binn_new(type_: i32, size: i32, buffer: *mut u8) -> *mut binn {
    let ptr = if let Some(malloc) = MALLOC_FN {
        malloc(mem::size_of::<binn>()) as *mut binn
    } else {
        alloc(Layout::new::<binn>()) as *mut binn
    };
    
    if !ptr.is_null() {
        (*ptr).header = 0;
        (*ptr).allocated = true;
        (*ptr).writable = true;
        (*ptr).dirty = false;
        (*ptr).pbuf = buffer;
        (*ptr).pre_allocated = !buffer.is_null();
        (*ptr).alloc_size = size;
        (*ptr).used_size = 0;
        (*ptr).type_ = type_;
        (*ptr).ptr = ptr::null_mut();
        (*ptr).size = 0;
        (*ptr).count = 0;
        (*ptr).freefn = None;
        (*ptr).disable_int_compression = false;
    }
    
    ptr
}

/// 释放binn容器
pub unsafe fn binn_free(item: *mut binn) {
    if item.is_null() {
        return;
    }
    
    if (*item).allocated {
        if let Some(free) = FREE_FN {
            free(item as *mut c_void);
        } else {
            dealloc(item as *mut u8, Layout::new::<binn>());
        }
    }
}

/// 清空容器内容
pub unsafe fn binn_list_free(ptr: *mut binn) {
    if !ptr.is_null() {
        binn_free(ptr);
    }
}