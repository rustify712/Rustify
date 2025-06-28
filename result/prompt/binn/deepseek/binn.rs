use std::ffi::CStr;
use std::os::raw::c_void;
use std::ptr;
use std::mem;
use std::cmp;
use std::fmt;
use std::str;
use std::string::String;
use std::vec::Vec;
use std::boxed::Box;
use std::sync::Arc;
use std::cell::RefCell;
use std::collections::HashMap;

// Constants
const BINN_VERSION: &str = "3.0.0";
const INVALID_BINN: i32 = 0;

// Storage Data Types
const BINN_STORAGE_NOBYTES: u8 = 0x00;
const BINN_STORAGE_BYTE: u8 = 0x20;
const BINN_STORAGE_WORD: u8 = 0x40;
const BINN_STORAGE_DWORD: u8 = 0x60;
const BINN_STORAGE_QWORD: u8 = 0x80;
const BINN_STORAGE_STRING: u8 = 0xA0;
const BINN_STORAGE_BLOB: u8 = 0xC0;
const BINN_STORAGE_CONTAINER: u8 = 0xE0;
const BINN_STORAGE_VIRTUAL: u32 = 0x80000;

const BINN_STORAGE_MIN: u8 = BINN_STORAGE_NOBYTES;
const BINN_STORAGE_MAX: u8 = BINN_STORAGE_CONTAINER;

const BINN_STORAGE_MASK: u8 = 0xE0;
const BINN_STORAGE_MASK16: u16 = 0xE000;
const BINN_STORAGE_HAS_MORE: u8 = 0x10;
const BINN_TYPE_MASK: u8 = 0x0F;
const BINN_TYPE_MASK16: u16 = 0x0FFF;

const BINN_MAX_VALUE_MASK: u32 = 0xFFFFF;

// Data Formats
const BINN_LIST: u8 = 0xE0;
const BINN_MAP: u8 = 0xE1;
const BINN_OBJECT: u8 = 0xE2;

const BINN_NULL: u8 = 0x00;
const BINN_TRUE: u8 = 0x01;
const BINN_FALSE: u8 = 0x02;

const BINN_UINT8: u8 = 0x20;
const BINN_INT8: u8 = 0x21;
const BINN_UINT16: u8 = 0x40;
const BINN_INT16: u8 = 0x41;
const BINN_UINT32: u8 = 0x60;
const BINN_INT32: u8 = 0x61;
const BINN_UINT64: u8 = 0x80;
const BINN_INT64: u8 = 0x81;

const BINN_SCHAR: u8 = BINN_INT8;
const BINN_UCHAR: u8 = BINN_UINT8;

const BINN_STRING: u8 = 0xA0;
const BINN_DATETIME: u8 = 0xA1;
const BINN_DATE: u8 = 0xA2;
const BINN_TIME: u8 = 0xA3;
const BINN_DECIMAL: u8 = 0xA4;
const BINN_CURRENCYSTR: u8 = 0xA5;
const BINN_SINGLE_STR: u8 = 0xA6;
const BINN_DOUBLE_STR: u8 = 0xA7;

const BINN_FLOAT32: u8 = 0x62;
const BINN_FLOAT64: u8 = 0x82;
const BINN_FLOAT: u8 = BINN_FLOAT32;
const BINN_SINGLE: u8 = BINN_FLOAT32;
const BINN_DOUBLE: u8 = BINN_FLOAT64;

const BINN_CURRENCY: u8 = 0x83;
const BINN_BLOB: u8 = 0xC0;

// Virtual types
const BINN_BOOL: u32 = 0x80061;

// Extended content types
const BINN_HTML: u16 = 0xB001;
const BINN_XML: u16 = 0xB002;
const BINN_JSON: u16 = 0xB003;
const BINN_JAVASCRIPT: u16 = 0xB004;
const BINN_CSS: u16 = 0xB005;

const BINN_JPEG: u16 = 0xD001;
const BINN_GIF: u16 = 0xD002;
const BINN_PNG: u16 = 0xD003;
const BINN_BMP: u16 = 0xD004;

// Type families
const BINN_FAMILY_NONE: u8 = 0x00;
const BINN_FAMILY_NULL: u8 = 0xF1;
const BINN_FAMILY_INT: u8 = 0xF2;
const BINN_FAMILY_FLOAT: u8 = 0xF3;
const BINN_FAMILY_STRING: u8 = 0xF4;
const BINN_FAMILY_BLOB: u8 = 0xF5;
const BINN_FAMILY_BOOL: u8 = 0xF6;
const BINN_FAMILY_BINN: u8 = 0xF7;

// Integer types related to signal
const BINN_SIGNED_INT: u8 = 11;
const BINN_UNSIGNED_INT: u8 = 22;

// BINN Structure
#[repr(C)]
struct BinnStruct {
    header: i32,
    allocated: bool,
    writable: bool,
    dirty: bool,
    pbuf: *mut c_void,
    pre_allocated: bool,
    alloc_size: i32,
    used_size: i32,
    type_: i32,
    ptr: *mut c_void,
    size: i32,
    count: i32,
    freefn: Option<extern "C" fn(*mut c_void)>,
    disable_int_compression: bool,
    vint8: i8,
    vint16: i16,
    vint32: i32,
    vint64: i64,
    vuint8: u8,
    vuint16: u16,
    vuint32: u32,
    vuint64: u64,
    vchar: i8,
    vuchar: u8,
    vshort: i16,
    vushort: u16,
    vint: i32,
    vuint: u32,
    vfloat: f32,
    vdouble: f64,
    vbool: bool,
}

type Binn = BinnStruct;

// General Functions
extern "C" fn binn_version() -> *const i8 {
    BINN_VERSION.as_ptr() as *const i8
}

extern "C" fn binn_set_alloc_functions(
    new_malloc: Option<extern "C" fn(usize) -> *mut c_void>,
    new_realloc: Option<extern "C" fn(*mut c_void, usize) -> *mut c_void>,
    new_free: Option<extern "C" fn(*mut c_void)>,
) {
    // Set the allocation functions
    // This is a placeholder, as Rust's memory management is different from C/C++
}

extern "C" fn binn_create_type(storage_type: i32, data_type_index: i32) -> i32 {
    if data_type_index < 0 {
        return -1;
    }
    if storage_type < BINN_STORAGE_MIN as i32 || storage_type > BINN_STORAGE_MAX as i32 {
        return -1;
    }
    if data_type_index < 16 {
        return storage_type | data_type_index;
    } else if data_type_index < 4096 {
        let storage_type = storage_type | (BINN_STORAGE_HAS_MORE as i32);
        let storage_type = storage_type << 8;
        let data_type_index = data_type_index >> 4;
        return storage_type | data_type_index;
    } else {
        return -1;
    }
}

extern "C" fn binn_get_type_info(long_type: i32, pstorage_type: *mut i32, pextra_type: *mut i32) -> bool {
    let mut storage_type = 0;
    let mut extra_type = 0;
    let mut retval = true;

    if long_type < 0 {
        retval = false;
    } else if long_type <= 0xff {
        storage_type = long_type & BINN_STORAGE_MASK as i32;
        extra_type = long_type & BINN_TYPE_MASK as i32;
    } else if long_type <= 0xffff {
        storage_type = long_type & BINN_STORAGE_MASK16 as i32;
        storage_type >>= 8;
        extra_type = long_type & BINN_TYPE_MASK16 as i32;
        extra_type >>= 4;
    } else if long_type & BINN_STORAGE_VIRTUAL as i32 != 0 {
        let long_type = long_type & 0xffff;
        return binn_get_type_info(long_type, pstorage_type, pextra_type);
    } else {
        storage_type = -1;
        extra_type = -1;
        retval = false;
    }

    if !pstorage_type.is_null() {
        unsafe { *pstorage_type = storage_type };
    }
    if !pextra_type.is_null() {
        unsafe { *pextra_type = extra_type };
    }

    retval
}

// Write Functions
extern "C" fn binn_new(type_: i32, size: i32, buffer: *mut c_void) -> *mut Binn {
    let item = Box::into_raw(Box::new(BinnStruct {
        header: 0,
        allocated: false,
        writable: false,
        dirty: false,
        pbuf: ptr::null_mut(),
        pre_allocated: false,
        alloc_size: 0,
        used_size: 0,
        type_: 0,
        ptr: ptr::null_mut(),
        size: 0,
        count: 0,
        freefn: None,
        disable_int_compression: false,
        vint8: 0,
        vint16: 0,
        vint32: 0,
        vint64: 0,
        vuint8: 0,
        vuint16: 0,
        vuint32: 0,
        vuint64: 0,
        vchar: 0,
        vuchar: 0,
        vshort: 0,
        vushort: 0,
        vint: 0,
        vuint: 0,
        vfloat: 0.0,
        vdouble: 0.0,
        vbool: false,
    }));

    if binn_create(item, type_, size, buffer) {
        unsafe { (*item).allocated = true };
        item
    } else {
        unsafe { Box::from_raw(item) };
        ptr::null_mut()
    }
}

extern "C" fn binn_create(item: *mut Binn, type_: i32, size: i32, pointer: *mut c_void) -> bool {
    if item.is_null() || size < 0 {
        return false;
    }

    unsafe {
        (*item).header = 0;
        (*item).allocated = false;
        (*item).writable = true;
        (*item).dirty = true;
        (*item).pbuf = ptr::null_mut();
        (*item).pre_allocated = false;
        (*item).alloc_size = 0;
        (*item).used_size = 0;
        (*item).type_ = type_;
        (*item).ptr = ptr::null_mut();
        (*item).size = 0;
        (*item).count = 0;
        (*item).freefn = None;
        (*item).disable_int_compression = false;

        if !pointer.is_null() {
            (*item).pre_allocated = true;
            (*item).pbuf = pointer;
            (*item).alloc_size = size;
        } else {
            (*item).pre_allocated = false;
            let size = if size == 0 { 256 } else { size };
            let pointer = libc::malloc(size as usize);
            if pointer.is_null() {
                return false;
            }
            (*item).pbuf = pointer;
            (*item).alloc_size = size;
        }

        (*item).header = 0x1F22B11F;
        (*item).used_size = 9;
        (*item).type_ = type_;
        (*item).dirty = true;
    }

    true
}

// Read Functions
extern "C" fn binn_ptr(ptr: *mut c_void) -> *mut c_void {
    if ptr.is_null() {
        return ptr::null_mut();
    }

    unsafe {
        if *(ptr as *const i32) == 0x1F22B11F {
            let item = ptr as *mut Binn;
            if (*item).writable && (*item).dirty {
                // Save header logic here
            }
            (*item).ptr
        } else {
            ptr
        }
    }
}

extern "C" fn binn_free(item: *mut Binn) {
    if item.is_null() {
        return;
    }

    unsafe {
        if (*item).writable && !(*item).pre_allocated {
            libc::free((*item).pbuf);
        }

        if let Some(freefn) = (*item).freefn {
            freefn((*item).ptr);
        }

        if (*item).allocated {
            Box::from_raw(item);
        } else {
            (*item).header = 0;
        }
    }
}


extern "C" fn binn_release(item: *mut Binn) -> *mut c_void {
    if item.is_null() {
        return ptr::null_mut();
    }

    unsafe {
        let data = binn_ptr(item as *mut c_void);
        if data > (*item).pbuf {
            libc::memmove((*item).pbuf, data, (*item).size as usize);
            data = (*item).pbuf;
        }

        if (*item).allocated {
            Box::from_raw(item);
        } else {
            (*item).header = 0;
        }

        data
    }
}

extern "C" fn binn_is_valid(ptr: *mut c_void, ptype: *mut i32, pcount: *mut i32, psize: *mut i32) -> bool {
    if ptr.is_null() {
        return false;
    }

    let mut type_ = 0;
    let mut count = 0;
    let mut size = 0;
    let mut header_size = 0;

    unsafe {
        if !IsValidBinnHeader(ptr, &mut type_, &mut count, &mut size, &mut header_size) {
            return false;
        }

        if !ptype.is_null() {
            *ptype = type_;
        }
        if !pcount.is_null() {
            *pcount = count;
        }
        if !psize.is_null() {
            *psize = size;
        }
    }

    true
}

extern "C" fn binn_list_add(list: *mut Binn, type_: i32, pvalue: *mut c_void, size: i32) -> bool {
    if list.is_null() || (*list).type_ != BINN_LIST as i32 || !(*list).writable {
        return false;
    }

    unsafe {
        if AddValue(list, type_, pvalue, size) {
            (*list).count += 1;
            true
        } else {
            false
        }
    }
}

extern "C" fn binn_map_set(map: *mut Binn, id: i32, type_: i32, pvalue: *mut c_void, size: i32) -> bool {
    if map.is_null() || (*map).type_ != BINN_MAP as i32 || !(*map).writable {
        return false;
    }

    unsafe {
        if AddValue(map, type_, pvalue, size) {
            (*map).count += 1;
            true
        } else {
            false
        }
    }
}

extern "C" fn binn_object_set(obj: *mut Binn, key: *const i8, type_: i32, pvalue: *mut c_void, size: i32) -> bool {
    if obj.is_null() || (*obj).type_ != BINN_OBJECT as i32 || !(*obj).writable {
        return false;
    }

    unsafe {
        if AddValue(obj, type_, pvalue, size) {
            (*obj).count += 1;
            true
        } else {
            false
        }
    }
}

extern "C" fn binn_value(type_: i32, pvalue: *mut c_void, size: i32, freefn: Option<extern "C" fn(*mut c_void)>) -> *mut Binn {
    let item = Box::into_raw(Box::new(BinnStruct {
        header: 0x1F22B11F,
        allocated: true,
        writable: false,
        dirty: false,
        pbuf: ptr::null_mut(),
        pre_allocated: false,
        alloc_size: 0,
        used_size: 0,
        type_: type_,
        ptr: ptr::null_mut(),
        size: 0,
        count: 0,
        freefn,
        disable_int_compression: false,
        vint8: 0,
        vint16: 0,
        vint32: 0,
        vint64: 0,
        vuint8: 0,
        vuint16: 0,
        vuint32: 0,
        vuint64: 0,
        vchar: 0,
        vuchar: 0,
        vshort: 0,
        vushort: 0,
        vint: 0,
        vuint: 0,
        vfloat: 0.0,
        vdouble: 0.0,
        vbool: false,
    }));

    unsafe {
        (*item).ptr = pvalue;
        (*item).size = size;
    }

    item
}

extern "C" fn binn_set_string(item: *mut Binn, str: *mut i8, pfree: Option<extern "C" fn(*mut c_void)>) -> bool {
    if item.is_null() || str.is_null() {
        return false;
    }

    unsafe {
        if pfree == Some(BINN_TRANSIENT) {
            let len = libc::strlen(str);
            let copy = libc::malloc(len + 1);
            if copy.is_null() {
                return false;
            }
            libc::strcpy(copy as *mut i8, str);
            (*item).ptr = copy;
            (*item).freefn = Some(libc::free);
        } else {
            (*item).ptr = str as *mut c_void;
            (*item).freefn = pfree;
        }

        (*item).type_ = BINN_STRING as i32;
        true
    }
}

extern "C" fn binn_set_blob(item: *mut Binn, ptr: *mut c_void, size: i32, pfree: Option<extern "C" fn(*mut c_void)>) -> bool {
    if item.is_null() || ptr.is_null() {
        return false;
    }

    unsafe {
        if pfree == Some(BINN_TRANSIENT) {
            let copy = libc::malloc(size as usize);
            if copy.is_null() {
                return false;
            }
            libc::memcpy(copy, ptr, size as usize);
            (*item).ptr = copy;
            (*item).freefn = Some(libc::free);
        } else {
            (*item).ptr = ptr;
            (*item).freefn = pfree;
        }

        (*item).type_ = BINN_BLOB as i32;
        (*item).size = size;
        true
    }
}

extern "C" fn binn_list_get_value(ptr: *mut c_void, pos: i32, value: *mut Binn) -> bool {
    if ptr.is_null() || value.is_null() {
        return false;
    }

    unsafe {
        let mut type_ = 0;
        let mut count = 0;
        let mut size = 0;
        let mut header_size = 0;

        if !IsValidBinnHeader(ptr, &mut type_, &mut count, &mut size, &mut header_size) {
            return false;
        }

        if type_ != BINN_LIST as i32 || count == 0 || pos <= 0 || pos > count {
            return false;
        }

        let mut p = ptr.add(header_size as usize);
        for _ in 0..(pos - 1) {
            p = AdvanceDataPos(p, ptr.add(size as usize));
            if p.is_null() || p < ptr {
                return false;
            }
        }

        GetValue(p, value)
    }
}

extern "C" fn binn_map_get_value(ptr: *mut c_void, id: i32, value: *mut Binn) -> bool {
    if ptr.is_null() || value.is_null() {
        return false;
    }

    unsafe {
        let mut type_ = 0;
        let mut count = 0;
        let mut size = 0;
        let mut header_size = 0;

        if !IsValidBinnHeader(ptr, &mut type_, &mut count, &mut size, &mut header_size) {
            return false;
        }

        if type_ != BINN_MAP as i32 || count == 0 {
            return false;
        }

        let p = SearchForID(ptr.add(header_size as usize), size, count, id);
        if p.is_null() {
            return false;
        }

        GetValue(p, value)
    }
}

extern "C" fn binn_object_get_value(ptr: *mut c_void, key: *const i8, value: *mut Binn) -> bool {
    if ptr.is_null() || key.is_null() || value.is_null() {
        return false;
    }

    unsafe {
        let mut type_ = 0;
        let mut count = 0;
        let mut size = 0;
        let mut header_size = 0;

        if !IsValidBinnHeader(ptr, &mut type_, &mut count, &mut size, &mut header_size) {
            return false;
        }

        if type_ != BINN_OBJECT as i32 || count == 0 {
            return false;
        }

        let p = SearchForKey(ptr.add(header_size as usize), size, count, key);
        if p.is_null() {
            return false;
        }

        GetValue(p, value)
    }
}

extern "C" fn binn_iter_init(iter: *mut BinnIter, ptr: *mut c_void, expected_type: i32) -> bool {
    if iter.is_null() || ptr.is_null() {
        return false;
    }

    unsafe {
        let mut type_ = 0;
        let mut count = 0;
        let mut size = 0;
        let mut header_size = 0;

        if !IsValidBinnHeader(ptr, &mut type_, &mut count, &mut size, &mut header_size) {
            return false;
        }

        if type_ != expected_type {
            return false;
        }

        (*iter).pnext = ptr.add(header_size as usize);
        (*iter).plimit = ptr.add(size as usize);
        (*iter).type_ = type_;
        (*iter).count = count;
        (*iter).current = 0;

        true
    }
}

extern "C" fn binn_list_next(iter: *mut BinnIter, value: *mut Binn) -> bool {
    if iter.is_null() || value.is_null() {
        return false;
    }

    unsafe {
        if (*iter).current >= (*iter).count {
            return false;
        }

        let pnow = (*iter).pnext;
        (*iter).pnext = AdvanceDataPos(pnow, (*iter).plimit);
        if (*iter).pnext.is_null() || (*iter).pnext < pnow {
            return false;
        }

        (*iter).current += 1;
        GetValue(pnow, value)
    }
}

extern "C" fn binn_map_next(iter: *mut BinnIter, pid: *mut i32, value: *mut Binn) -> bool {
    if iter.is_null() || value.is_null() {
        return false;
    }

    unsafe {
        if (*iter).current >= (*iter).count {
            return false;
        }

        let pnow = (*iter).pnext;
        let id = read_map_id(&mut (*iter).pnext, (*iter).plimit);
        if (*iter).pnext.is_null() || (*iter).pnext < pnow {
            return false;
        }

        if !pid.is_null() {
            *pid = id;
        }

        (*iter).current += 1;
        GetValue((*iter).pnext, value)
    }
}

extern "C" fn binn_object_next(iter: *mut BinnIter, pkey: *mut i8, value: *mut Binn) -> bool {
    if iter.is_null() || value.is_null() {
        return false;
    }

    unsafe {
        if (*iter).current >= (*iter).count {
            return false;
        }

        let pnow = (*iter).pnext;
        let len = *(*iter).pnext as usize;
        (*iter).pnext = (*iter).pnext.add(1);

        if !pkey.is_null() {
            libc::strncpy(pkey, (*iter).pnext as *const i8, len);
            *pkey.add(len) = 0;
        }

        (*iter).pnext = (*iter).pnext.add(len);
        (*iter).pnext = AdvanceDataPos((*iter).pnext, (*iter).plimit);
        if (*iter).pnext.is_null() || (*iter).pnext < pnow {
            return false;
        }

        (*iter).current += 1;
        GetValue((*iter).pnext, value)
    }
}

extern "C" fn binn_get_int32(value: *mut Binn, pint: *mut i32) -> bool {
    if value.is_null() || pint.is_null() {
        return false;
    }

    unsafe {
        match (*value).type_ {
            BINN_INT32 => {
                *pint = (*value).vint32;
                true
            }
            BINN_INT64 => {
                *pint = (*value).vint64 as i32;
                true
            }
            BINN_FLOAT32 => {
                *pint = (*value).vfloat as i32;
                true
            }
            BINN_FLOAT64 => {
                *pint = (*value).vdouble as i32;
                true
            }
            _ => false,
        }
    }
}

extern "C" fn binn_get_int64(value: *mut Binn, pint: *mut i64) -> bool {
    if value.is_null() || pint.is_null() {
        return false;
    }

    unsafe {
        match (*value).type_ {
            BINN_INT32 => {
                *pint = (*value).vint32 as i64;
                true
            }
            BINN_INT64 => {
                *pint = (*value).vint64;
                true
            }
            BINN_FLOAT32 => {
                *pint = (*value).vfloat as i64;
                true
            }
            BINN_FLOAT64 => {
                *pint = (*value).vdouble as i64;
                true
            }
            _ => false,
        }
    }
}

extern "C" fn binn_get_double(value: *mut Binn, pfloat: *mut f64) -> bool {
    if value.is_null() || pfloat.is_null() {
        return false;
    }

    unsafe {
        match (*value).type_ {
            BINN_FLOAT32 => {
                *pfloat = (*value).vfloat as f64;
                true
            }
            BINN_FLOAT64 => {
                *pfloat = (*value).vdouble;
                true
            }
            _ => false,
        }
    }
}

extern "C" fn binn_get_bool(value: *mut Binn, pbool: *mut bool) -> bool {
    if value.is_null() || pbool.is_null() {
        return false;
    }

    unsafe {
        match (*value).type_ {
            BINN_BOOL => {
                *pbool = (*value).vbool;
                true
            }
            BINN_TRUE => {
                *pbool = true;
                true
            }
            BINN_FALSE => {
                *pbool = false;
                true
            }
            _ => false,
        }
    }
}

extern "C" fn binn_get_str(value: *mut Binn) -> *const i8 {
    if value.is_null() {
        return ptr::null();
    }

    unsafe {
        match (*value).type_ {
            BINN_STRING => (*value).ptr as *const i8,
            _ => ptr::null(),
        }
    }
}

extern "C" fn binn_get_blob(value: *mut Binn, psize: *mut i32) -> *const c_void {
    if value.is_null() {
        return ptr::null();
    }

    unsafe {
        match (*value).type_ {
            BINN_BLOB => {
                if !psize.is_null() {
                    *psize = (*value).size;
                }
                (*value).ptr
            }
            _ => ptr::null(),
        }
    }
}

extern "C" fn binn_is_container(item: *mut Binn) -> bool {
    if item.is_null() {
        return false;
    }

    unsafe {
        match (*item).type_ {
            BINN_LIST | BINN_MAP | BINN_OBJECT => true,
            _ => false,
        }
    }
}

extern "C" fn binn_list_value(ptr: *mut c_void, pos: i32) -> *mut Binn {
    let value = Box::into_raw(Box::new(BinnStruct {
        header: 0x1F22B11F,
        allocated: true,
        writable: false,
        dirty: false,
        pbuf: ptr::null_mut(),
        pre_allocated: false,
        alloc_size: 0,
        used_size: 0,
        type_: 0,
        ptr: ptr::null_mut(),
        size: 0,
        count: 0,
        freefn: None,
        disable_int_compression: false,
        vint8: 0,
        vint16: 0,
        vint32: 0,
        vint64: 0,
        vuint8: 0,
        vuint16: 0,
        vuint32: 0,
        vuint64: 0,
        vchar: 0,
        vuchar: 0,
        vshort: 0,
        vushort: 0,
        vint: 0,
        vuint: 0,
        vfloat: 0.0,
        vdouble: 0.0,
        vbool: false,
    }));

    unsafe {
        if binn_list_get_value(ptr, pos, value) {
            value
        } else {
            Box::from_raw(value);
            ptr::null_mut()
        }
    }
}

extern "C" fn binn_map_value(ptr: *mut c_void, id: i32) -> *mut Binn {
    let value = Box::into_raw(Box::new(BinnStruct {
        header: 0x1F22B11F,
        allocated: true,
        writable: false,
        dirty: false,
        pbuf: ptr::null_mut(),
        pre_allocated: false,
        alloc_size: 0,
        used_size: 0,
        type_: 0,
        ptr: ptr::null_mut(),
        size: 0,
        count: 0,
        freefn: None,
        disable_int_compression: false,
        vint8: 0,
        vint16: 0,
        vint32: 0,
        vint64: 0,
        vuint8: 0,
        vuint16: 0,
        vuint32: 0,
        vuint64: 0,
        vchar: 0,
        vuchar: 0,
        vshort: 0,
        vushort: 0,
        vint: 0,
        vuint: 0,
        vfloat: 0.0,
        vdouble: 0.0,
        vbool: false,
    }));

    unsafe {
        if binn_map_get_value(ptr, id, value) {
            value
        } else {
            Box::from_raw(value);
            ptr::null_mut()
        }
    }
}

extern "C" fn binn_object_value(ptr: *mut c_void, key: *const i8) -> *mut Binn {
    let value = Box::into_raw(Box::new(BinnStruct {
        header: 0x1F22B11F,
        allocated: true,
        writable: false,
        dirty: false,
        pbuf: ptr::null_mut(),
        pre_allocated: false,
        alloc_size: 0,
        used_size: 0,
        type_: 0,
        ptr: ptr::null_mut(),
        size: 0,
        count: 0,
        freefn: None,
        disable_int_compression: false,
        vint8: 0,
        vint16: 0,
        vint32: 0,
        vint64: 0,
        vuint8: 0,
        vuint16: 0,
        vuint32: 0,
        vuint64: 0,
        vchar: 0,
        vuchar: 0,
        vshort: 0,
        vushort: 0,
        vint: 0,
        vuint: 0,
        vfloat: 0.0,
        vdouble: 0.0,
        vbool: false,
    }));

    unsafe {
        if binn_object_get_value(ptr, key, value) {
            value
        } else {
            Box::from_raw(value);
            ptr::null_mut()
        }
    }
}

extern "C" fn binn_list_read(list: *mut c_void, pos: i32, ptype: *mut i32, psize: *mut i32) -> *const c_void {
    let mut value = BinnStruct {
        header: 0x1F22B11F,
        allocated: false,
        writable: false,
        dirty: false,
        pbuf: ptr::null_mut(),
        pre_allocated: false,
        alloc_size: 0,
        used_size: 0,
        type_: 0,
        ptr: ptr::null_mut(),
        size: 0,
        count: 0,
        freefn: None,
        disable_int_compression: false,
        vint8: 0,
        vint16: 0,
        vint32: 0,
        vint64: 0,
        vuint8: 0,
        vuint16: 0,
        vuint32: 0,
        vuint64: 0,
        vchar: 0,
        vuchar: 0,
        vshort: 0,
        vushort: 0,
        vint: 0,
        vuint: 0,
        vfloat: 0.0,
        vdouble: 0.0,
        vbool: false,
    };

    unsafe {
        if binn_list_get_value(list, pos, &mut value) {
            if !ptype.is_null() {
                *ptype = value.type_;
            }
            if !psize.is_null() {
                *psize = value.size;
            }
            value.ptr
        } else {
            ptr::null()
        }
    }
}

extern "C" fn binn_map_read(map: *mut c_void, id: i32, ptype: *mut i32, psize: *mut i32) -> *const c_void {
    let mut value = BinnStruct {
        header: 0x1F22B11F,
        allocated: false,
        writable: false,
        dirty: false,
        pbuf: ptr::null_mut(),
        pre_allocated: false,
        alloc_size: 0,
        used_size: 0,
        type_: 0,
        ptr: ptr::null_mut(),
        size: 0,
        count: 0,
        freefn: None,
        disable_int_compression: false,
        vint8: 0,
        vint16: 0,
        vint32: 0,
        vint64: 0,
        vuint8: 0,
        vuint16: 0,
        vuint32: 0,
        vuint64: 0,
        vchar: 0,
        vuchar: 0,
        vshort: 0,
        vushort: 0,
        vint: 0,
        vuint: 0,
        vfloat: 0.0,
        vdouble: 0.0,
        vbool: false,
    };

    unsafe {
        if binn_map_get_value(map, id, &mut value) {
            if !ptype.is_null() {
                *ptype = value.type_;
            }
            if !psize.is_null() {
                *psize = value.size;
            }
            value.ptr
        } else {
            ptr::null()
        }
    }
}

extern "C" fn binn_object_read(obj: *mut c_void, key: *const i8, ptype: *mut i32, psize: *mut i32) -> *const c_void {
    let mut value = BinnStruct {
        header: 0x1F22B11F,
        allocated: false,
        writable: false,
        dirty: false,
        pbuf: ptr::null_mut(),
        pre_allocated: false,
        alloc_size: 0,
        used_size: 0,
        type_: 0,
        ptr: ptr::null_mut(),
        size: 0,
        count: 0,
        freefn: None,
        disable_int_compression: false,
        vint8: 0,
        vint16: 0,
        vint32: 0,
        vint64: 0,
        vuint8: 0,
        vuint16: 0,
        vuint32: 0,
        vuint64: 0,
        vchar: 0,
        vuchar: 0,
        vshort: 0,
        vushort: 0,
        vint: 0,
        vuint: 0,
        vfloat: 0.0,
        vdouble: 0.0,
        vbool: false,
    };

    unsafe {
        if binn_object_get_value(obj, key, &mut value) {
            if !ptype.is_null() {
                *ptype = value.type_;
            }
            if !psize.is_null() {
                *psize = value.size;
            }
            value.ptr
        } else {
            ptr::null()
        }
    }
}

extern "C" fn binn_list_get(ptr: *mut c_void, pos: i32, type_: i32, pvalue: *mut c_void, psize: *mut i32) -> bool {
    if ptr.is_null() || pvalue.is_null() {
        return false;
    }

    unsafe {
        let mut value = BinnStruct {
            header: 0x1F22B11F,
            allocated: false,
            writable: false,
            dirty: false,
            pbuf: ptr::null_mut(),
            pre_allocated: false,
            alloc_size: 0,
            used_size: 0,
            type_: 0,
            ptr: ptr::null_mut(),
            size: 0,
            count: 0,
            freefn: None,
            disable_int_compression: false,
            vint8: 0,
            vint16: 0,
            vint32: 0,
            vint64: 0,
            vuint8: 0,
            vuint16: 0,
            vuint32: 0,
            vuint64: 0,
            vchar: 0,
            vuchar: 0,
            vshort: 0,
            vushort: 0,
            vint: 0,
            vuint: 0,
            vfloat: 0.0,
            vdouble: 0.0,
            vbool: false,
        };

        if !binn_list_get_value(ptr, pos, &mut value) {
            return false;
        }

        if value.type_ != type_ {
            return false;
        }

        if !psize.is_null() {
            *psize = value.size;
        }

        match type_ {
            BINN_INT8 => *(pvalue as *mut i8) = value.vint8,
            BINN_INT16 => *(pvalue as *mut i16) = value.vint16,
            BINN_INT32 => *(pvalue as *mut i32) = value.vint32,
            BINN_INT64 => *(pvalue as *mut i64) = value.vint64,
            BINN_UINT8 => *(pvalue as *mut u8) = value.vuint8,
            BINN_UINT16 => *(pvalue as *mut u16) = value.vuint16,
            BINN_UINT32 => *(pvalue as *mut u32) = value.vuint32,
            BINN_UINT64 => *(pvalue as *mut u64) = value.vuint64,
            BINN_FLOAT32 => *(pvalue as *mut f32) = value.vfloat,
            BINN_FLOAT64 => *(pvalue as *mut f64) = value.vdouble,
            BINN_BOOL => *(pvalue as *mut bool) = value.vbool,
            BINN_STRING => *(pvalue as *mut *const i8) = value.ptr as *const i8,
            BINN_BLOB => *(pvalue as *mut *const c_void) = value.ptr,
            _ => return false,
        }

        true
    }
}

extern "C" fn binn_map_get(ptr: *mut c_void, id: i32, type_: i32, pvalue: *mut c_void, psize: *mut i32) -> bool {
    if ptr.is_null() || pvalue.is_null() {
        return false;
    }

    unsafe {
        let mut value = BinnStruct {
            header: 0x1F22B11F,
            allocated: false,
            writable: false,
            dirty: false,
            pbuf: ptr::null_mut(),
            pre_allocated: false,
            alloc_size: 0,
            used_size: 0,
            type_: 0,
            ptr: ptr::null_mut(),
            size: 0,
            count: 0,
            freefn: None,
            disable_int_compression: false,
            vint8: 0,
            vint16: 0,
            vint32: 0,
            vint64: 0,
            vuint8: 0,
            vuint16: 0,
            vuint32: 0,
            vuint64: 0,
            vchar: 0,
            vuchar: 0,
            vshort: 0,
            vushort: 0,
            vint: 0,
            vuint: 0,
            vfloat: 0.0,
            vdouble: 0.0,
            vbool: false,
        };

        if !binn_map_get_value(ptr, id, &mut value) {
            return false;
        }

        if value.type_ != type_ {
            return false;
        }

        if !psize.is_null() {
            *psize = value.size;
        }

        match type_ {
            BINN_INT8 => *(pvalue as *mut i8) = value.vint8,
            BINN_INT16 => *(pvalue as *mut i16) = value.vint16,
            BINN_INT32 => *(pvalue as *mut i32) = value.vint32,
            BINN_INT64 => *(pvalue as *mut i64) = value.vint64,
            BINN_UINT8 => *(pvalue as *mut u8) = value.vuint8,
            BINN_UINT16 => *(pvalue as *mut u16) = value.vuint16,
            BINN_UINT32 => *(pvalue as *mut u32) = value.vuint32,
            BINN_UINT64 => *(pvalue as *mut u64) = value.vuint64,
            BINN_FLOAT32 => *(pvalue as *mut f32) = value.vfloat,
            BINN_FLOAT64 => *(pvalue as *mut f64) = value.vdouble,
            BINN_BOOL => *(pvalue as *mut bool) = value.vbool,
            BINN_STRING => *(pvalue as *mut *const i8) = value.ptr as *const i8,
            BINN_BLOB => *(pvalue as *mut *const c_void) = value.ptr,
            _ => return false,
        }

        true
    }
}

extern "C" fn binn_object_get(ptr: *mut c_void, key: *const i8, type_: i32, pvalue: *mut c_void, psize: *mut i32) -> bool {
    if ptr.is_null() || key.is_null() || pvalue.is_null() {
        return false;
    }

    unsafe {
        let mut value = BinnStruct {
            header: 0x1F22B11F,
            allocated: false,
            writable: false,
            dirty: false,
            pbuf: ptr::null_mut(),
            pre_allocated: false,
            alloc_size: 0,
            used_size: 0,
            type_: 0,
            ptr: ptr::null_mut(),
            size: 0,
            count: 0,
            freefn: None,
            disable_int_compression: false,
            vint8: 0,
            vint16: 0,
            vint32: 0,
            vint64: 0,
            vuint8: 0,
            vuint16: 0,
            vuint32: 0,
            vuint64: 0,
            vchar: 0,
            vuchar: 0,
            vshort: 0,
            vushort: 0,
            vint: 0,
            vuint: 0,
            vfloat: 0.0,
            vdouble: 0.0,
            vbool: false,
        };

        if !binn_object_get_value(ptr, key, &mut value) {
            return false;
        }

        if value.type_ != type_ {
            return false;
        }

        if !psize.is_null() {
            *psize = value.size;
        }

        match type_ {
            BINN_INT8 => *(pvalue as *mut i8) = value.vint8,
            BINN_INT16 => *(pvalue as *mut i16) = value.vint16,
            BINN_INT32 => *(pvalue as *mut i32) = value.vint32,
            BINN_INT64 => *(pvalue as *mut i64) = value.vint64,
            BINN_UINT8 => *(pvalue as *mut u8) = value.vuint8,
            BINN_UINT16 => *(pvalue as *mut u16) = value.vuint16,
            BINN_UINT32 => *(pvalue as *mut u32) = value.vuint32,
            BINN_UINT64 => *(pvalue as *mut u64) = value.vuint64,
            BINN_FLOAT32 => *(pvalue as *mut f32) = value.vfloat,
            BINN_FLOAT64 => *(pvalue as *mut f64) = value.vdouble,
            BINN_BOOL => *(pvalue as *mut bool) = value.vbool,
            BINN_STRING => *(pvalue as *mut *const i8) = value.ptr as *const i8,
            BINN_BLOB => *(pvalue as *mut *const c_void) = value.ptr,
            _ => return false,
        }

        true
    }
}

extern "C" fn binn_list_int8(list: *mut c_void, pos: i32) -> i8 {
    let mut value: i8 = 0;
    unsafe {
        binn_list_get(list, pos, BINN_INT8 as i32, &mut value as *mut i8 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_list_int16(list: *mut c_void, pos: i32) -> i16 {
    let mut value: i16 = 0;
    unsafe {
        binn_list_get(list, pos, BINN_INT16 as i32, &mut value as *mut i16 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_list_int32(list: *mut c_void, pos: i32) -> i32 {
    let mut value: i32 = 0;
    unsafe {
        binn_list_get(list, pos, BINN_INT32 as i32, &mut value as *mut i32 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_list_int64(list: *mut c_void, pos: i32) -> i64 {
    let mut value: i64 = 0;
    unsafe {
        binn_list_get(list, pos, BINN_INT64 as i32, &mut value as *mut i64 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_list_uint8(list: *mut c_void, pos: i32) -> u8 {
    let mut value: u8 = 0;
    unsafe {
        binn_list_get(list, pos, BINN_UINT8 as i32, &mut value as *mut u8 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_list_uint16(list: *mut c_void, pos: i32) -> u16 {
    let mut value: u16 = 0;
    unsafe {
        binn_list_get(list, pos, BINN_UINT16 as i32, &mut value as *mut u16 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_list_uint32(list: *mut c_void, pos: i32) -> u32 {
    let mut value: u32 = 0;
    unsafe {
        binn_list_get(list, pos, BINN_UINT32 as i32, &mut value as *mut u32 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_list_uint64(list: *mut c_void, pos: i32) -> u64 {
    let mut value: u64 = 0;
    unsafe {
        binn_list_get(list, pos, BINN_UINT64 as i32, &mut value as *mut u64 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_list_float(list: *mut c_void, pos: i32) -> f32 {
    let mut value: f32 = 0.0;
    unsafe {
        binn_list_get(list, pos, BINN_FLOAT32 as i32, &mut value as *mut f32 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_list_double(list: *mut c_void, pos: i32) -> f64 {
    let mut value: f64 = 0.0;
    unsafe {
        binn_list_get(list, pos, BINN_FLOAT64 as i32, &mut value as *mut f64 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_list_bool(list: *mut c_void, pos: i32) -> bool {
    let mut value: bool = false;
    unsafe {
        binn_list_get(list, pos, BINN_BOOL as i32, &mut value as *mut bool as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_list_null(list: *mut c_void, pos: i32) -> bool {
    unsafe {
        binn_list_get(list, pos, BINN_NULL as i32, ptr::null_mut(), ptr::null_mut())
    }
}

extern "C" fn binn_list_str(list: *mut c_void, pos: i32) -> *const i8 {
    let mut value: *const i8 = ptr::null();
    unsafe {
        binn_list_get(list, pos, BINN_STRING as i32, &mut value as *mut *const i8 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_list_blob(list: *mut c_void, pos: i32, psize: *mut i32) -> *const c_void {
    let mut value: *const c_void = ptr::null();
    unsafe {
        binn_list_get(list, pos, BINN_BLOB as i32, &mut value as *mut *const c_void as *mut c_void, psize);
    }
    value
}

extern "C" fn binn_list_list(list: *mut c_void, pos: i32) -> *const c_void {
    let mut value: *const c_void = ptr::null();
    unsafe {
        binn_list_get(list, pos, BINN_LIST as i32, &mut value as *mut *const c_void as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_list_map(list: *mut c_void, pos: i32) -> *const c_void {
    let mut value: *const c_void = ptr::null();
    unsafe {
        binn_list_get(list, pos, BINN_MAP as i32, &mut value as *mut *const c_void as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_list_object(list: *mut c_void, pos: i32) -> *const c_void {
    let mut value: *const c_void = ptr::null();
    unsafe {
        binn_list_get(list, pos, BINN_OBJECT as i32, &mut value as *mut *const c_void as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_int8(map: *mut c_void, id: i32) -> i8 {
    let mut value: i8 = 0;
    unsafe {
        binn_map_get(map, id, BINN_INT8 as i32, &mut value as *mut i8 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_int16(map: *mut c_void, id: i32) -> i16 {
    let mut value: i16 = 0;
    unsafe {
        binn_map_get(map, id, BINN_INT16 as i32, &mut value as *mut i16 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_int32(map: *mut c_void, id: i32) -> i32 {
    let mut value: i32 = 0;
    unsafe {
        binn_map_get(map, id, BINN_INT32 as i32, &mut value as *mut i32 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_int64(map: *mut c_void, id: i32) -> i64 {
    let mut value: i64 = 0;
    unsafe {
        binn_map_get(map, id, BINN_INT64 as i32, &mut value as *mut i64 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_uint8(map: *mut c_void, id: i32) -> u8 {
    let mut value: u8 = 0;
    unsafe {
        binn_map_get(map, id, BINN_UINT8 as i32, &mut value as *mut u8 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_uint16(map: *mut c_void, id: i32) -> u16 {
    let mut value: u16 = 0;
    unsafe {
        binn_map_get(map, id, BINN_UINT16 as i32, &mut value as *mut u16 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_uint32(map: *mut c_void, id: i32) -> u32 {
    let mut value: u32 = 0;
    unsafe {
        binn_map_get(map, id, BINN_UINT32 as i32, &mut value as *mut u32 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_uint64(map: *mut c_void, id: i32) -> u64 {
    let mut value: u64 = 0;
    unsafe {
        binn_map_get(map, id, BINN_UINT64 as i32, &mut value as *mut u64 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_float(map: *mut c_void, id: i32) -> f32 {
    let mut value: f32 = 0.0;
    unsafe {
        binn_map_get(map, id, BINN_FLOAT32 as i32, &mut value as *mut f32 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_double(map: *mut c_void, id: i32) -> f64 {
    let mut value: f64 = 0.0;
    unsafe {
        binn_map_get(map, id, BINN_FLOAT64 as i32, &mut value as *mut f64 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_bool(map: *mut c_void, id: i32) -> bool {
    let mut value: bool = false;
    unsafe {
        binn_map_get(map, id, BINN_BOOL as i32, &mut value as *mut bool as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_null(map: *mut c_void, id: i32) -> bool {
    unsafe {
        binn_map_get(map, id, BINN_NULL as i32, ptr::null_mut(), ptr::null_mut())
    }
}

extern "C" fn binn_map_str(map: *mut c_void, id: i32) -> *const i8 {
    let mut value: *const i8 = ptr::null();
    unsafe {
        binn_map_get(map, id, BINN_STRING as i32, &mut value as *mut *const i8 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_blob(map: *mut c_void, id: i32, psize: *mut i32) -> *const c_void {
    let mut value: *const c_void = ptr::null();
    unsafe {
        binn_map_get(map, id, BINN_BLOB as i32, &mut value as *mut *const c_void as *mut c_void, psize);
    }
    value
}

extern "C" fn binn_map_list(map: *mut c_void, id: i32) -> *const c_void {
    let mut value: *const c_void = ptr::null();
    unsafe {
        binn_map_get(map, id, BINN_LIST as i32, &mut value as *mut *const c_void as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_map(map: *mut c_void, id: i32) -> *const c_void {
    let mut value: *const c_void = ptr::null();
    unsafe {
        binn_map_get(map, id, BINN_MAP as i32, &mut value as *mut *const c_void as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_map_object(map: *mut c_void, id: i32) -> *const c_void {
    let mut value: *const c_void = ptr::null();
    unsafe {
        binn_map_get(map, id, BINN_OBJECT as i32, &mut value as *mut *const c_void as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_object_int8(obj: *mut c_void, key: *const i8) -> i8 {
    let mut value: i8 = 0;
    unsafe {
        binn_object_get(obj, key, BINN_INT8 as i32, &mut value as *mut i8 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_object_int16(obj: *mut c_void, key: *const i8) -> i16 {
    let mut value: i16 = 0;
    unsafe {
        binn_object_get(obj, key, BINN_INT16 as i32, &mut value as *mut i16 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_object_int32(obj: *mut c_void, key: *const i8) -> i32 {
    let mut value: i32 = 0;
    unsafe {
        binn_object_get(obj, key, BINN_INT32 as i32, &mut value as *mut i32 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_object_int64(obj: *mut c_void, key: *const i8) -> i64 {
    let mut value: i64 = 0;
    unsafe {
        binn_object_get(obj, key, BINN_INT64 as i32, &mut value as *mut i64 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_object_uint8(obj: *mut c_void, key: *const i8) -> u8 {
    let mut value: u8 = 0;
    unsafe {
        binn_object_get(obj, key, BINN_UINT8 as i32, &mut value as *mut u8 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_object_uint16(obj: *mut c_void, key: *const i8) -> u16 {
    let mut value: u16 = 0;
    unsafe {
        binn_object_get(obj, key, BINN_UINT16 as i32, &mut value as *mut u16 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_object_uint32(obj: *mut c_void, key: *const i8) -> u32 {
    let mut value: u32 = 0;
    unsafe {
        binn_object_get(obj, key, BINN_UINT32 as i32, &mut value as *mut u32 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_object_uint64(obj: *mut c_void, key: *const i8) -> u64 {
    let mut value: u64 = 0;
    unsafe {
        binn_object_get(obj, key, BINN_UINT64 as i32, &mut value as *mut u64 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_object_float(obj: *mut c_void, key: *const i8) -> f32 {
    let mut value: f32 = 0.0;
    unsafe {
        binn_object_get(obj, key, BINN_FLOAT32 as i32, &mut value as *mut f32 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_object_double(obj: *mut c_void, key: *const i8) -> f64 {
    let mut value: f64 = 0.0;
    unsafe {
        binn_object_get(obj, key, BINN_FLOAT64 as i32, &mut value as *mut f64 as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_object_bool(obj: *mut c_void, key: *const i8) -> bool {
    let mut value: bool = false;
    unsafe {
        binn_object_get(obj, key, BINN_BOOL as i32, &mut value as *mut bool as *mut c_void, ptr::null_mut());
    }
    value
}

extern "C" fn binn_object_null(obj: *mut c_void, key: *const i8) -> bool {
    unsafe {
        binn_object_get(obj, key, BINN_NULL as i32, ptr::null_mut(), ptr::null_mut())
    }
}

extern "C" fn binn_object_str(obj: *mut c_void, key: *const i8) -> *const i8 {
    let mut value: *const i8 = ptr::null();
    unsafe {
        binn_object_get(obj, key, BINN_STRING as i32, &mut value as *mut *const i8 as *mut c_void, ptr::null_mut());
    }
    value
}