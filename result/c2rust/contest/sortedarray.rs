use ::libc;
extern "C" {
    fn memmove(
        _: *mut libc::c_void,
        _: *const libc::c_void,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
    fn alloc_test_malloc(bytes: size_t) -> *mut libc::c_void;
    fn alloc_test_free(ptr: *mut libc::c_void);
    fn alloc_test_realloc(ptr: *mut libc::c_void, bytes: size_t) -> *mut libc::c_void;
}
pub type size_t = libc::c_ulong;
pub type SortedArrayValue = *mut libc::c_void;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _SortedArray {
    pub data: *mut SortedArrayValue,
    pub length: libc::c_uint,
    pub _alloced: libc::c_uint,
    pub equ_func: SortedArrayEqualFunc,
    pub cmp_func: SortedArrayCompareFunc,
}
pub type SortedArrayCompareFunc = Option::<
    unsafe extern "C" fn(SortedArrayValue, SortedArrayValue) -> libc::c_int,
>;
pub type SortedArrayEqualFunc = Option::<
    unsafe extern "C" fn(SortedArrayValue, SortedArrayValue) -> libc::c_int,
>;
pub type SortedArray = _SortedArray;
unsafe extern "C" fn sortedarray_first_index(
    mut sortedarray: *mut SortedArray,
    mut data: SortedArrayValue,
    mut left: libc::c_uint,
    mut right: libc::c_uint,
) -> libc::c_uint {
    let mut index: libc::c_uint = left;
    while left < right {
        index = left.wrapping_add(right).wrapping_div(2 as libc::c_int as libc::c_uint);
        let mut order: libc::c_int = ((*sortedarray).cmp_func)
            .expect(
                "non-null function pointer",
            )(data, *((*sortedarray).data).offset(index as isize));
        if order > 0 as libc::c_int {
            left = index.wrapping_add(1 as libc::c_int as libc::c_uint);
        } else {
            right = index;
        }
    }
    return index;
}
unsafe extern "C" fn sortedarray_last_index(
    mut sortedarray: *mut SortedArray,
    mut data: SortedArrayValue,
    mut left: libc::c_uint,
    mut right: libc::c_uint,
) -> libc::c_uint {
    let mut index: libc::c_uint = right;
    while left < right {
        index = left.wrapping_add(right).wrapping_div(2 as libc::c_int as libc::c_uint);
        let mut order: libc::c_int = ((*sortedarray).cmp_func)
            .expect(
                "non-null function pointer",
            )(data, *((*sortedarray).data).offset(index as isize));
        if order <= 0 as libc::c_int {
            left = index.wrapping_add(1 as libc::c_int as libc::c_uint);
        } else {
            right = index;
        }
    }
    return index;
}
#[no_mangle]
pub unsafe extern "C" fn sortedarray_get(
    mut array: *mut SortedArray,
    mut i: libc::c_uint,
) -> *mut SortedArrayValue {
    if array.is_null() {
        return 0 as *mut SortedArrayValue;
    }
    return *((*array).data).offset(i as isize) as *mut SortedArrayValue;
}
#[no_mangle]
pub unsafe extern "C" fn sortedarray_length(
    mut array: *mut SortedArray,
) -> libc::c_uint {
    return (*array).length;
}
#[no_mangle]
pub unsafe extern "C" fn sortedarray_new(
    mut length: libc::c_uint,
    mut equ_func: SortedArrayEqualFunc,
    mut cmp_func: SortedArrayCompareFunc,
) -> *mut SortedArray {
    if equ_func.is_none() || cmp_func.is_none() {
        return 0 as *mut SortedArray;
    }
    if length == 0 as libc::c_int as libc::c_uint {
        length = 16 as libc::c_int as libc::c_uint;
    }
    let mut array: *mut SortedArrayValue = alloc_test_malloc(
        (::core::mem::size_of::<SortedArrayValue>() as libc::c_ulong)
            .wrapping_mul(length as libc::c_ulong),
    ) as *mut SortedArrayValue;
    if array.is_null() {
        return 0 as *mut SortedArray;
    }
    let mut sortedarray: *mut SortedArray = alloc_test_malloc(
        ::core::mem::size_of::<SortedArray>() as libc::c_ulong,
    ) as *mut SortedArray;
    if sortedarray.is_null() {
        alloc_test_free(array as *mut libc::c_void);
        return 0 as *mut SortedArray;
    }
    (*sortedarray).data = array;
    (*sortedarray).length = 0 as libc::c_int as libc::c_uint;
    (*sortedarray)._alloced = length;
    (*sortedarray).equ_func = equ_func;
    (*sortedarray).cmp_func = cmp_func;
    return sortedarray;
}
#[no_mangle]
pub unsafe extern "C" fn sortedarray_free(mut sortedarray: *mut SortedArray) {
    if !sortedarray.is_null() {
        alloc_test_free((*sortedarray).data as *mut libc::c_void);
        alloc_test_free(sortedarray as *mut libc::c_void);
    }
}
#[no_mangle]
pub unsafe extern "C" fn sortedarray_remove(
    mut sortedarray: *mut SortedArray,
    mut index: libc::c_uint,
) {
    sortedarray_remove_range(sortedarray, index, 1 as libc::c_int as libc::c_uint);
}
#[no_mangle]
pub unsafe extern "C" fn sortedarray_remove_range(
    mut sortedarray: *mut SortedArray,
    mut index: libc::c_uint,
    mut length: libc::c_uint,
) {
    if index > (*sortedarray).length
        || index.wrapping_add(length) > (*sortedarray).length
    {
        return;
    }
    memmove(
        &mut *((*sortedarray).data).offset(index as isize) as *mut SortedArrayValue
            as *mut libc::c_void,
        &mut *((*sortedarray).data).offset(index.wrapping_add(length) as isize)
            as *mut SortedArrayValue as *const libc::c_void,
        (((*sortedarray).length).wrapping_sub(index.wrapping_add(length))
            as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<SortedArrayValue>() as libc::c_ulong),
    );
    (*sortedarray).length = ((*sortedarray).length).wrapping_sub(length);
}
#[no_mangle]
pub unsafe extern "C" fn sortedarray_insert(
    mut sortedarray: *mut SortedArray,
    mut data: SortedArrayValue,
) -> libc::c_int {
    let mut left: libc::c_uint = 0 as libc::c_int as libc::c_uint;
    let mut right: libc::c_uint = (*sortedarray).length;
    let mut index: libc::c_uint = 0 as libc::c_int as libc::c_uint;
    right = if right > 1 as libc::c_int as libc::c_uint {
        right
    } else {
        0 as libc::c_int as libc::c_uint
    };
    while left != right {
        index = left.wrapping_add(right).wrapping_div(2 as libc::c_int as libc::c_uint);
        let mut order: libc::c_int = ((*sortedarray).cmp_func)
            .expect(
                "non-null function pointer",
            )(data, *((*sortedarray).data).offset(index as isize));
        if order < 0 as libc::c_int {
            right = index;
        } else {
            if !(order > 0 as libc::c_int) {
                break;
            }
            left = index.wrapping_add(1 as libc::c_int as libc::c_uint);
        }
    }
    if (*sortedarray).length > 0 as libc::c_int as libc::c_uint
        && ((*sortedarray).cmp_func)
            .expect(
                "non-null function pointer",
            )(data, *((*sortedarray).data).offset(index as isize)) > 0 as libc::c_int
    {
        index = index.wrapping_add(1);
        index;
    }
    if ((*sortedarray).length).wrapping_add(1 as libc::c_int as libc::c_uint)
        > (*sortedarray)._alloced
    {
        let mut newsize: libc::c_uint = 0;
        let mut data_0: *mut SortedArrayValue = 0 as *mut SortedArrayValue;
        newsize = ((*sortedarray)._alloced)
            .wrapping_mul(2 as libc::c_int as libc::c_uint);
        data_0 = alloc_test_realloc(
            (*sortedarray).data as *mut libc::c_void,
            (::core::mem::size_of::<SortedArrayValue>() as libc::c_ulong)
                .wrapping_mul(newsize as libc::c_ulong),
        ) as *mut SortedArrayValue;
        if data_0.is_null() {
            return 0 as libc::c_int
        } else {
            (*sortedarray).data = data_0;
            (*sortedarray)._alloced = newsize;
        }
    }
    memmove(
        &mut *((*sortedarray).data)
            .offset(index.wrapping_add(1 as libc::c_int as libc::c_uint) as isize)
            as *mut SortedArrayValue as *mut libc::c_void,
        &mut *((*sortedarray).data).offset(index as isize) as *mut SortedArrayValue
            as *const libc::c_void,
        (((*sortedarray).length).wrapping_sub(index) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<SortedArrayValue>() as libc::c_ulong),
    );
    let ref mut fresh0 = *((*sortedarray).data).offset(index as isize);
    *fresh0 = data;
    (*sortedarray).length = ((*sortedarray).length).wrapping_add(1);
    (*sortedarray).length;
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn sortedarray_index_of(
    mut sortedarray: *mut SortedArray,
    mut data: SortedArrayValue,
) -> libc::c_int {
    if sortedarray.is_null() {
        return -(1 as libc::c_int);
    }
    let mut left: libc::c_uint = 0 as libc::c_int as libc::c_uint;
    let mut right: libc::c_uint = (*sortedarray).length;
    let mut index: libc::c_uint = 0 as libc::c_int as libc::c_uint;
    right = if right > 1 as libc::c_int as libc::c_uint {
        right
    } else {
        0 as libc::c_int as libc::c_uint
    };
    while left != right {
        index = left.wrapping_add(right).wrapping_div(2 as libc::c_int as libc::c_uint);
        let mut order: libc::c_int = ((*sortedarray).cmp_func)
            .expect(
                "non-null function pointer",
            )(data, *((*sortedarray).data).offset(index as isize));
        if order < 0 as libc::c_int {
            right = index;
        } else if order > 0 as libc::c_int {
            left = index.wrapping_add(1 as libc::c_int as libc::c_uint);
        } else {
            left = sortedarray_first_index(sortedarray, data, left, index);
            right = sortedarray_last_index(sortedarray, data, index, right);
            index = left;
            while index <= right {
                if ((*sortedarray).equ_func)
                    .expect(
                        "non-null function pointer",
                    )(data, *((*sortedarray).data).offset(index as isize)) != 0
                {
                    return index as libc::c_int;
                }
                index = index.wrapping_add(1);
                index;
            }
            return -(1 as libc::c_int);
        }
    }
    return -(1 as libc::c_int);
}
#[no_mangle]
pub unsafe extern "C" fn sortedarray_clear(mut sortedarray: *mut SortedArray) {
    (*sortedarray).length = 0 as libc::c_int as libc::c_uint;
}
