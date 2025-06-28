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
pub type ArrayListValue = *mut libc::c_void;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _ArrayList {
    pub data: *mut ArrayListValue,
    pub length: libc::c_uint,
    pub _alloced: libc::c_uint,
}
pub type ArrayList = _ArrayList;
pub type ArrayListEqualFunc = Option::<
    unsafe extern "C" fn(ArrayListValue, ArrayListValue) -> libc::c_int,
>;
pub type ArrayListCompareFunc = Option::<
    unsafe extern "C" fn(ArrayListValue, ArrayListValue) -> libc::c_int,
>;
#[no_mangle]
pub unsafe extern "C" fn arraylist_new(mut length: libc::c_uint) -> *mut ArrayList {
    let mut new_arraylist: *mut ArrayList = 0 as *mut ArrayList;
    if length <= 0 as libc::c_int as libc::c_uint {
        length = 16 as libc::c_int as libc::c_uint;
    }
    new_arraylist = alloc_test_malloc(
        ::core::mem::size_of::<ArrayList>() as libc::c_ulong,
    ) as *mut ArrayList;
    if new_arraylist.is_null() {
        return 0 as *mut ArrayList;
    }
    (*new_arraylist)._alloced = length;
    (*new_arraylist).length = 0 as libc::c_int as libc::c_uint;
    (*new_arraylist)
        .data = alloc_test_malloc(
        (length as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<ArrayListValue>() as libc::c_ulong),
    ) as *mut ArrayListValue;
    if ((*new_arraylist).data).is_null() {
        alloc_test_free(new_arraylist as *mut libc::c_void);
        return 0 as *mut ArrayList;
    }
    return new_arraylist;
}
#[no_mangle]
pub unsafe extern "C" fn arraylist_free(mut arraylist: *mut ArrayList) {
    if !arraylist.is_null() {
        alloc_test_free((*arraylist).data as *mut libc::c_void);
        alloc_test_free(arraylist as *mut libc::c_void);
    }
}
unsafe extern "C" fn arraylist_enlarge(mut arraylist: *mut ArrayList) -> libc::c_int {
    let mut data: *mut ArrayListValue = 0 as *mut ArrayListValue;
    let mut newsize: libc::c_uint = 0;
    newsize = ((*arraylist)._alloced).wrapping_mul(2 as libc::c_int as libc::c_uint);
    data = alloc_test_realloc(
        (*arraylist).data as *mut libc::c_void,
        (::core::mem::size_of::<ArrayListValue>() as libc::c_ulong)
            .wrapping_mul(newsize as libc::c_ulong),
    ) as *mut ArrayListValue;
    if data.is_null() {
        return 0 as libc::c_int
    } else {
        (*arraylist).data = data;
        (*arraylist)._alloced = newsize;
        return 1 as libc::c_int;
    };
}
#[no_mangle]
pub unsafe extern "C" fn arraylist_insert(
    mut arraylist: *mut ArrayList,
    mut index: libc::c_uint,
    mut data: ArrayListValue,
) -> libc::c_int {
    if index > (*arraylist).length {
        return 0 as libc::c_int;
    }
    if ((*arraylist).length).wrapping_add(1 as libc::c_int as libc::c_uint)
        > (*arraylist)._alloced
    {
        if arraylist_enlarge(arraylist) == 0 {
            return 0 as libc::c_int;
        }
    }
    memmove(
        &mut *((*arraylist).data)
            .offset(index.wrapping_add(1 as libc::c_int as libc::c_uint) as isize)
            as *mut ArrayListValue as *mut libc::c_void,
        &mut *((*arraylist).data).offset(index as isize) as *mut ArrayListValue
            as *const libc::c_void,
        (((*arraylist).length).wrapping_sub(index) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<ArrayListValue>() as libc::c_ulong),
    );
    let ref mut fresh0 = *((*arraylist).data).offset(index as isize);
    *fresh0 = data;
    (*arraylist).length = ((*arraylist).length).wrapping_add(1);
    (*arraylist).length;
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn arraylist_append(
    mut arraylist: *mut ArrayList,
    mut data: ArrayListValue,
) -> libc::c_int {
    return arraylist_insert(arraylist, (*arraylist).length, data);
}
#[no_mangle]
pub unsafe extern "C" fn arraylist_prepend(
    mut arraylist: *mut ArrayList,
    mut data: ArrayListValue,
) -> libc::c_int {
    return arraylist_insert(arraylist, 0 as libc::c_int as libc::c_uint, data);
}
#[no_mangle]
pub unsafe extern "C" fn arraylist_remove_range(
    mut arraylist: *mut ArrayList,
    mut index: libc::c_uint,
    mut length: libc::c_uint,
) {
    if index > (*arraylist).length || index.wrapping_add(length) > (*arraylist).length {
        return;
    }
    memmove(
        &mut *((*arraylist).data).offset(index as isize) as *mut ArrayListValue
            as *mut libc::c_void,
        &mut *((*arraylist).data).offset(index.wrapping_add(length) as isize)
            as *mut ArrayListValue as *const libc::c_void,
        (((*arraylist).length).wrapping_sub(index.wrapping_add(length)) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<ArrayListValue>() as libc::c_ulong),
    );
    (*arraylist).length = ((*arraylist).length).wrapping_sub(length);
}
#[no_mangle]
pub unsafe extern "C" fn arraylist_remove(
    mut arraylist: *mut ArrayList,
    mut index: libc::c_uint,
) {
    arraylist_remove_range(arraylist, index, 1 as libc::c_int as libc::c_uint);
}
#[no_mangle]
pub unsafe extern "C" fn arraylist_index_of(
    mut arraylist: *mut ArrayList,
    mut callback: ArrayListEqualFunc,
    mut data: ArrayListValue,
) -> libc::c_int {
    let mut i: libc::c_uint = 0;
    i = 0 as libc::c_int as libc::c_uint;
    while i < (*arraylist).length {
        if callback
            .expect(
                "non-null function pointer",
            )(*((*arraylist).data).offset(i as isize), data) != 0 as libc::c_int
        {
            return i as libc::c_int;
        }
        i = i.wrapping_add(1);
        i;
    }
    return -(1 as libc::c_int);
}
#[no_mangle]
pub unsafe extern "C" fn arraylist_clear(mut arraylist: *mut ArrayList) {
    (*arraylist).length = 0 as libc::c_int as libc::c_uint;
}
unsafe extern "C" fn arraylist_sort_internal(
    mut list_data: *mut ArrayListValue,
    mut list_length: libc::c_uint,
    mut compare_func: ArrayListCompareFunc,
) {
    let mut pivot: ArrayListValue = 0 as *mut libc::c_void;
    let mut tmp: ArrayListValue = 0 as *mut libc::c_void;
    let mut i: libc::c_uint = 0;
    let mut list1_length: libc::c_uint = 0;
    let mut list2_length: libc::c_uint = 0;
    if list_length <= 1 as libc::c_int as libc::c_uint {
        return;
    }
    pivot = *list_data
        .offset(list_length.wrapping_sub(1 as libc::c_int as libc::c_uint) as isize);
    list1_length = 0 as libc::c_int as libc::c_uint;
    i = 0 as libc::c_int as libc::c_uint;
    while i < list_length.wrapping_sub(1 as libc::c_int as libc::c_uint) {
        if compare_func
            .expect("non-null function pointer")(*list_data.offset(i as isize), pivot)
            < 0 as libc::c_int
        {
            tmp = *list_data.offset(i as isize);
            let ref mut fresh1 = *list_data.offset(i as isize);
            *fresh1 = *list_data.offset(list1_length as isize);
            let ref mut fresh2 = *list_data.offset(list1_length as isize);
            *fresh2 = tmp;
            list1_length = list1_length.wrapping_add(1);
            list1_length;
        }
        i = i.wrapping_add(1);
        i;
    }
    list2_length = list_length
        .wrapping_sub(list1_length)
        .wrapping_sub(1 as libc::c_int as libc::c_uint);
    let ref mut fresh3 = *list_data
        .offset(list_length.wrapping_sub(1 as libc::c_int as libc::c_uint) as isize);
    *fresh3 = *list_data.offset(list1_length as isize);
    let ref mut fresh4 = *list_data.offset(list1_length as isize);
    *fresh4 = pivot;
    arraylist_sort_internal(list_data, list1_length, compare_func);
    arraylist_sort_internal(
        &mut *list_data
            .offset(
                list1_length.wrapping_add(1 as libc::c_int as libc::c_uint) as isize,
            ),
        list2_length,
        compare_func,
    );
}
#[no_mangle]
pub unsafe extern "C" fn arraylist_sort(
    mut arraylist: *mut ArrayList,
    mut compare_func: ArrayListCompareFunc,
) {
    arraylist_sort_internal((*arraylist).data, (*arraylist).length, compare_func);
}
