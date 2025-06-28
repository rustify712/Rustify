use ::libc;
extern "C" {
    fn alloc_test_malloc(bytes: size_t) -> *mut libc::c_void;
    fn alloc_test_free(ptr: *mut libc::c_void);
}
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _ListEntry {
    pub data: ListValue,
    pub prev: *mut ListEntry,
    pub next: *mut ListEntry,
}
pub type ListEntry = _ListEntry;
pub type ListValue = *mut libc::c_void;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _ListIterator {
    pub prev_next: *mut *mut ListEntry,
    pub current: *mut ListEntry,
}
pub type ListIterator = _ListIterator;
pub type ListCompareFunc = Option::<
    unsafe extern "C" fn(ListValue, ListValue) -> libc::c_int,
>;
pub type ListEqualFunc = Option::<
    unsafe extern "C" fn(ListValue, ListValue) -> libc::c_int,
>;
#[no_mangle]
pub unsafe extern "C" fn list_free(mut list: *mut ListEntry) {
    let mut entry: *mut ListEntry = 0 as *mut ListEntry;
    entry = list;
    while !entry.is_null() {
        let mut next: *mut ListEntry = 0 as *mut ListEntry;
        next = (*entry).next;
        alloc_test_free(entry as *mut libc::c_void);
        entry = next;
    }
}
#[no_mangle]
pub unsafe extern "C" fn list_prepend(
    mut list: *mut *mut ListEntry,
    mut data: ListValue,
) -> *mut ListEntry {
    let mut newentry: *mut ListEntry = 0 as *mut ListEntry;
    if list.is_null() {
        return 0 as *mut ListEntry;
    }
    newentry = alloc_test_malloc(::core::mem::size_of::<ListEntry>() as libc::c_ulong)
        as *mut ListEntry;
    if newentry.is_null() {
        return 0 as *mut ListEntry;
    }
    (*newentry).data = data;
    if !(*list).is_null() {
        (**list).prev = newentry;
    }
    (*newentry).prev = 0 as *mut ListEntry;
    (*newentry).next = *list;
    *list = newentry;
    return newentry;
}
#[no_mangle]
pub unsafe extern "C" fn list_append(
    mut list: *mut *mut ListEntry,
    mut data: ListValue,
) -> *mut ListEntry {
    let mut rover: *mut ListEntry = 0 as *mut ListEntry;
    let mut newentry: *mut ListEntry = 0 as *mut ListEntry;
    if list.is_null() {
        return 0 as *mut ListEntry;
    }
    newentry = alloc_test_malloc(::core::mem::size_of::<ListEntry>() as libc::c_ulong)
        as *mut ListEntry;
    if newentry.is_null() {
        return 0 as *mut ListEntry;
    }
    (*newentry).data = data;
    (*newentry).next = 0 as *mut ListEntry;
    if (*list).is_null() {
        *list = newentry;
        (*newentry).prev = 0 as *mut ListEntry;
    } else {
        rover = *list;
        while !((*rover).next).is_null() {
            rover = (*rover).next;
        }
        (*newentry).prev = rover;
        (*rover).next = newentry;
    }
    return newentry;
}
#[no_mangle]
pub unsafe extern "C" fn list_data(mut listentry: *mut ListEntry) -> ListValue {
    if listentry.is_null() {
        return 0 as *mut libc::c_void;
    }
    return (*listentry).data;
}
#[no_mangle]
pub unsafe extern "C" fn list_set_data(
    mut listentry: *mut ListEntry,
    mut value: ListValue,
) {
    if !listentry.is_null() {
        (*listentry).data = value;
    }
}
#[no_mangle]
pub unsafe extern "C" fn list_prev(mut listentry: *mut ListEntry) -> *mut ListEntry {
    if listentry.is_null() {
        return 0 as *mut ListEntry;
    }
    return (*listentry).prev;
}
#[no_mangle]
pub unsafe extern "C" fn list_next(mut listentry: *mut ListEntry) -> *mut ListEntry {
    if listentry.is_null() {
        return 0 as *mut ListEntry;
    }
    return (*listentry).next;
}
#[no_mangle]
pub unsafe extern "C" fn list_nth_entry(
    mut list: *mut ListEntry,
    mut n: libc::c_uint,
) -> *mut ListEntry {
    let mut entry: *mut ListEntry = 0 as *mut ListEntry;
    let mut i: libc::c_uint = 0;
    entry = list;
    i = 0 as libc::c_int as libc::c_uint;
    while i < n {
        if entry.is_null() {
            return 0 as *mut ListEntry;
        }
        entry = (*entry).next;
        i = i.wrapping_add(1);
        i;
    }
    return entry;
}
#[no_mangle]
pub unsafe extern "C" fn list_nth_data(
    mut list: *mut ListEntry,
    mut n: libc::c_uint,
) -> ListValue {
    let mut entry: *mut ListEntry = 0 as *mut ListEntry;
    entry = list_nth_entry(list, n);
    if entry.is_null() { return 0 as *mut libc::c_void } else { return (*entry).data };
}
#[no_mangle]
pub unsafe extern "C" fn list_length(mut list: *mut ListEntry) -> libc::c_uint {
    let mut entry: *mut ListEntry = 0 as *mut ListEntry;
    let mut length: libc::c_uint = 0;
    length = 0 as libc::c_int as libc::c_uint;
    entry = list;
    while !entry.is_null() {
        length = length.wrapping_add(1);
        length;
        entry = (*entry).next;
    }
    return length;
}
#[no_mangle]
pub unsafe extern "C" fn list_to_array(mut list: *mut ListEntry) -> *mut ListValue {
    let mut rover: *mut ListEntry = 0 as *mut ListEntry;
    let mut array: *mut ListValue = 0 as *mut ListValue;
    let mut length: libc::c_uint = 0;
    let mut i: libc::c_uint = 0;
    length = list_length(list);
    array = alloc_test_malloc(
        (::core::mem::size_of::<ListValue>() as libc::c_ulong)
            .wrapping_mul(length as libc::c_ulong),
    ) as *mut ListValue;
    if array.is_null() {
        return 0 as *mut ListValue;
    }
    rover = list;
    i = 0 as libc::c_int as libc::c_uint;
    while i < length {
        let ref mut fresh0 = *array.offset(i as isize);
        *fresh0 = (*rover).data;
        rover = (*rover).next;
        i = i.wrapping_add(1);
        i;
    }
    return array;
}
#[no_mangle]
pub unsafe extern "C" fn list_remove_entry(
    mut list: *mut *mut ListEntry,
    mut entry: *mut ListEntry,
) -> libc::c_int {
    if list.is_null() || (*list).is_null() || entry.is_null() {
        return 0 as libc::c_int;
    }
    if ((*entry).prev).is_null() {
        *list = (*entry).next;
        if !((*entry).next).is_null() {
            (*(*entry).next).prev = 0 as *mut ListEntry;
        }
    } else {
        (*(*entry).prev).next = (*entry).next;
        if !((*entry).next).is_null() {
            (*(*entry).next).prev = (*entry).prev;
        }
    }
    alloc_test_free(entry as *mut libc::c_void);
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn list_remove_data(
    mut list: *mut *mut ListEntry,
    mut callback: ListEqualFunc,
    mut data: ListValue,
) -> libc::c_uint {
    let mut entries_removed: libc::c_uint = 0;
    let mut rover: *mut ListEntry = 0 as *mut ListEntry;
    let mut next: *mut ListEntry = 0 as *mut ListEntry;
    if list.is_null() || callback.is_none() {
        return 0 as libc::c_int as libc::c_uint;
    }
    entries_removed = 0 as libc::c_int as libc::c_uint;
    rover = *list;
    while !rover.is_null() {
        next = (*rover).next;
        if callback.expect("non-null function pointer")((*rover).data, data) != 0 {
            if ((*rover).prev).is_null() {
                *list = (*rover).next;
            } else {
                (*(*rover).prev).next = (*rover).next;
            }
            if !((*rover).next).is_null() {
                (*(*rover).next).prev = (*rover).prev;
            }
            alloc_test_free(rover as *mut libc::c_void);
            entries_removed = entries_removed.wrapping_add(1);
            entries_removed;
        }
        rover = next;
    }
    return entries_removed;
}
unsafe extern "C" fn list_sort_internal(
    mut list: *mut *mut ListEntry,
    mut compare_func: ListCompareFunc,
) -> *mut ListEntry {
    let mut pivot: *mut ListEntry = 0 as *mut ListEntry;
    let mut rover: *mut ListEntry = 0 as *mut ListEntry;
    let mut less_list: *mut ListEntry = 0 as *mut ListEntry;
    let mut more_list: *mut ListEntry = 0 as *mut ListEntry;
    let mut less_list_end: *mut ListEntry = 0 as *mut ListEntry;
    let mut more_list_end: *mut ListEntry = 0 as *mut ListEntry;
    if list.is_null() || compare_func.is_none() {
        return 0 as *mut ListEntry;
    }
    if (*list).is_null() || ((**list).next).is_null() {
        return *list;
    }
    pivot = *list;
    less_list = 0 as *mut ListEntry;
    more_list = 0 as *mut ListEntry;
    rover = (**list).next;
    while !rover.is_null() {
        let mut next: *mut ListEntry = (*rover).next;
        if compare_func.expect("non-null function pointer")((*rover).data, (*pivot).data)
            < 0 as libc::c_int
        {
            (*rover).prev = 0 as *mut ListEntry;
            (*rover).next = less_list;
            if !less_list.is_null() {
                (*less_list).prev = rover;
            }
            less_list = rover;
        } else {
            (*rover).prev = 0 as *mut ListEntry;
            (*rover).next = more_list;
            if !more_list.is_null() {
                (*more_list).prev = rover;
            }
            more_list = rover;
        }
        rover = next;
    }
    less_list_end = list_sort_internal(&mut less_list, compare_func);
    more_list_end = list_sort_internal(&mut more_list, compare_func);
    *list = less_list;
    if less_list.is_null() {
        (*pivot).prev = 0 as *mut ListEntry;
        *list = pivot;
    } else {
        (*pivot).prev = less_list_end;
        (*less_list_end).next = pivot;
    }
    (*pivot).next = more_list;
    if !more_list.is_null() {
        (*more_list).prev = pivot;
    }
    if more_list.is_null() { return pivot } else { return more_list_end };
}
#[no_mangle]
pub unsafe extern "C" fn list_sort(
    mut list: *mut *mut ListEntry,
    mut compare_func: ListCompareFunc,
) {
    list_sort_internal(list, compare_func);
}
#[no_mangle]
pub unsafe extern "C" fn list_find_data(
    mut list: *mut ListEntry,
    mut callback: ListEqualFunc,
    mut data: ListValue,
) -> *mut ListEntry {
    let mut rover: *mut ListEntry = 0 as *mut ListEntry;
    rover = list;
    while !rover.is_null() {
        if callback.expect("non-null function pointer")((*rover).data, data)
            != 0 as libc::c_int
        {
            return rover;
        }
        rover = (*rover).next;
    }
    return 0 as *mut ListEntry;
}
#[no_mangle]
pub unsafe extern "C" fn list_iterate(
    mut list: *mut *mut ListEntry,
    mut iter: *mut ListIterator,
) {
    (*iter).prev_next = list;
    (*iter).current = 0 as *mut ListEntry;
}
#[no_mangle]
pub unsafe extern "C" fn list_iter_has_more(mut iter: *mut ListIterator) -> libc::c_int {
    if ((*iter).current).is_null() || (*iter).current != *(*iter).prev_next {
        return (*(*iter).prev_next != 0 as *mut libc::c_void as *mut ListEntry)
            as libc::c_int
    } else {
        return ((*(*iter).current).next != 0 as *mut libc::c_void as *mut ListEntry)
            as libc::c_int
    };
}
#[no_mangle]
pub unsafe extern "C" fn list_iter_next(mut iter: *mut ListIterator) -> ListValue {
    if ((*iter).current).is_null() || (*iter).current != *(*iter).prev_next {
        (*iter).current = *(*iter).prev_next;
    } else {
        (*iter).prev_next = &mut (*(*iter).current).next;
        (*iter).current = (*(*iter).current).next;
    }
    if ((*iter).current).is_null() {
        return 0 as *mut libc::c_void
    } else {
        return (*(*iter).current).data
    };
}
#[no_mangle]
pub unsafe extern "C" fn list_iter_remove(mut iter: *mut ListIterator) {
    if !(((*iter).current).is_null() || (*iter).current != *(*iter).prev_next) {
        *(*iter).prev_next = (*(*iter).current).next;
        if !((*(*iter).current).next).is_null() {
            (*(*(*iter).current).next).prev = (*(*iter).current).prev;
        }
        alloc_test_free((*iter).current as *mut libc::c_void);
        (*iter).current = 0 as *mut ListEntry;
    }
}
