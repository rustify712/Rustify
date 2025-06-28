use ::libc;
extern "C" {
    fn alloc_test_malloc(bytes: size_t) -> *mut libc::c_void;
    fn alloc_test_free(ptr: *mut libc::c_void);
}
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _SListEntry {
    pub data: SListValue,
    pub next: *mut SListEntry,
}
pub type SListEntry = _SListEntry;
pub type SListValue = *mut libc::c_void;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _SListIterator {
    pub prev_next: *mut *mut SListEntry,
    pub current: *mut SListEntry,
}
pub type SListIterator = _SListIterator;
pub type SListCompareFunc = Option::<
    unsafe extern "C" fn(SListValue, SListValue) -> libc::c_int,
>;
pub type SListEqualFunc = Option::<
    unsafe extern "C" fn(SListValue, SListValue) -> libc::c_int,
>;
#[no_mangle]
pub unsafe extern "C" fn slist_free(mut list: *mut SListEntry) {
    let mut entry: *mut SListEntry = 0 as *mut SListEntry;
    entry = list;
    while !entry.is_null() {
        let mut next: *mut SListEntry = 0 as *mut SListEntry;
        next = (*entry).next;
        alloc_test_free(entry as *mut libc::c_void);
        entry = next;
    }
}
#[no_mangle]
pub unsafe extern "C" fn slist_prepend(
    mut list: *mut *mut SListEntry,
    mut data: SListValue,
) -> *mut SListEntry {
    let mut newentry: *mut SListEntry = 0 as *mut SListEntry;
    newentry = alloc_test_malloc(::core::mem::size_of::<SListEntry>() as libc::c_ulong)
        as *mut SListEntry;
    if newentry.is_null() {
        return 0 as *mut SListEntry;
    }
    (*newentry).data = data;
    (*newentry).next = *list;
    *list = newentry;
    return newentry;
}
#[no_mangle]
pub unsafe extern "C" fn slist_append(
    mut list: *mut *mut SListEntry,
    mut data: SListValue,
) -> *mut SListEntry {
    let mut rover: *mut SListEntry = 0 as *mut SListEntry;
    let mut newentry: *mut SListEntry = 0 as *mut SListEntry;
    newentry = alloc_test_malloc(::core::mem::size_of::<SListEntry>() as libc::c_ulong)
        as *mut SListEntry;
    if newentry.is_null() {
        return 0 as *mut SListEntry;
    }
    (*newentry).data = data;
    (*newentry).next = 0 as *mut SListEntry;
    if (*list).is_null() {
        *list = newentry;
    } else {
        rover = *list;
        while !((*rover).next).is_null() {
            rover = (*rover).next;
        }
        (*rover).next = newentry;
    }
    return newentry;
}
#[no_mangle]
pub unsafe extern "C" fn slist_data(mut listentry: *mut SListEntry) -> SListValue {
    return (*listentry).data;
}
#[no_mangle]
pub unsafe extern "C" fn slist_set_data(
    mut listentry: *mut SListEntry,
    mut data: SListValue,
) {
    if !listentry.is_null() {
        (*listentry).data = data;
    }
}
#[no_mangle]
pub unsafe extern "C" fn slist_next(mut listentry: *mut SListEntry) -> *mut SListEntry {
    return (*listentry).next;
}
#[no_mangle]
pub unsafe extern "C" fn slist_nth_entry(
    mut list: *mut SListEntry,
    mut n: libc::c_uint,
) -> *mut SListEntry {
    let mut entry: *mut SListEntry = 0 as *mut SListEntry;
    let mut i: libc::c_uint = 0;
    entry = list;
    i = 0 as libc::c_int as libc::c_uint;
    while i < n {
        if entry.is_null() {
            return 0 as *mut SListEntry;
        }
        entry = (*entry).next;
        i = i.wrapping_add(1);
        i;
    }
    return entry;
}
#[no_mangle]
pub unsafe extern "C" fn slist_nth_data(
    mut list: *mut SListEntry,
    mut n: libc::c_uint,
) -> SListValue {
    let mut entry: *mut SListEntry = 0 as *mut SListEntry;
    entry = slist_nth_entry(list, n);
    if entry.is_null() { return 0 as *mut libc::c_void } else { return (*entry).data };
}
#[no_mangle]
pub unsafe extern "C" fn slist_length(mut list: *mut SListEntry) -> libc::c_uint {
    let mut entry: *mut SListEntry = 0 as *mut SListEntry;
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
pub unsafe extern "C" fn slist_to_array(mut list: *mut SListEntry) -> *mut SListValue {
    let mut rover: *mut SListEntry = 0 as *mut SListEntry;
    let mut array: *mut SListValue = 0 as *mut SListValue;
    let mut length: libc::c_uint = 0;
    let mut i: libc::c_uint = 0;
    length = slist_length(list);
    array = alloc_test_malloc(
        (::core::mem::size_of::<SListValue>() as libc::c_ulong)
            .wrapping_mul(length as libc::c_ulong),
    ) as *mut SListValue;
    if array.is_null() {
        return 0 as *mut SListValue;
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
pub unsafe extern "C" fn slist_remove_entry(
    mut list: *mut *mut SListEntry,
    mut entry: *mut SListEntry,
) -> libc::c_int {
    let mut rover: *mut SListEntry = 0 as *mut SListEntry;
    if (*list).is_null() || entry.is_null() {
        return 0 as libc::c_int;
    }
    if *list == entry {
        *list = (*entry).next;
    } else {
        rover = *list;
        while !rover.is_null() && (*rover).next != entry {
            rover = (*rover).next;
        }
        if rover.is_null() {
            return 0 as libc::c_int
        } else {
            (*rover).next = (*entry).next;
        }
    }
    alloc_test_free(entry as *mut libc::c_void);
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn slist_remove_data(
    mut list: *mut *mut SListEntry,
    mut callback: SListEqualFunc,
    mut data: SListValue,
) -> libc::c_uint {
    let mut rover: *mut *mut SListEntry = 0 as *mut *mut SListEntry;
    let mut next: *mut SListEntry = 0 as *mut SListEntry;
    let mut entries_removed: libc::c_uint = 0;
    entries_removed = 0 as libc::c_int as libc::c_uint;
    rover = list;
    while !(*rover).is_null() {
        if callback.expect("non-null function pointer")((**rover).data, data)
            != 0 as libc::c_int
        {
            next = (**rover).next;
            alloc_test_free(*rover as *mut libc::c_void);
            *rover = next;
            entries_removed = entries_removed.wrapping_add(1);
            entries_removed;
        } else {
            rover = &mut (**rover).next;
        }
    }
    return entries_removed;
}
unsafe extern "C" fn slist_sort_internal(
    mut list: *mut *mut SListEntry,
    mut compare_func: SListCompareFunc,
) -> *mut SListEntry {
    let mut pivot: *mut SListEntry = 0 as *mut SListEntry;
    let mut rover: *mut SListEntry = 0 as *mut SListEntry;
    let mut less_list: *mut SListEntry = 0 as *mut SListEntry;
    let mut more_list: *mut SListEntry = 0 as *mut SListEntry;
    let mut less_list_end: *mut SListEntry = 0 as *mut SListEntry;
    let mut more_list_end: *mut SListEntry = 0 as *mut SListEntry;
    if (*list).is_null() || ((**list).next).is_null() {
        return *list;
    }
    pivot = *list;
    less_list = 0 as *mut SListEntry;
    more_list = 0 as *mut SListEntry;
    rover = (**list).next;
    while !rover.is_null() {
        let mut next: *mut SListEntry = (*rover).next;
        if compare_func.expect("non-null function pointer")((*rover).data, (*pivot).data)
            < 0 as libc::c_int
        {
            (*rover).next = less_list;
            less_list = rover;
        } else {
            (*rover).next = more_list;
            more_list = rover;
        }
        rover = next;
    }
    less_list_end = slist_sort_internal(&mut less_list, compare_func);
    more_list_end = slist_sort_internal(&mut more_list, compare_func);
    *list = less_list;
    if less_list.is_null() {
        *list = pivot;
    } else {
        (*less_list_end).next = pivot;
    }
    (*pivot).next = more_list;
    if more_list.is_null() { return pivot } else { return more_list_end };
}
#[no_mangle]
pub unsafe extern "C" fn slist_sort(
    mut list: *mut *mut SListEntry,
    mut compare_func: SListCompareFunc,
) {
    slist_sort_internal(list, compare_func);
}
#[no_mangle]
pub unsafe extern "C" fn slist_find_data(
    mut list: *mut SListEntry,
    mut callback: SListEqualFunc,
    mut data: SListValue,
) -> *mut SListEntry {
    let mut rover: *mut SListEntry = 0 as *mut SListEntry;
    rover = list;
    while !rover.is_null() {
        if callback.expect("non-null function pointer")((*rover).data, data)
            != 0 as libc::c_int
        {
            return rover;
        }
        rover = (*rover).next;
    }
    return 0 as *mut SListEntry;
}
#[no_mangle]
pub unsafe extern "C" fn slist_iterate(
    mut list: *mut *mut SListEntry,
    mut iter: *mut SListIterator,
) {
    (*iter).prev_next = list;
    (*iter).current = 0 as *mut SListEntry;
}
#[no_mangle]
pub unsafe extern "C" fn slist_iter_has_more(
    mut iter: *mut SListIterator,
) -> libc::c_int {
    if ((*iter).current).is_null() || (*iter).current != *(*iter).prev_next {
        return (*(*iter).prev_next != 0 as *mut libc::c_void as *mut SListEntry)
            as libc::c_int
    } else {
        return ((*(*iter).current).next != 0 as *mut libc::c_void as *mut SListEntry)
            as libc::c_int
    };
}
#[no_mangle]
pub unsafe extern "C" fn slist_iter_next(mut iter: *mut SListIterator) -> SListValue {
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
pub unsafe extern "C" fn slist_iter_remove(mut iter: *mut SListIterator) {
    if !(((*iter).current).is_null() || (*iter).current != *(*iter).prev_next) {
        *(*iter).prev_next = (*(*iter).current).next;
        alloc_test_free((*iter).current as *mut libc::c_void);
        (*iter).current = 0 as *mut SListEntry;
    }
}
