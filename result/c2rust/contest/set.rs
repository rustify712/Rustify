use ::libc;
extern "C" {
    fn alloc_test_malloc(bytes: size_t) -> *mut libc::c_void;
    fn alloc_test_free(ptr: *mut libc::c_void);
    fn alloc_test_calloc(nmemb: size_t, bytes: size_t) -> *mut libc::c_void;
}
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _Set {
    pub table: *mut *mut SetEntry,
    pub entries: libc::c_uint,
    pub table_size: libc::c_uint,
    pub prime_index: libc::c_uint,
    pub hash_func: SetHashFunc,
    pub equal_func: SetEqualFunc,
    pub free_func: SetFreeFunc,
}
pub type SetFreeFunc = Option::<unsafe extern "C" fn(SetValue) -> ()>;
pub type SetValue = *mut libc::c_void;
pub type SetEqualFunc = Option::<
    unsafe extern "C" fn(SetValue, SetValue) -> libc::c_int,
>;
pub type SetHashFunc = Option::<unsafe extern "C" fn(SetValue) -> libc::c_uint>;
pub type SetEntry = _SetEntry;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _SetEntry {
    pub data: SetValue,
    pub next: *mut SetEntry,
}
pub type Set = _Set;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _SetIterator {
    pub set: *mut Set,
    pub next_entry: *mut SetEntry,
    pub next_chain: libc::c_uint,
}
pub type SetIterator = _SetIterator;
static mut set_primes: [libc::c_uint; 24] = [
    193 as libc::c_int as libc::c_uint,
    389 as libc::c_int as libc::c_uint,
    769 as libc::c_int as libc::c_uint,
    1543 as libc::c_int as libc::c_uint,
    3079 as libc::c_int as libc::c_uint,
    6151 as libc::c_int as libc::c_uint,
    12289 as libc::c_int as libc::c_uint,
    24593 as libc::c_int as libc::c_uint,
    49157 as libc::c_int as libc::c_uint,
    98317 as libc::c_int as libc::c_uint,
    196613 as libc::c_int as libc::c_uint,
    393241 as libc::c_int as libc::c_uint,
    786433 as libc::c_int as libc::c_uint,
    1572869 as libc::c_int as libc::c_uint,
    3145739 as libc::c_int as libc::c_uint,
    6291469 as libc::c_int as libc::c_uint,
    12582917 as libc::c_int as libc::c_uint,
    25165843 as libc::c_int as libc::c_uint,
    50331653 as libc::c_int as libc::c_uint,
    100663319 as libc::c_int as libc::c_uint,
    201326611 as libc::c_int as libc::c_uint,
    402653189 as libc::c_int as libc::c_uint,
    805306457 as libc::c_int as libc::c_uint,
    1610612741 as libc::c_int as libc::c_uint,
];
static mut set_num_primes: libc::c_uint = 0;
unsafe extern "C" fn set_allocate_table(mut set: *mut Set) -> libc::c_int {
    if (*set).prime_index < set_num_primes {
        (*set).table_size = set_primes[(*set).prime_index as usize];
    } else {
        (*set)
            .table_size = ((*set).entries)
            .wrapping_mul(10 as libc::c_int as libc::c_uint);
    }
    (*set)
        .table = alloc_test_calloc(
        (*set).table_size as size_t,
        ::core::mem::size_of::<*mut SetEntry>() as libc::c_ulong,
    ) as *mut *mut SetEntry;
    return ((*set).table != 0 as *mut libc::c_void as *mut *mut SetEntry) as libc::c_int;
}
unsafe extern "C" fn set_free_entry(mut set: *mut Set, mut entry: *mut SetEntry) {
    if ((*set).free_func).is_some() {
        ((*set).free_func).expect("non-null function pointer")((*entry).data);
    }
    alloc_test_free(entry as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn set_new(
    mut hash_func: SetHashFunc,
    mut equal_func: SetEqualFunc,
) -> *mut Set {
    let mut new_set: *mut Set = 0 as *mut Set;
    new_set = alloc_test_malloc(::core::mem::size_of::<Set>() as libc::c_ulong)
        as *mut Set;
    if new_set.is_null() {
        return 0 as *mut Set;
    }
    (*new_set).hash_func = hash_func;
    (*new_set).equal_func = equal_func;
    (*new_set).entries = 0 as libc::c_int as libc::c_uint;
    (*new_set).prime_index = 0 as libc::c_int as libc::c_uint;
    (*new_set).free_func = None;
    if set_allocate_table(new_set) == 0 {
        alloc_test_free(new_set as *mut libc::c_void);
        return 0 as *mut Set;
    }
    return new_set;
}
#[no_mangle]
pub unsafe extern "C" fn set_free(mut set: *mut Set) {
    let mut rover: *mut SetEntry = 0 as *mut SetEntry;
    let mut next: *mut SetEntry = 0 as *mut SetEntry;
    let mut i: libc::c_uint = 0;
    i = 0 as libc::c_int as libc::c_uint;
    while i < (*set).table_size {
        rover = *((*set).table).offset(i as isize);
        while !rover.is_null() {
            next = (*rover).next;
            set_free_entry(set, rover);
            rover = next;
        }
        i = i.wrapping_add(1);
        i;
    }
    alloc_test_free((*set).table as *mut libc::c_void);
    alloc_test_free(set as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn set_register_free_function(
    mut set: *mut Set,
    mut free_func: SetFreeFunc,
) {
    (*set).free_func = free_func;
}
unsafe extern "C" fn set_enlarge(mut set: *mut Set) -> libc::c_int {
    let mut rover: *mut SetEntry = 0 as *mut SetEntry;
    let mut next: *mut SetEntry = 0 as *mut SetEntry;
    let mut old_table: *mut *mut SetEntry = 0 as *mut *mut SetEntry;
    let mut old_table_size: libc::c_uint = 0;
    let mut old_prime_index: libc::c_uint = 0;
    let mut index: libc::c_uint = 0;
    let mut i: libc::c_uint = 0;
    old_table = (*set).table;
    old_table_size = (*set).table_size;
    old_prime_index = (*set).prime_index;
    (*set).prime_index = ((*set).prime_index).wrapping_add(1);
    (*set).prime_index;
    if set_allocate_table(set) == 0 {
        (*set).table = old_table;
        (*set).table_size = old_table_size;
        (*set).prime_index = old_prime_index;
        return 0 as libc::c_int;
    }
    i = 0 as libc::c_int as libc::c_uint;
    while i < old_table_size {
        rover = *old_table.offset(i as isize);
        while !rover.is_null() {
            next = (*rover).next;
            index = (((*set).hash_func)
                .expect("non-null function pointer")((*rover).data))
                .wrapping_rem((*set).table_size);
            (*rover).next = *((*set).table).offset(index as isize);
            let ref mut fresh0 = *((*set).table).offset(index as isize);
            *fresh0 = rover;
            rover = next;
        }
        i = i.wrapping_add(1);
        i;
    }
    alloc_test_free(old_table as *mut libc::c_void);
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn set_insert(
    mut set: *mut Set,
    mut data: SetValue,
) -> libc::c_int {
    let mut newentry: *mut SetEntry = 0 as *mut SetEntry;
    let mut rover: *mut SetEntry = 0 as *mut SetEntry;
    let mut index: libc::c_uint = 0;
    if ((*set).entries)
        .wrapping_mul(3 as libc::c_int as libc::c_uint)
        .wrapping_div((*set).table_size) > 0 as libc::c_int as libc::c_uint
    {
        if set_enlarge(set) == 0 {
            return 0 as libc::c_int;
        }
    }
    index = (((*set).hash_func).expect("non-null function pointer")(data))
        .wrapping_rem((*set).table_size);
    rover = *((*set).table).offset(index as isize);
    while !rover.is_null() {
        if ((*set).equal_func).expect("non-null function pointer")(data, (*rover).data)
            != 0 as libc::c_int
        {
            return 0 as libc::c_int;
        }
        rover = (*rover).next;
    }
    newentry = alloc_test_malloc(::core::mem::size_of::<SetEntry>() as libc::c_ulong)
        as *mut SetEntry;
    if newentry.is_null() {
        return 0 as libc::c_int;
    }
    (*newentry).data = data;
    (*newentry).next = *((*set).table).offset(index as isize);
    let ref mut fresh1 = *((*set).table).offset(index as isize);
    *fresh1 = newentry;
    (*set).entries = ((*set).entries).wrapping_add(1);
    (*set).entries;
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn set_remove(
    mut set: *mut Set,
    mut data: SetValue,
) -> libc::c_int {
    let mut rover: *mut *mut SetEntry = 0 as *mut *mut SetEntry;
    let mut entry: *mut SetEntry = 0 as *mut SetEntry;
    let mut index: libc::c_uint = 0;
    index = (((*set).hash_func).expect("non-null function pointer")(data))
        .wrapping_rem((*set).table_size);
    rover = &mut *((*set).table).offset(index as isize) as *mut *mut SetEntry;
    while !(*rover).is_null() {
        if ((*set).equal_func).expect("non-null function pointer")(data, (**rover).data)
            != 0 as libc::c_int
        {
            entry = *rover;
            *rover = (*entry).next;
            (*set).entries = ((*set).entries).wrapping_sub(1);
            (*set).entries;
            set_free_entry(set, entry);
            return 1 as libc::c_int;
        }
        rover = &mut (**rover).next;
    }
    return 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn set_query(
    mut set: *mut Set,
    mut data: SetValue,
) -> libc::c_int {
    let mut rover: *mut SetEntry = 0 as *mut SetEntry;
    let mut index: libc::c_uint = 0;
    index = (((*set).hash_func).expect("non-null function pointer")(data))
        .wrapping_rem((*set).table_size);
    rover = *((*set).table).offset(index as isize);
    while !rover.is_null() {
        if ((*set).equal_func).expect("non-null function pointer")(data, (*rover).data)
            != 0 as libc::c_int
        {
            return 1 as libc::c_int;
        }
        rover = (*rover).next;
    }
    return 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn set_num_entries(mut set: *mut Set) -> libc::c_uint {
    return (*set).entries;
}
#[no_mangle]
pub unsafe extern "C" fn set_to_array(mut set: *mut Set) -> *mut SetValue {
    let mut array: *mut SetValue = 0 as *mut SetValue;
    let mut array_counter: libc::c_int = 0;
    let mut i: libc::c_uint = 0;
    let mut rover: *mut SetEntry = 0 as *mut SetEntry;
    array = alloc_test_malloc(
        (::core::mem::size_of::<SetValue>() as libc::c_ulong)
            .wrapping_mul((*set).entries as libc::c_ulong),
    ) as *mut SetValue;
    if array.is_null() {
        return 0 as *mut SetValue;
    }
    array_counter = 0 as libc::c_int;
    i = 0 as libc::c_int as libc::c_uint;
    while i < (*set).table_size {
        rover = *((*set).table).offset(i as isize);
        while !rover.is_null() {
            let ref mut fresh2 = *array.offset(array_counter as isize);
            *fresh2 = (*rover).data;
            array_counter += 1;
            array_counter;
            rover = (*rover).next;
        }
        i = i.wrapping_add(1);
        i;
    }
    return array;
}
#[no_mangle]
pub unsafe extern "C" fn set_union(mut set1: *mut Set, mut set2: *mut Set) -> *mut Set {
    let mut iterator: SetIterator = SetIterator {
        set: 0 as *mut Set,
        next_entry: 0 as *mut SetEntry,
        next_chain: 0,
    };
    let mut new_set: *mut Set = 0 as *mut Set;
    let mut value: SetValue = 0 as *mut libc::c_void;
    new_set = set_new((*set1).hash_func, (*set1).equal_func);
    if new_set.is_null() {
        return 0 as *mut Set;
    }
    set_iterate(set1, &mut iterator);
    while set_iter_has_more(&mut iterator) != 0 {
        value = set_iter_next(&mut iterator);
        if set_insert(new_set, value) == 0 {
            set_free(new_set);
            return 0 as *mut Set;
        }
    }
    set_iterate(set2, &mut iterator);
    while set_iter_has_more(&mut iterator) != 0 {
        value = set_iter_next(&mut iterator);
        if set_query(new_set, value) == 0 as libc::c_int {
            if set_insert(new_set, value) == 0 {
                set_free(new_set);
                return 0 as *mut Set;
            }
        }
    }
    return new_set;
}
#[no_mangle]
pub unsafe extern "C" fn set_intersection(
    mut set1: *mut Set,
    mut set2: *mut Set,
) -> *mut Set {
    let mut new_set: *mut Set = 0 as *mut Set;
    let mut iterator: SetIterator = SetIterator {
        set: 0 as *mut Set,
        next_entry: 0 as *mut SetEntry,
        next_chain: 0,
    };
    let mut value: SetValue = 0 as *mut libc::c_void;
    new_set = set_new((*set1).hash_func, (*set2).equal_func);
    if new_set.is_null() {
        return 0 as *mut Set;
    }
    set_iterate(set1, &mut iterator);
    while set_iter_has_more(&mut iterator) != 0 {
        value = set_iter_next(&mut iterator);
        if set_query(set2, value) != 0 as libc::c_int {
            if set_insert(new_set, value) == 0 {
                set_free(new_set);
                return 0 as *mut Set;
            }
        }
    }
    return new_set;
}
#[no_mangle]
pub unsafe extern "C" fn set_iterate(mut set: *mut Set, mut iter: *mut SetIterator) {
    let mut chain: libc::c_uint = 0;
    (*iter).set = set;
    (*iter).next_entry = 0 as *mut SetEntry;
    chain = 0 as libc::c_int as libc::c_uint;
    while chain < (*set).table_size {
        if !(*((*set).table).offset(chain as isize)).is_null() {
            (*iter).next_entry = *((*set).table).offset(chain as isize);
            break;
        } else {
            chain = chain.wrapping_add(1);
            chain;
        }
    }
    (*iter).next_chain = chain;
}
#[no_mangle]
pub unsafe extern "C" fn set_iter_next(mut iterator: *mut SetIterator) -> SetValue {
    let mut set: *mut Set = 0 as *mut Set;
    let mut result: SetValue = 0 as *mut libc::c_void;
    let mut current_entry: *mut SetEntry = 0 as *mut SetEntry;
    let mut chain: libc::c_uint = 0;
    set = (*iterator).set;
    if ((*iterator).next_entry).is_null() {
        return 0 as *mut libc::c_void;
    }
    current_entry = (*iterator).next_entry;
    result = (*current_entry).data;
    if !((*current_entry).next).is_null() {
        (*iterator).next_entry = (*current_entry).next;
    } else {
        (*iterator).next_entry = 0 as *mut SetEntry;
        chain = ((*iterator).next_chain).wrapping_add(1 as libc::c_int as libc::c_uint);
        while chain < (*set).table_size {
            if !(*((*set).table).offset(chain as isize)).is_null() {
                (*iterator).next_entry = *((*set).table).offset(chain as isize);
                break;
            } else {
                chain = chain.wrapping_add(1);
                chain;
            }
        }
        (*iterator).next_chain = chain;
    }
    return result;
}
#[no_mangle]
pub unsafe extern "C" fn set_iter_has_more(
    mut iterator: *mut SetIterator,
) -> libc::c_int {
    return ((*iterator).next_entry != 0 as *mut libc::c_void as *mut SetEntry)
        as libc::c_int;
}
unsafe extern "C" fn run_static_initializers() {
    set_num_primes = (::core::mem::size_of::<[libc::c_uint; 24]>() as libc::c_ulong)
        .wrapping_div(::core::mem::size_of::<libc::c_int>() as libc::c_ulong)
        as libc::c_uint;
}
#[used]
#[cfg_attr(target_os = "linux", link_section = ".init_array")]
#[cfg_attr(target_os = "windows", link_section = ".CRT$XIB")]
#[cfg_attr(target_os = "macos", link_section = "__DATA,__mod_init_func")]
static INIT_ARRAY: [unsafe extern "C" fn(); 1] = [run_static_initializers];
