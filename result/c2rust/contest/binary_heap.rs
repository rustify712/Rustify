use ::libc;
extern "C" {
    fn alloc_test_malloc(bytes: size_t) -> *mut libc::c_void;
    fn alloc_test_free(ptr: *mut libc::c_void);
    fn alloc_test_realloc(ptr: *mut libc::c_void, bytes: size_t) -> *mut libc::c_void;
}
pub type size_t = libc::c_ulong;
pub type BinaryHeapType = libc::c_uint;
pub const BINARY_HEAP_TYPE_MAX: BinaryHeapType = 1;
pub const BINARY_HEAP_TYPE_MIN: BinaryHeapType = 0;
pub type BinaryHeapValue = *mut libc::c_void;
pub type BinaryHeapCompareFunc = Option::<
    unsafe extern "C" fn(BinaryHeapValue, BinaryHeapValue) -> libc::c_int,
>;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _BinaryHeap {
    pub heap_type: BinaryHeapType,
    pub values: *mut BinaryHeapValue,
    pub num_values: libc::c_uint,
    pub alloced_size: libc::c_uint,
    pub compare_func: BinaryHeapCompareFunc,
}
pub type BinaryHeap = _BinaryHeap;
unsafe extern "C" fn binary_heap_cmp(
    mut heap: *mut BinaryHeap,
    mut data1: BinaryHeapValue,
    mut data2: BinaryHeapValue,
) -> libc::c_int {
    if (*heap).heap_type as libc::c_uint
        == BINARY_HEAP_TYPE_MIN as libc::c_int as libc::c_uint
    {
        return ((*heap).compare_func).expect("non-null function pointer")(data1, data2)
    } else {
        return -((*heap).compare_func).expect("non-null function pointer")(data1, data2)
    };
}
#[no_mangle]
pub unsafe extern "C" fn binary_heap_new(
    mut heap_type: BinaryHeapType,
    mut compare_func: BinaryHeapCompareFunc,
) -> *mut BinaryHeap {
    let mut heap: *mut BinaryHeap = 0 as *mut BinaryHeap;
    heap = alloc_test_malloc(::core::mem::size_of::<BinaryHeap>() as libc::c_ulong)
        as *mut BinaryHeap;
    if heap.is_null() {
        return 0 as *mut BinaryHeap;
    }
    (*heap).heap_type = heap_type;
    (*heap).num_values = 0 as libc::c_int as libc::c_uint;
    (*heap).compare_func = compare_func;
    (*heap).alloced_size = 16 as libc::c_int as libc::c_uint;
    (*heap)
        .values = alloc_test_malloc(
        (::core::mem::size_of::<BinaryHeapValue>() as libc::c_ulong)
            .wrapping_mul((*heap).alloced_size as libc::c_ulong),
    ) as *mut BinaryHeapValue;
    if ((*heap).values).is_null() {
        alloc_test_free(heap as *mut libc::c_void);
        return 0 as *mut BinaryHeap;
    }
    return heap;
}
#[no_mangle]
pub unsafe extern "C" fn binary_heap_free(mut heap: *mut BinaryHeap) {
    alloc_test_free((*heap).values as *mut libc::c_void);
    alloc_test_free(heap as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn binary_heap_insert(
    mut heap: *mut BinaryHeap,
    mut value: BinaryHeapValue,
) -> libc::c_int {
    let mut new_values: *mut BinaryHeapValue = 0 as *mut BinaryHeapValue;
    let mut index: libc::c_uint = 0;
    let mut new_size: libc::c_uint = 0;
    let mut parent: libc::c_uint = 0;
    if (*heap).num_values >= (*heap).alloced_size {
        new_size = ((*heap).alloced_size).wrapping_mul(2 as libc::c_int as libc::c_uint);
        new_values = alloc_test_realloc(
            (*heap).values as *mut libc::c_void,
            (::core::mem::size_of::<BinaryHeapValue>() as libc::c_ulong)
                .wrapping_mul(new_size as libc::c_ulong),
        ) as *mut BinaryHeapValue;
        if new_values.is_null() {
            return 0 as libc::c_int;
        }
        (*heap).alloced_size = new_size;
        (*heap).values = new_values;
    }
    index = (*heap).num_values;
    (*heap).num_values = ((*heap).num_values).wrapping_add(1);
    (*heap).num_values;
    while index > 0 as libc::c_int as libc::c_uint {
        parent = index
            .wrapping_sub(1 as libc::c_int as libc::c_uint)
            .wrapping_div(2 as libc::c_int as libc::c_uint);
        if binary_heap_cmp(heap, *((*heap).values).offset(parent as isize), value)
            < 0 as libc::c_int
        {
            break;
        }
        let ref mut fresh0 = *((*heap).values).offset(index as isize);
        *fresh0 = *((*heap).values).offset(parent as isize);
        index = parent;
    }
    let ref mut fresh1 = *((*heap).values).offset(index as isize);
    *fresh1 = value;
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn binary_heap_pop(mut heap: *mut BinaryHeap) -> BinaryHeapValue {
    let mut result: BinaryHeapValue = 0 as *mut libc::c_void;
    let mut new_value: BinaryHeapValue = 0 as *mut libc::c_void;
    let mut index: libc::c_uint = 0;
    let mut next_index: libc::c_uint = 0;
    let mut child1: libc::c_uint = 0;
    let mut child2: libc::c_uint = 0;
    if (*heap).num_values == 0 as libc::c_int as libc::c_uint {
        return 0 as *mut libc::c_void;
    }
    result = *((*heap).values).offset(0 as libc::c_int as isize);
    new_value = *((*heap).values)
        .offset(
            ((*heap).num_values).wrapping_sub(1 as libc::c_int as libc::c_uint) as isize,
        );
    (*heap).num_values = ((*heap).num_values).wrapping_sub(1);
    (*heap).num_values;
    index = 0 as libc::c_int as libc::c_uint;
    loop {
        child1 = index
            .wrapping_mul(2 as libc::c_int as libc::c_uint)
            .wrapping_add(1 as libc::c_int as libc::c_uint);
        child2 = index
            .wrapping_mul(2 as libc::c_int as libc::c_uint)
            .wrapping_add(2 as libc::c_int as libc::c_uint);
        if child1 < (*heap).num_values
            && binary_heap_cmp(
                heap,
                new_value,
                *((*heap).values).offset(child1 as isize),
            ) > 0 as libc::c_int
        {
            if child2 < (*heap).num_values
                && binary_heap_cmp(
                    heap,
                    *((*heap).values).offset(child1 as isize),
                    *((*heap).values).offset(child2 as isize),
                ) > 0 as libc::c_int
            {
                next_index = child2;
            } else {
                next_index = child1;
            }
        } else if child2 < (*heap).num_values
            && binary_heap_cmp(
                heap,
                new_value,
                *((*heap).values).offset(child2 as isize),
            ) > 0 as libc::c_int
        {
            next_index = child2;
        } else {
            let ref mut fresh2 = *((*heap).values).offset(index as isize);
            *fresh2 = new_value;
            break;
        }
        let ref mut fresh3 = *((*heap).values).offset(index as isize);
        *fresh3 = *((*heap).values).offset(next_index as isize);
        index = next_index;
    }
    return result;
}
#[no_mangle]
pub unsafe extern "C" fn binary_heap_num_entries(
    mut heap: *mut BinaryHeap,
) -> libc::c_uint {
    return (*heap).num_values;
}
