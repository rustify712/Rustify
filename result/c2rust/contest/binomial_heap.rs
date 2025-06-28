use ::libc;
extern "C" {
    fn memcpy(
        _: *mut libc::c_void,
        _: *const libc::c_void,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
    fn alloc_test_malloc(bytes: size_t) -> *mut libc::c_void;
    fn alloc_test_free(ptr: *mut libc::c_void);
    fn alloc_test_calloc(nmemb: size_t, bytes: size_t) -> *mut libc::c_void;
}
pub type size_t = libc::c_ulong;
pub type BinomialHeapType = libc::c_uint;
pub const BINOMIAL_HEAP_TYPE_MAX: BinomialHeapType = 1;
pub const BINOMIAL_HEAP_TYPE_MIN: BinomialHeapType = 0;
pub type BinomialHeapValue = *mut libc::c_void;
pub type BinomialHeapCompareFunc = Option::<
    unsafe extern "C" fn(BinomialHeapValue, BinomialHeapValue) -> libc::c_int,
>;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _BinomialHeap {
    pub heap_type: BinomialHeapType,
    pub compare_func: BinomialHeapCompareFunc,
    pub num_values: libc::c_uint,
    pub roots: *mut *mut BinomialTree,
    pub roots_length: libc::c_uint,
}
pub type BinomialTree = _BinomialTree;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _BinomialTree {
    pub value: BinomialHeapValue,
    pub order: libc::c_ushort,
    pub refcount: libc::c_ushort,
    pub subtrees: *mut *mut BinomialTree,
}
pub type BinomialHeap = _BinomialHeap;
unsafe extern "C" fn binomial_heap_cmp(
    mut heap: *mut BinomialHeap,
    mut data1: BinomialHeapValue,
    mut data2: BinomialHeapValue,
) -> libc::c_int {
    if (*heap).heap_type as libc::c_uint
        == BINOMIAL_HEAP_TYPE_MIN as libc::c_int as libc::c_uint
    {
        return ((*heap).compare_func).expect("non-null function pointer")(data1, data2)
    } else {
        return -((*heap).compare_func).expect("non-null function pointer")(data1, data2)
    };
}
unsafe extern "C" fn binomial_tree_ref(mut tree: *mut BinomialTree) {
    if !tree.is_null() {
        (*tree).refcount = ((*tree).refcount).wrapping_add(1);
        (*tree).refcount;
    }
}
unsafe extern "C" fn binomial_tree_unref(mut tree: *mut BinomialTree) {
    let mut i: libc::c_int = 0;
    if tree.is_null() {
        return;
    }
    (*tree).refcount = ((*tree).refcount).wrapping_sub(1);
    (*tree).refcount;
    if (*tree).refcount as libc::c_int == 0 as libc::c_int {
        i = 0 as libc::c_int;
        while i < (*tree).order as libc::c_int {
            binomial_tree_unref(*((*tree).subtrees).offset(i as isize));
            i += 1;
            i;
        }
        alloc_test_free((*tree).subtrees as *mut libc::c_void);
        alloc_test_free(tree as *mut libc::c_void);
    }
}
unsafe extern "C" fn binomial_tree_merge(
    mut heap: *mut BinomialHeap,
    mut tree1: *mut BinomialTree,
    mut tree2: *mut BinomialTree,
) -> *mut BinomialTree {
    let mut new_tree: *mut BinomialTree = 0 as *mut BinomialTree;
    let mut tmp: *mut BinomialTree = 0 as *mut BinomialTree;
    let mut i: libc::c_int = 0;
    if binomial_heap_cmp(heap, (*tree1).value, (*tree2).value) > 0 as libc::c_int {
        tmp = tree1;
        tree1 = tree2;
        tree2 = tmp;
    }
    new_tree = alloc_test_malloc(::core::mem::size_of::<BinomialTree>() as libc::c_ulong)
        as *mut BinomialTree;
    if new_tree.is_null() {
        return 0 as *mut BinomialTree;
    }
    (*new_tree).refcount = 0 as libc::c_int as libc::c_ushort;
    (*new_tree)
        .order = ((*tree1).order as libc::c_int + 1 as libc::c_int) as libc::c_ushort;
    (*new_tree).value = (*tree1).value;
    (*new_tree)
        .subtrees = alloc_test_malloc(
        (::core::mem::size_of::<*mut BinomialTree>() as libc::c_ulong)
            .wrapping_mul((*new_tree).order as libc::c_ulong),
    ) as *mut *mut BinomialTree;
    if ((*new_tree).subtrees).is_null() {
        alloc_test_free(new_tree as *mut libc::c_void);
        return 0 as *mut BinomialTree;
    }
    memcpy(
        (*new_tree).subtrees as *mut libc::c_void,
        (*tree1).subtrees as *const libc::c_void,
        (::core::mem::size_of::<*mut BinomialTree>() as libc::c_ulong)
            .wrapping_mul((*tree1).order as libc::c_ulong),
    );
    let ref mut fresh0 = *((*new_tree).subtrees)
        .offset(((*new_tree).order as libc::c_int - 1 as libc::c_int) as isize);
    *fresh0 = tree2;
    i = 0 as libc::c_int;
    while i < (*new_tree).order as libc::c_int {
        binomial_tree_ref(*((*new_tree).subtrees).offset(i as isize));
        i += 1;
        i;
    }
    return new_tree;
}
unsafe extern "C" fn binomial_heap_merge_undo(
    mut new_roots: *mut *mut BinomialTree,
    mut count: libc::c_uint,
) {
    let mut i: libc::c_uint = 0;
    i = 0 as libc::c_int as libc::c_uint;
    while i <= count {
        binomial_tree_unref(*new_roots.offset(i as isize));
        i = i.wrapping_add(1);
        i;
    }
    alloc_test_free(new_roots as *mut libc::c_void);
}
unsafe extern "C" fn binomial_heap_merge(
    mut heap: *mut BinomialHeap,
    mut other: *mut BinomialHeap,
) -> libc::c_int {
    let mut new_roots: *mut *mut BinomialTree = 0 as *mut *mut BinomialTree;
    let mut new_roots_length: libc::c_uint = 0;
    let mut vals: [*mut BinomialTree; 3] = [0 as *mut BinomialTree; 3];
    let mut num_vals: libc::c_int = 0;
    let mut carry: *mut BinomialTree = 0 as *mut BinomialTree;
    let mut new_carry: *mut BinomialTree = 0 as *mut BinomialTree;
    let mut max: libc::c_uint = 0;
    let mut i: libc::c_uint = 0;
    if (*heap).roots_length > (*other).roots_length {
        max = ((*heap).roots_length).wrapping_add(1 as libc::c_int as libc::c_uint);
    } else {
        max = ((*other).roots_length).wrapping_add(1 as libc::c_int as libc::c_uint);
    }
    new_roots = alloc_test_malloc(
        (::core::mem::size_of::<*mut BinomialTree>() as libc::c_ulong)
            .wrapping_mul(max as libc::c_ulong),
    ) as *mut *mut BinomialTree;
    if new_roots.is_null() {
        return 0 as libc::c_int;
    }
    new_roots_length = 0 as libc::c_int as libc::c_uint;
    carry = 0 as *mut BinomialTree;
    i = 0 as libc::c_int as libc::c_uint;
    while i < max {
        num_vals = 0 as libc::c_int;
        if i < (*heap).roots_length && !(*((*heap).roots).offset(i as isize)).is_null() {
            vals[num_vals as usize] = *((*heap).roots).offset(i as isize);
            num_vals += 1;
            num_vals;
        }
        if i < (*other).roots_length && !(*((*other).roots).offset(i as isize)).is_null()
        {
            vals[num_vals as usize] = *((*other).roots).offset(i as isize);
            num_vals += 1;
            num_vals;
        }
        if !carry.is_null() {
            vals[num_vals as usize] = carry;
            num_vals += 1;
            num_vals;
        }
        if num_vals & 1 as libc::c_int != 0 as libc::c_int {
            let ref mut fresh1 = *new_roots.offset(i as isize);
            *fresh1 = vals[(num_vals - 1 as libc::c_int) as usize];
            binomial_tree_ref(*new_roots.offset(i as isize));
            new_roots_length = i.wrapping_add(1 as libc::c_int as libc::c_uint);
        } else {
            let ref mut fresh2 = *new_roots.offset(i as isize);
            *fresh2 = 0 as *mut BinomialTree;
        }
        if num_vals & 2 as libc::c_int != 0 as libc::c_int {
            new_carry = binomial_tree_merge(
                heap,
                vals[0 as libc::c_int as usize],
                vals[1 as libc::c_int as usize],
            );
            if new_carry.is_null() {
                binomial_heap_merge_undo(new_roots, i);
                binomial_tree_unref(carry);
                return 0 as libc::c_int;
            }
        } else {
            new_carry = 0 as *mut BinomialTree;
        }
        binomial_tree_unref(carry);
        carry = new_carry;
        binomial_tree_ref(carry);
        i = i.wrapping_add(1);
        i;
    }
    i = 0 as libc::c_int as libc::c_uint;
    while i < (*heap).roots_length {
        if !(*((*heap).roots).offset(i as isize)).is_null() {
            binomial_tree_unref(*((*heap).roots).offset(i as isize));
        }
        i = i.wrapping_add(1);
        i;
    }
    alloc_test_free((*heap).roots as *mut libc::c_void);
    (*heap).roots = new_roots;
    (*heap).roots_length = new_roots_length;
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn binomial_heap_new(
    mut heap_type: BinomialHeapType,
    mut compare_func: BinomialHeapCompareFunc,
) -> *mut BinomialHeap {
    let mut new_heap: *mut BinomialHeap = 0 as *mut BinomialHeap;
    new_heap = alloc_test_calloc(
        1 as libc::c_int as size_t,
        ::core::mem::size_of::<BinomialHeap>() as libc::c_ulong,
    ) as *mut BinomialHeap;
    if new_heap.is_null() {
        return 0 as *mut BinomialHeap;
    }
    (*new_heap).heap_type = heap_type;
    (*new_heap).compare_func = compare_func;
    return new_heap;
}
#[no_mangle]
pub unsafe extern "C" fn binomial_heap_free(mut heap: *mut BinomialHeap) {
    let mut i: libc::c_uint = 0;
    i = 0 as libc::c_int as libc::c_uint;
    while i < (*heap).roots_length {
        binomial_tree_unref(*((*heap).roots).offset(i as isize));
        i = i.wrapping_add(1);
        i;
    }
    alloc_test_free((*heap).roots as *mut libc::c_void);
    alloc_test_free(heap as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn binomial_heap_insert(
    mut heap: *mut BinomialHeap,
    mut value: BinomialHeapValue,
) -> libc::c_int {
    let mut fake_heap: BinomialHeap = BinomialHeap {
        heap_type: BINOMIAL_HEAP_TYPE_MIN,
        compare_func: None,
        num_values: 0,
        roots: 0 as *mut *mut BinomialTree,
        roots_length: 0,
    };
    let mut new_tree: *mut BinomialTree = 0 as *mut BinomialTree;
    let mut result: libc::c_int = 0;
    new_tree = alloc_test_malloc(::core::mem::size_of::<BinomialTree>() as libc::c_ulong)
        as *mut BinomialTree;
    if new_tree.is_null() {
        return 0 as libc::c_int;
    }
    (*new_tree).value = value;
    (*new_tree).order = 0 as libc::c_int as libc::c_ushort;
    (*new_tree).refcount = 1 as libc::c_int as libc::c_ushort;
    (*new_tree).subtrees = 0 as *mut *mut BinomialTree;
    fake_heap.heap_type = (*heap).heap_type;
    fake_heap.compare_func = (*heap).compare_func;
    fake_heap.num_values = 1 as libc::c_int as libc::c_uint;
    fake_heap.roots = &mut new_tree;
    fake_heap.roots_length = 1 as libc::c_int as libc::c_uint;
    result = binomial_heap_merge(heap, &mut fake_heap);
    if result != 0 as libc::c_int {
        (*heap).num_values = ((*heap).num_values).wrapping_add(1);
        (*heap).num_values;
    }
    binomial_tree_unref(new_tree);
    return result;
}
#[no_mangle]
pub unsafe extern "C" fn binomial_heap_pop(
    mut heap: *mut BinomialHeap,
) -> BinomialHeapValue {
    let mut least_tree: *mut BinomialTree = 0 as *mut BinomialTree;
    let mut fake_heap: BinomialHeap = BinomialHeap {
        heap_type: BINOMIAL_HEAP_TYPE_MIN,
        compare_func: None,
        num_values: 0,
        roots: 0 as *mut *mut BinomialTree,
        roots_length: 0,
    };
    let mut result: BinomialHeapValue = 0 as *mut libc::c_void;
    let mut i: libc::c_uint = 0;
    let mut least_index: libc::c_uint = 0;
    if (*heap).num_values == 0 as libc::c_int as libc::c_uint {
        return 0 as *mut libc::c_void;
    }
    least_index = (2147483647 as libc::c_int as libc::c_uint)
        .wrapping_mul(2 as libc::c_uint)
        .wrapping_add(1 as libc::c_uint);
    i = 0 as libc::c_int as libc::c_uint;
    while i < (*heap).roots_length {
        if !(*((*heap).roots).offset(i as isize)).is_null() {
            if least_index
                == (2147483647 as libc::c_int as libc::c_uint)
                    .wrapping_mul(2 as libc::c_uint)
                    .wrapping_add(1 as libc::c_uint)
                || binomial_heap_cmp(
                    heap,
                    (**((*heap).roots).offset(i as isize)).value,
                    (**((*heap).roots).offset(least_index as isize)).value,
                ) < 0 as libc::c_int
            {
                least_index = i;
            }
        }
        i = i.wrapping_add(1);
        i;
    }
    least_tree = *((*heap).roots).offset(least_index as isize);
    let ref mut fresh3 = *((*heap).roots).offset(least_index as isize);
    *fresh3 = 0 as *mut BinomialTree;
    fake_heap.heap_type = (*heap).heap_type;
    fake_heap.compare_func = (*heap).compare_func;
    fake_heap.roots = (*least_tree).subtrees;
    fake_heap.roots_length = (*least_tree).order as libc::c_uint;
    if binomial_heap_merge(heap, &mut fake_heap) != 0 {
        result = (*least_tree).value;
        binomial_tree_unref(least_tree);
        (*heap).num_values = ((*heap).num_values).wrapping_sub(1);
        (*heap).num_values;
        return result;
    } else {
        let ref mut fresh4 = *((*heap).roots).offset(least_index as isize);
        *fresh4 = least_tree;
        return 0 as *mut libc::c_void;
    };
}
#[no_mangle]
pub unsafe extern "C" fn binomial_heap_num_entries(
    mut heap: *mut BinomialHeap,
) -> libc::c_uint {
    return (*heap).num_values;
}
