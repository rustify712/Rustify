EXPERIENCE_E0502 = """## E0502: cannot borrow `*` as mutable because it is also borrowed as immutable
该错误意味你尝试同时存在可变和不可变的借用，这违背了 Rust 的借用规则。
Rust 的借用规则要求在同一时间，数据只能有一个可变借用或任意多个不可变借用，但不能同时存在可变和不可变借用。
解决思路：
1. 重构借用的顺序或作用域
通过重新安排代码中的借用顺序，可以避免同时存在可变借用和不可变借用。通常，先使用可变借用操作，再使用不可变借用，或者反过来，都能解决这个问题。
示例：
```rust
let mut node_inner = ...;

// 先进行可变借用操作
let node_mut = &mut node_inner;

// 完成可变借用的操作后，再进行不可变借用
let node_ref = &node_inner;

// 这样就避免了在同一作用域内同时存在可变借用和不可变借用
```
2. 使用 RefCell 或 Mutex
如果在运行时需要进行动态借用检查，并且需要同时进行可变和不可变的借用，可以考虑使用 RefCell（单线程情况） 或 Mutex（多线程情况） 来实现。
RefCell 或 Mutex 允许你在不同的地方进行不同的借用，但它会在运行时对借用规则进行检查，确保不会违反 Rust 的借用规则。
示例：
```rust
use std::cell::RefCell;
let node_inner = RefCell::new(...);

// 使用 RefCell 的 borrow 和 borrow_mut 方法
{
    let mut node_mut = node_inner.borrow_mut();
    // 对 node_mut 做一些操作
}

{
    let node_ref = node_inner.borrow();
    // 对 node_ref 做一些操作
}
```
3. 使用 clone 或复制
在某些情况下，如果数据不大且性能损失可接受，使用 clone 可以避免同时借用冲突，通过复制数据来避免对原始数据的同时借用。
示例：
```rust
let mut node_inner = ...;

// 使用 clone 来避免借用冲突
let node_ref = node_inner.clone(); // 克隆数据
// 对 node_ref 做不可变操作
```
"""

# Rust 错误修复经验
FIXING_EXPERIENCES = {
    ("E0502", ): EXPERIENCE_E0502
}