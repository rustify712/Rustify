use std::any::Any;

mod bounds;
mod node;
mod point;
mod quadtree;

pub use bounds::Bounds;
pub use node::Node;
pub use point::Point;
pub use quadtree::QuadTree;

pub type KeyFreeFn = fn(&mut Box<dyn Any>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_integration() {
        let mut tree = QuadTree::new(0.0, 0.0, 100.0, 100.0).unwrap();
        assert!(tree.insert(10.0, 10.0, Box::new("test")));
        assert_eq!(tree.len(), 1);
        
        let point = tree.search(10.0, 10.0).unwrap();
        assert_eq!(point.x, 10.0);
        assert_eq!(point.y, 10.0);
    }
}