use super::*;

#[test]
fn test_bounds() {
    let mut bounds = Bounds::new();
    assert_eq!(bounds.nw.x, f64::INFINITY);
    assert_eq!(bounds.nw.y, f64::NEG_INFINITY);
    assert_eq!(bounds.se.x, f64::NEG_INFINITY);
    assert_eq!(bounds.se.y, f64::INFINITY);

    bounds.extend(10.0, 20.0);
    assert_eq!(bounds.nw.x, 10.0);
    assert_eq!(bounds.nw.y, 20.0);
    assert_eq!(bounds.se.x, 10.0);
    assert_eq!(bounds.se.y, 20.0);
}

#[test]
fn test_point() {
    let point = Point::new(5.0, 10.0);
    assert_eq!(point.x, 5.0);
    assert_eq!(point.y, 10.0);
}

#[test]
fn test_node() {
    let node = Node::new();
    assert!(node.is_empty());
    assert!(!node.is_pointer());
    assert!(!node.is_leaf());

    let node = Node::with_bounds(0.0, 0.0, 100.0, 100.0).unwrap();
    assert!(node.bounds.is_some());
}

#[test]
fn test_quadtree_basic() {
    let mut tree = QuadTree::new(0.0, 0.0, 100.0, 100.0).unwrap();
    assert_eq!(tree.len(), 0);

    assert!(tree.insert(10.0, 10.0, Box::new("test1")));
    assert_eq!(tree.len(), 1);

    assert!(tree.insert(20.0, 20.0, Box::new("test2")));
    assert_eq!(tree.len(), 2);
}

#[test]
fn test_quadtree_search() {
    let mut tree = QuadTree::new(0.0, 0.0, 100.0, 100.0).unwrap();
    tree.insert(10.0, 10.0, Box::new("test1"));
    tree.insert(20.0, 20.0, Box::new("test2"));

    assert!(tree.search(10.0, 10.0).is_some());
    assert!(tree.search(20.0, 20.0).is_some());
    assert!(tree.search(30.0, 30.0).is_none());
}

#[test]
fn test_quadtree_boundary() {
    let mut tree = QuadTree::new(0.0, 0.0, 100.0, 100.0).unwrap();
    assert!(!tree.insert(-10.0, -10.0, Box::new("invalid")));
    assert!(!tree.insert(110.0, 110.0, Box::new("invalid")));
    assert_eq!(tree.len(), 0);
}

#[test]
fn test_quadtree_duplicate() {
    let mut tree = QuadTree::new(0.0, 0.0, 100.0, 100.0).unwrap();
    assert!(tree.insert(10.0, 10.0, Box::new("test1")));
    assert!(tree.insert(10.0, 10.0, Box::new("test2")));
    assert_eq!(tree.len(), 1);
}

#[test]
fn test_quadtree_split() {
    let mut tree = QuadTree::new(0.0, 0.0, 100.0, 100.0).unwrap();
    tree.insert(10.0, 10.0, Box::new("test1"));
    tree.insert(11.0, 11.0, Box::new("test2"));
    assert_eq!(tree.len(), 2);
}