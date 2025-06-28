use rand::random;
use std::f64;

/// A point in the 2D space.
#[derive(Debug, Clone)]
pub struct QuadtreePoint {
    pub x: f64,
    pub y: f64,
}

/// A bounding box that defines the area covered by a node.
#[derive(Debug, Clone)]
pub struct QuadtreeBounds {
    pub nw: QuadtreePoint,
    pub se: QuadtreePoint,
    pub width: f64,
    pub height: f64,
}

/// A node in the quadtree, which may either contain a point or be subdivided into quadrants.
#[derive(Debug)]
pub struct QuadtreeNode {
    pub ne: Option<Box<QuadtreeNode>>,
    pub nw: Option<Box<QuadtreeNode>>,
    pub se: Option<Box<QuadtreeNode>>,
    pub sw: Option<Box<QuadtreeNode>>,
    pub bounds: Option<QuadtreeBounds>,
    pub point: Option<QuadtreePoint>,
    pub key: Option<Box<dyn std::any::Any>>,
}

/// The main structure representing the quadtree itself.
#[derive(Debug)]
pub struct Quadtree {
    pub root: QuadtreeNode,
    pub key_free: Option<Box<dyn Fn(&mut dyn std::any::Any)>>,
    pub length: usize,
}

/// Creates a new point.
pub fn quadtree_point_new(x: f64, y: f64) -> QuadtreePoint {
    QuadtreePoint { x, y }
}

/// Creates a new empty bounds object.
pub fn quadtree_bounds_new() -> QuadtreeBounds {
    QuadtreeBounds {
        nw: QuadtreePoint { x: 0.0, y: 0.0 },
        se: QuadtreePoint { x: 0.0, y: 0.0 },
        width: 0.0,
        height: 0.0,
    }
}

/// Extends a bounds object to include a new point.
pub fn quadtree_bounds_extend(bounds: &mut QuadtreeBounds, x: f64, y: f64) {
    if x < bounds.nw.x {
        bounds.nw.x = x;
    }
    if y > bounds.nw.y {
        bounds.nw.y = y;
    }
    if x > bounds.se.x {
        bounds.se.x = x;
    }
    if y < bounds.se.y {
        bounds.se.y = y;
    }
    bounds.width = bounds.se.x - bounds.nw.x;
    bounds.height = bounds.nw.y - bounds.se.y;
}

/// Creates a new empty quadtree node.
pub fn quadtree_node_new() -> QuadtreeNode {
    QuadtreeNode {
        ne: None,
        nw: None,
        se: None,
        sw: None,
        bounds: None,
        point: None,
        key: None,
    }
}

/// Frees the memory used by a quadtree node.
pub fn quadtree_node_free(node: &mut QuadtreeNode, key_free: Option<&dyn Fn(&mut dyn std::any::Any)>) {
    if let Some(key_free_fn) = key_free {
        if let Some(mut key) = node.key.take() {
            key_free_fn(&mut *key);
        }
    }
    node.point.take();
}

/// Checks if a node is a pointer (it has child nodes).
pub fn quadtree_node_ispointer(node: &QuadtreeNode) -> bool {
    node.nw.is_some() || node.ne.is_some() || node.sw.is_some() || node.se.is_some()
}

/// Checks if a node is empty (it does not contain a point).
pub fn quadtree_node_isempty(node: &QuadtreeNode) -> bool {
    node.point.is_none()
}

/// Checks if a node is a leaf (it contains a single point).
pub fn quadtree_node_isleaf(node: &QuadtreeNode) -> bool {
    node.point.is_some() && quadtree_node_ispointer(node) == false
}

/// Resets a node to be empty.
pub fn quadtree_node_reset(node: &mut QuadtreeNode, key_free: Option<&dyn Fn(&mut dyn std::any::Any)>) {
    if let Some(key_free_fn) = key_free {
        if let Some(mut key) = node.key.take() {
            key_free_fn(&mut *key);
        }
    }
    node.point.take();
    node.key.take();
}

/// Creates a new node with specific bounds.
pub fn quadtree_node_with_bounds(minx: f64, miny: f64, maxx: f64, maxy: f64) -> QuadtreeNode {
    let nw = QuadtreePoint { x: minx, y: maxy };
    let se = QuadtreePoint { x: maxx, y: miny };
    QuadtreeNode {
        bounds: Some(QuadtreeBounds { nw, se, width: maxx - minx, height: maxy - miny }),
        point: None,
        key: None,
        ne: None,
        nw: None,
        se: None,
        sw: None,
    }
}

/// Creates a new quadtree with the given bounds.
pub fn quadtree_new(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Option<Quadtree> {
    let root = quadtree_node_with_bounds(minx, miny, maxx, maxy);
    Some(Quadtree {
        root,
        key_free: None,
        length: 0,
    })
}

/// Inserts a point into the quadtree.
pub fn quadtree_insert(tree: &mut Quadtree, x: f64, y: f64, key: Option<Box<dyn std::any::Any>>) -> bool {
    let point = quadtree_point_new(x, y);
    if !node_contains(&tree.root, &point) {
        return false;
    }
    if !insert_(tree, &mut tree.root, &point, key) {
        return false;
    }
    tree.length += 1;
    true
}

/// Searches for a point in the quadtree.
pub fn quadtree_search(tree: &Quadtree, x: f64, y: f64) -> Option<&QuadtreePoint> {
    find_(&tree.root, x, y)
}

/// Frees the memory used by the quadtree.
pub fn quadtree_free(tree: &mut Quadtree) {
    if let Some(key_free_fn) = &tree.key_free {
        quadtree_node_free(&mut tree.root, Some(key_free_fn));
    } else {
        quadtree_node_free(&mut tree.root, None);
    }
}

/// Walks the quadtree in a pre-order traversal, applying descent and ascent functions.
pub fn quadtree_walk<F>(root: &QuadtreeNode, descent: F, ascent: F)
where
    F: Fn(&QuadtreeNode),
{
    descent(root);
    if let Some(ref nw) = root.nw {
        quadtree_walk(nw, &descent, &ascent);
    }
    if let Some(ref ne) = root.ne {
        quadtree_walk(ne, &descent, &ascent);
    }
    if let Some(ref sw) = root.sw {
        quadtree_walk(sw, &descent, &ascent);
    }
    if let Some(ref se) = root.se {
        quadtree_walk(se, &descent, &ascent);
    }
    ascent(root);
}

/// Helper function to check if a node contains a point.
fn node_contains(node: &QuadtreeNode, point: &QuadtreePoint) -> bool {
    if let Some(bounds) = &node.bounds {
        bounds.nw.x < point.x && bounds.nw.y > point.y && bounds.se.x > point.x && bounds.se.y < point.y
    } else {
        false
    }
}

/// Helper function to insert a point into the quadtree.
fn insert_(tree: &mut Quadtree, node: &mut QuadtreeNode, point: &QuadtreePoint, key: Option<Box<dyn std::any::Any>>) -> bool {
    if quadtree_node_isempty(node) {
        node.point = Some(point.clone());
        node.key = key;
        return true;
    } else if quadtree_node_isleaf(node) {
        if let Some(existing_point) = &node.point {
            if existing_point.x == point.x && existing_point.y == point.y {
                quadtree_node_reset(node, tree.key_free.as_ref());
                node.point = Some(point.clone());
                node.key = key;
                return false;
            }
        }
        if !split_node(tree, node) {
            return false;
        }
        return insert_(tree, node, point, key);
    } else if quadtree_node_ispointer(node) {
        if let Some(quadrant) = get_quadrant(node, point) {
            return insert_(tree, quadrant, point, key);
        }
        return false;
    }
    false
}

/// Helper function to check if a node contains the point.
fn get_quadrant<'a>(node: &'a QuadtreeNode, point: &QuadtreePoint) -> Option<&'a mut QuadtreeNode> {
    if node_contains(&node, point) {
        return Some(node);
    }
    None
}

/// Helper function to split a node into quadrants.
fn split_node(tree: &mut Quadtree, node: &mut QuadtreeNode) -> bool {
    let bounds = node.bounds.clone().unwrap();
    let hw = bounds.width / 2.0;
    let hh = bounds.height / 2.0;

    let nw = quadtree_node_with_bounds(bounds.nw.x, bounds.nw.y, bounds.nw.x + hw, bounds.nw.y - hh);
    let ne = quadtree_node_with_bounds(bounds.nw.x + hw, bounds.nw.y, bounds.nw.x + hw * 2.0, bounds.nw.y - hh);
    let sw = quadtree_node_with_bounds(bounds.nw.x, bounds.nw.y - hh * 2.0, bounds.nw.x + hw, bounds.nw.y - hh);
    let se = quadtree_node_with_bounds(bounds.nw.x + hw, bounds.nw.y - hh * 2.0, bounds.nw.x + hw * 2.0, bounds.nw.y - hh);

    node.nw = Some(Box::new(nw));
    node.ne = Some(Box::new(ne));
    node.sw = Some(Box::new(sw));
    node.se = Some(Box::new(se));

    node.point.take();
    node.key.take();

    true
}

/// Helper function to recursively find a point in the tree.
fn find_(node: &QuadtreeNode, x: f64, y: f64) -> Option<&QuadtreePoint> {
    if quadtree_node_isleaf(node) {
        if let Some(point) = &node.point {
            if point.x == x && point.y == y {
                return Some(point);
            }
        }
    } else if let Some(quadrant) = get_quadrant(node, &QuadtreePoint { x, y }) {
        return find_(quadrant, x, y);
    }
    None
}
