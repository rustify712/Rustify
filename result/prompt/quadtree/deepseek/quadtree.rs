// quadtree.rs

use std::ptr;

const QUADTREE_VERSION: &str = "0.1.0";

#[derive(Debug)]
struct QuadtreePoint {
    x: f64,
    y: f64,
}

impl QuadtreePoint {
    fn new(x: f64, y: f64) -> Self {
        QuadtreePoint { x, y }
    }
}

#[derive(Debug)]
struct QuadtreeBounds {
    nw: Box<QuadtreePoint>,
    se: Box<QuadtreePoint>,
    width: f64,
    height: f64,
}

impl QuadtreeBounds {
    fn new() -> Self {
        QuadtreeBounds {
            nw: Box::new(QuadtreePoint::new(0.0, 0.0)),
            se: Box::new(QuadtreePoint::new(0.0, 0.0)),
            width: 0.0,
            height: 0.0,
        }
    }

    fn extend(&mut self, x: f64, y: f64) {
        if x < self.nw.x {
            self.nw.x = x;
        }
        if y > self.nw.y {
            self.nw.y = y;
        }
        if x > self.se.x {
            self.se.x = x;
        }
        if y < self.se.y {
            self.se.y = y;
        }
        self.width = self.se.x - self.nw.x;
        self.height = self.nw.y - self.se.y;
    }
}

#[derive(Debug)]
struct QuadtreeNode {
    ne: Option<Box<QuadtreeNode>>,
    nw: Option<Box<QuadtreeNode>>,
    se: Option<Box<QuadtreeNode>>,
    sw: Option<Box<QuadtreeNode>>,
    bounds: Option<Box<QuadtreeBounds>>,
    point: Option<Box<QuadtreePoint>>,
    key: Option<Box<dyn std::any::Any>>,
}

impl QuadtreeNode {
    fn new() -> Self {
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

    fn is_pointer(&self) -> bool {
        self.ne.is_some() || self.nw.is_some() || self.se.is_some() || self.sw.is_some()
    }

    fn is_empty(&self) -> bool {
        self.point.is_none()
    }

    fn is_leaf(&self) -> bool {
        self.ne.is_none() && self.nw.is_none() && self.se.is_none() && self.sw.is_none()
    }

    fn reset(&mut self, key_free: Option<fn(&mut Box<dyn std::any::Any>)>) {
        if let Some(key) = self.key.take() {
            if let Some(free_fn) = key_free {
                free_fn(Box::new(key));
            }
        }
        self.point = None;
    }

    fn with_bounds(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Option<Box<Self>> {
        let mut node = QuadtreeNode::new();
        let bounds = QuadtreeBounds {
            nw: Box::new(QuadtreePoint::new(minx, maxy)),
            se: Box::new(QuadtreePoint::new(maxx, miny)),
            width: maxx - minx,
            height: maxy - miny,
        };
        node.bounds = Some(Box::new(bounds));
        Some(Box::new(node))
    }
}

#[derive(Debug)]
struct Quadtree {
    root: Option<Box<QuadtreeNode>>,
    key_free: Option<fn(&mut Box<dyn std::any::Any>)>,
    length: usize,
}

impl Quadtree {
    fn new(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Option<Box<Self>> {
        let root = QuadtreeNode::with_bounds(minx, miny, maxx, maxy)?;
        Some(Box::new(Quadtree {
            root: Some(root),
            key_free: None,
            length: 0,
        }))
    }

    fn insert(&mut self, x: f64, y: f64, key: Box<dyn std::any::Any>) -> bool {
        let point = QuadtreePoint::new(x, y);
        if let Some(root) = &mut self.root {
            if !node_contains(root, &point) {
                return false;
            }
            if insert(self, root, point, key) {
                self.length += 1;
                return true;
            }
        }
        false
    }

    fn search(&self, x: f64, y: f64) -> Option<&QuadtreePoint> {
        if let Some(root) = &self.root {
            return find(root, x, y);
        }
        None
    }

    fn free(&mut self) {
        if let Some(root) = self.root.take() {
            if let Some(key_free) = self.key_free {
                node_free(root, key_free);
            } else {
                node_free(root, |_| {});
            }
        }
    }

    fn walk(&self, descent: fn(&QuadtreeNode), ascent: fn(&QuadtreeNode)) {
        if let Some(root) = &self.root {
            walk_node(root, descent, ascent);
        }
    }
}

fn node_contains(node: &QuadtreeNode, point: &QuadtreePoint) -> bool {
    if let Some(bounds) = &node.bounds {
        return bounds.nw.x < point.x
            && bounds.nw.y > point.y
            && bounds.se.x > point.x
            && bounds.se.y < point.y;
    }
    false
}

fn get_quadrant(root: &QuadtreeNode, point: &QuadtreePoint) -> Option<&QuadtreeNode> {
    if node_contains(root.nw.as_ref()?, point) {
        return root.nw.as_ref();
    }
    if node_contains(root.ne.as_ref()?, point) {
        return root.ne.as_ref();
    }
    if node_contains(root.sw.as_ref()?, point) {
        return root.sw.as_ref();
    }
    if node_contains(root.se.as_ref()?, point) {
        return root.se.as_ref();
    }
    None
}

fn split_node(tree: &mut Quadtree, node: &mut QuadtreeNode) -> bool {
    let bounds = node.bounds.as_ref()?;
    let x = bounds.nw.x;
    let y = bounds.nw.y;
    let hw = bounds.width / 2.0;
    let hh = bounds.height / 2.0;

    let nw = QuadtreeNode::with_bounds(x, y - hh, x + hw, y)?;
    let ne = QuadtreeNode::with_bounds(x + hw, y - hh, x + hw * 2.0, y)?;
    let sw = QuadtreeNode::with_bounds(x, y - hh * 2.0, x + hw, y - hh)?;
    let se = QuadtreeNode::with_bounds(x + hw, y - hh * 2.0, x + hw * 2.0, y - hh)?;

    node.nw = Some(nw);
    node.ne = Some(ne);
    node.sw = Some(sw);
    node.se = Some(se);

    let old_point = node.point.take();
    let old_key = node.key.take();

    if let Some(point) = old_point {
        if let Some(key) = old_key {
            return insert(tree, node, *point, key);
        }
    }
    false
}

fn find(node: &QuadtreeNode, x: f64, y: f64) -> Option<&QuadtreePoint> {
    if node.is_leaf() {
        if let Some(point) = &node.point {
            if point.x == x && point.y == y {
                return Some(point);
            }
        }
    } else {
        let test_point = QuadtreePoint::new(x, y);
        if let Some(quadrant) = get_quadrant(node, &test_point) {
            return find(quadrant, x, y);
        }
    }
    None
}

fn insert(tree: &mut Quadtree, root: &mut QuadtreeNode, point: QuadtreePoint, key: Box<dyn std::any::Any>) -> bool {
    if root.is_empty() {
        root.point = Some(Box::new(point));
        root.key = Some(key);
        true
    } else if root.is_leaf() {
        if let Some(root_point) = &root.point {
            if root_point.x == point.x && root_point.y == point.y {
                root.reset(tree.key_free);
                root.point = Some(Box::new(point));
                root.key = Some(key);
                return true;
            }
        }
        if !split_node(tree, root) {
            return false;
        }
        insert(tree, root, point, key)
    } else if root.is_pointer() {
        if let Some(quadrant) = get_quadrant(root, &point) {
            return insert(tree, unsafe { &mut *(quadrant as *const QuadtreeNode as *mut QuadtreeNode) }, point, key);
        }
        false
    } else {
        false
    }
}

fn node_free(node: Box<QuadtreeNode>, key_free: fn(&mut Box<dyn std::any::Any>)) {
    if let Some(key) = node.key {
        key_free(Box::new(key));
    }
    if let Some(nw) = node.nw {
        node_free(nw, key_free);
    }
    if let Some(ne) = node.ne {
        node_free(ne, key_free);
    }
    if let Some(sw) = node.sw {
        node_free(sw, key_free);
    }
    if let Some(se) = node.se {
        node_free(se, key_free);
    }
}

fn walk_node(node: &QuadtreeNode, descent: fn(&QuadtreeNode), ascent: fn(&QuadtreeNode)) {
    descent(node);
    if let Some(nw) = &node.nw {
        walk_node(nw, descent, ascent);
    }
    if let Some(ne) = &node.ne {
        walk_node(ne, descent, ascent);
    }
    if let Some(sw) = &node.sw {
        walk_node(sw, descent, ascent);
    }
    if let Some(se) = &node.se {
        walk_node(se, descent, ascent);
    }
    ascent(node);
}