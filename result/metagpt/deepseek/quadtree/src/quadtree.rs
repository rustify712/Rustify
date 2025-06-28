use std::f64;
use super::{Node, Point};

pub struct QuadTree {
    root: Box<Node>,
    key_free: Option<fn(&mut Box<dyn std::any::Any>)>,
    length: usize,
}

impl QuadTree {
    pub fn new(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Option<Self> {
        let root = Node::with_bounds(minx, miny, maxx, maxy)?;
        Some(QuadTree {
            root: Box::new(root),
            key_free: None,
            length: 0,
        })
    }

    pub fn insert(&mut self, x: f64, y: f64, key: Box<dyn std::any::Any>) -> bool {
        let point = Point::new(x, y);
        if !self.node_contains(&self.root, &point) {
            return false;
        }
        let mut root = unsafe { &mut *(self.root.as_mut() as *mut Node) };
        let result = self.insert_impl(root, point, key);
        if result {
            self.length += 1;
        }
        result
    }

    pub fn search(&self, x: f64, y: f64) -> Option<&Point> {
        self.find(&self.root, x, y)
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn set_key_free(&mut self, free_fn: fn(&mut Box<dyn std::any::Any>)) {
        self.key_free = Some(free_fn);
    }

    fn node_contains(&self, outer: &Node, point: &Point) -> bool {
        outer.bounds.as_ref().map_or(false, |bounds| {
            bounds.nw.x < point.x &&
            bounds.nw.y > point.y &&
            bounds.se.x > point.x &&
            bounds.se.y < point.y
        })
    }

    fn insert_impl(&mut self, root: &mut Node, point: Point, key: Box<dyn std::any::Any>) -> bool {
        if root.is_empty() {
            root.point = Some(point);
            root.key = Some(key);
            true
        } else if root.is_leaf() {
            if let Some(existing) = &root.point {
                if existing.x == point.x && existing.y == point.y {
                    self.reset_node(root);
                    root.point = Some(point);
                    root.key = Some(key);
                    return false;
                }
            }
            if !self.split_node(root) {
                return false;
            }
            self.insert_impl(root, point, key)
        } else if root.is_pointer() {
            if let Some(quadrant) = self.get_quadrant(root, &point) {
                self.insert_impl(quadrant, point, key)
            } else {
                false
            }
        } else {
            false
        }
    }

    fn split_node(&mut self, node: &mut Node) -> bool {
        let bounds = node.bounds.as_ref().unwrap();
        let x = bounds.nw.x;
        let y = bounds.nw.y;
        let hw = bounds.width / 2.0;
        let hh = bounds.height / 2.0;

        let nodes = [
            Node::with_bounds(x, y - hh, x + hw, y),
            Node::with_bounds(x + hw, y - hh, x + hw * 2.0, y),
            Node::with_bounds(x, y - hh * 2.0, x + hw, y - hh),
            Node::with_bounds(x + hw, y - hh * 2.0, x + hw * 2.0, y - hh),
        ];

        if nodes.iter().any(|n| n.is_none()) {
            return false;
        }

        node.nw = Some(Box::new(nodes[0].take().unwrap()));
        node.ne = Some(Box::new(nodes[1].take().unwrap()));
        node.sw = Some(Box::new(nodes[2].take().unwrap()));
        node.se = Some(Box::new(nodes[3].take().unwrap()));

        if let (Some(point), Some(key)) = (node.point.take(), node.key.take()) {
            self.insert_impl(node, point, key)
        } else {
            false
        }
    }

    fn find<'a>(&self, node: &'a Node, x: f64, y: f64) -> Option<&'a Point> {
        if node.is_leaf() {
            node.point.as_ref().filter(|p| p.x == x && p.y == y)
        } else {
            let test_point = Point::new(x, y);
            self.get_quadrant(node, &test_point)
                .and_then(|q| self.find(q, x, y))
        }
    }

    fn get_quadrant<'a>(&self, root: &'a Node, point: &Point) -> Option<&'a Node> {
        [&root.nw, &root.ne, &root.sw, &root.se]
            .iter()
            .find_map(|q| q.as_ref().filter(|n| self.node_contains(n, point)))
            .map(|n| &**n)
    }

    fn reset_node(&mut self, node: &mut Node) {
        if let Some(free_fn) = self.key_free {
            if let Some(key) = &mut node.key {
                free_fn(key);
            }
        }
        node.point = None;
        node.key = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadtree_operations() {
        let mut tree = QuadTree::new(0.0, 0.0, 100.0, 100.0).unwrap();
        assert!(tree.insert(10.0, 10.0, Box::new("test")));
        assert_eq!(tree.len(), 1);
        assert!(tree.search(10.0, 10.0).is_some());
    }
}