use std::f64::INFINITY;

#[derive(Debug, Clone)]
pub struct Point {
    x: f64,
    y: f64,
}

#[derive(Debug)]
pub struct Bounds {
    nw: Point,
    se: Point,
    width: f64,
    height: f64,
}

#[derive(Debug)]
pub struct Node<T> {
    ne: Option<Box<Node<T>>>,
    nw: Option<Box<Node<T>>>,
    se: Option<Box<Node<T>>>,
    sw: Option<Box<Node<T>>>,
    bounds: Option<Bounds>,
    point: Option<Point>,
    key: Option<T>,
}

#[derive(Debug)]
pub struct QuadTree<T> {
    root: Node<T>,
    length: usize,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
}

impl Bounds {
    pub fn new() -> Self {
        Bounds {
            nw: Point::new(INFINITY, -INFINITY),
            se: Point::new(-INFINITY, INFINITY),
            width: 0.0,
            height: 0.0,
        }
    }

    pub fn extend(&mut self, x: f64, y: f64) {
        self.nw.x = self.nw.x.min(x);
        self.nw.y = self.nw.y.max(y);
        self.se.x = self.se.x.max(x);
        self.se.y = self.se.y.min(y);
        self.width = (self.nw.x - self.se.x).abs();
        self.height = (self.nw.y - self.se.y).abs();
    }
}

impl<T> Node<T> {
    pub fn new() -> Self {
        Node {
            ne: None,
            nw: None,
            se: None,
            sw: None,
            bounds: None,
            point: None,
            key: None,
        }
    }

    pub fn with_bounds(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Self {
        let mut node = Node::new();
        let mut bounds = Bounds::new();
        bounds.extend(maxx, maxy);
        bounds.extend(minx, miny);
        node.bounds = Some(bounds);
        node
    }

    fn is_pointer(&self) -> bool {
        self.nw.is_some() && self.ne.is_some() && self.sw.is_some() && self.se.is_some() && !self.is_leaf()
    }

    fn is_empty(&self) -> bool {
        self.nw.is_none() && self.ne.is_none() && self.sw.is_none() && self.se.is_none() && !self.is_leaf()
    }

    fn is_leaf(&self) -> bool {
        self.point.is_some()
    }

    fn contains(&self, point: &Point) -> bool {
        if let Some(bounds) = &self.bounds {
            bounds.nw.x < point.x && bounds.nw.y > point.y && bounds.se.x > point.x && bounds.se.y < point.y
        } else {
            false
        }
    }

    fn get_quadrant_mut<'a>(&'a mut self, point: &Point) -> Option<&'a mut Node<T>> {
        if let Some(ref mut nw) = self.nw { if nw.contains(point) { return Some(nw); } }
        if let Some(ref mut ne) = self.ne { if ne.contains(point) { return Some(ne); } }
        if let Some(ref mut sw) = self.sw { if sw.contains(point) { return Some(sw); } }
        if let Some(ref mut se) = self.se { if se.contains(point) { return Some(se); } }
        None
    }

    fn get_quadrant<'a>(&'a self, point: &Point) -> Option<&'a Node<T>> {
        if let Some(ref nw) = self.nw { if nw.contains(point) { return Some(nw); } }
        if let Some(ref ne) = self.ne { if ne.contains(point) { return Some(ne); } }
        if let Some(ref sw) = self.sw { if sw.contains(point) { return Some(sw); } }
        if let Some(ref se) = self.se { if se.contains(point) { return Some(se); } }
        None
    }

    fn split(&mut self) -> bool {
        if let Some(bounds) = &self.bounds {
            let x = bounds.nw.x;
            let y = bounds.nw.y;
            let hw = bounds.width / 2.0;
            let hh = bounds.height / 2.0;

            self.nw = Some(Box::new(Node::with_bounds(x, y - hh, x + hw, y)));
            self.ne = Some(Box::new(Node::with_bounds(x + hw, y - hh, x + hw * 2.0, y)));
            self.sw = Some(Box::new(Node::with_bounds(x, y - hh * 2.0, x + hw, y - hh)));
            self.se = Some(Box::new(Node::with_bounds(x + hw, y - hh * 2.0, x + hw * 2.0, y - hh)));
            true
        } else {
            false
        }
    }
}

impl<T> QuadTree<T> {
    pub fn new(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Self {
        QuadTree {
            root: Node::with_bounds(minx, miny, maxx, maxy),
            length: 0,
        }
    }

    pub fn insert(&mut self, x: f64, y: f64, key: T) -> bool {
        let point = Point::new(x, y);
        if !self.root.contains(&point) {
            return false;
        }

        self.insert_internal(&point, key)
    }

    fn insert_internal(&mut self, point: &Point, key: T) -> bool {
        let mut current = &mut self.root;

        loop {
            if current.is_empty() {
                current.point = Some(point.clone());
                current.key = Some(key);
                self.length += 1;  // 每次成功插入时更新长度
                println!("Inserted point: ({}, {}), tree length: {}", point.x, point.y, self.length);
                return true;
            } else if current.is_leaf() {
                if let Some(ref existing_point) = current.point {
                    if existing_point.x == point.x && existing_point.y == point.y {
                        current.point = Some(point.clone());
                        current.key = Some(key);
                        println!("Updated point: ({}, {}), tree length: {}", point.x, point.y, self.length);
                        return true;
                    } else {
                        // 如果点不相等，尝试分裂
                        let old_point = current.point.take();
                        let old_key = current.key.take();

                        if !current.split() {
                            current.point = old_point;
                            current.key = old_key;
                            return false;
                        }

                        if let (Some(old_point), Some(old_key)) = (old_point, old_key) {
                            if let Some(quadrant) = current.get_quadrant_mut(&old_point) {
                                quadrant.point = Some(old_point);
                                quadrant.key = Some(old_key);
                            }
                        }

                        println!("Splitting node, tree length: {}", self.length);
                        continue;  // 重新递归
                    }
                }
            } else if current.is_pointer() {
                if let Some(next) = current.get_quadrant_mut(point) {
                    current = next;
                } else {
                    return false;
                }
            }
        }
    }



    pub fn search<'a>(&'a self, x: f64, y: f64) -> Option<&'a Point> {
        self.find_in_node(&self.root, x, y)
    }

    fn find_in_node<'a>(&self, node: &'a Node<T>, x: f64, y: f64) -> Option<&'a Point> {
        if node.is_leaf() {
            if let Some(point) = &node.point {
                if point.x == x && point.y == y {
                    return Some(point);
                }
            }
        } else {
            let test_point = Point::new(x, y);
            if let Some(quadrant) = node.get_quadrant(&test_point) {
                return self.find_in_node(quadrant, x, y);
            }
        }
        None
    }

    pub fn walk<F, G>(&self, descent: F, ascent: G)
    where
        F: Fn(&Node<T>),
        G: Fn(&Node<T>),
    {
        self.walk_node(&self.root, &descent, &ascent);
    }

    fn walk_node<F, G>(&self, node: &Node<T>, descent: &F, ascent: &G)
    where
        F: Fn(&Node<T>),
        G: Fn(&Node<T>),
    {
        descent(node);
        if let Some(ref nw) = node.nw { self.walk_node(nw, descent, ascent); }
        if let Some(ref ne) = node.ne { self.walk_node(ne, descent, ascent); }
        if let Some(ref sw) = node.sw { self.walk_node(sw, descent, ascent); }
        if let Some(ref se) = node.se { self.walk_node(se, descent, ascent); }
        ascent(node);
    }
}
