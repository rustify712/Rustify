use std::f64::{INFINITY, NEG_INFINITY};
use super::point::Point;

#[derive(Debug)]
pub struct Bounds {
    pub nw: Box<Point>,
    pub se: Box<Point>,
    pub width: f64,
    pub height: f64,
}

impl Bounds {
    pub fn new() -> Self {
        Bounds {
            nw: Box::new(Point {
                x: INFINITY,
                y: NEG_INFINITY,
            }),
            se: Box::new(Point {
                x: NEG_INFINITY,
                y: INFINITY,
            }),
            width: 0.0,
            height: 0.0,
        }
    }

    pub fn extend(&mut self, x: f64, y: f64) {
        self.nw.x = x.min(self.nw.x);
        self.nw.y = y.max(self.nw.y);
        self.se.x = x.max(self.se.x);
        self.se.y = y.min(self.se.y);
        self.width = (self.nw.x - self.se.x).abs();
        self.height = (self.nw.y - self.se.y).abs();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_new() {
        let bounds = Bounds::new();
        assert_eq!(bounds.nw.x, INFINITY);
        assert_eq!(bounds.nw.y, NEG_INFINITY);
        assert_eq!(bounds.se.x, NEG_INFINITY);
        assert_eq!(bounds.se.y, INFINITY);
        assert_eq!(bounds.width, 0.0);
        assert_eq!(bounds.height, 0.0);
    }

    #[test]
    fn test_bounds_extend() {
        let mut bounds = Bounds::new();
        bounds.extend(10.0, 20.0);
        assert_eq!(bounds.nw.x, 10.0);
        assert_eq!(bounds.nw.y, 20.0);
        assert_eq!(bounds.se.x, 10.0);
        assert_eq!(bounds.se.y, 20.0);
        assert_eq!(bounds.width, 0.0);
        assert_eq!(bounds.height, 0.0);

        bounds.extend(5.0, 25.0);
        assert_eq!(bounds.nw.x, 5.0);
        assert_eq!(bounds.nw.y, 25.0);
        assert_eq!(bounds.se.x, 10.0);
        assert_eq!(bounds.se.y, 20.0);
        assert_eq!(bounds.width, 5.0);
        assert_eq!(bounds.height, 5.0);
    }
}