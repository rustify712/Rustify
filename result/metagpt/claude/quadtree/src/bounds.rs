use crate::Point;

#[derive(Debug)]
pub struct Bounds {
    pub nw: Point,
    pub se: Point,
    pub width: f64,
    pub height: f64,
}

impl Bounds {
    pub fn new() -> Self {
        Bounds {
            nw: Point::new(0.0, 0.0),
            se: Point::new(0.0, 0.0),
            width: 0.0,
            height: 0.0,
        }
    }

    pub fn extend(&mut self, x: f64, y: f64) {
        if self.width == 0.0 && self.height == 0.0 {
            self.nw = Point::new(x, y);
            self.se = Point::new(x, y);
        } else {
            let nw_x = self.nw.x.min(x);
            let nw_y = self.nw.y.max(y);
            let se_x = self.se.x.max(x);
            let se_y = self.se.y.min(y);
            
            self.nw = Point::new(nw_x, nw_y);
            self.se = Point::new(se_x, se_y);
        }
        
        self.width = self.se.x - self.nw.x;
        self.height = self.nw.y - self.se.y;
    }
}