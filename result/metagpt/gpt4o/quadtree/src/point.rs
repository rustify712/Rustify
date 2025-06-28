pub struct QuadtreePoint {
    pub x: f64,
    pub y: f64,
}

impl QuadtreePoint {
    pub fn new(x: f64, y: f64) -> Option<Box<Self>> {
        Some(Box::new(QuadtreePoint { x, y }))
    }

    pub fn free(point: Box<Self>) {
        // Dropping point automatically handles memory deallocation
        drop(point);
    }
}