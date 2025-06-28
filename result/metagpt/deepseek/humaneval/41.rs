pub fn car_race_collision(n: i32) -> i32 {
    n * n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_car_race_collision() {
        assert_eq!(car_race_collision(2), 4);
        assert_eq!(car_race_collision(5), 25);
    }
}