pub fn poly(xs: &[f64], x: f64) -> f64 {
    xs.iter()
        .enumerate()
        .map(|(i, &coeff)| coeff * x.powi(i as i32))
        .sum()
}

pub fn find_zero(xs: &[f64]) -> f64 {
    let mut ans = 0.0;
    let mut value = poly(xs, ans);
    
    while value.abs() > 1e-6 {
        let driv: f64 = xs.iter()
            .enumerate()
            .skip(1)
            .map(|(i, &coeff)| coeff * ans.powi(i as i32 - 1) * i as f64)
            .sum();
            
        ans -= value / driv;
        value = poly(xs, ans);
    }
    
    ans
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_poly() {
        assert_relative_eq!(poly(&[1.0, 2.0], 3.0), 7.0);
        assert_relative_eq!(poly(&[-6.0, 11.0, -6.0, 1.0], 2.0), 0.0);
    }

    #[test]
    fn test_find_zero() {
        assert_relative_eq!(find_zero(&[1.0, 2.0]), -0.5, epsilon = 1e-6);
        assert_relative_eq!(find_zero(&[-6.0, 11.0, -6.0, 1.0]), 1.0, epsilon = 1e-6);
    }
}