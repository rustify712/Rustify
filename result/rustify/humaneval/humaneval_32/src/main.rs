/// Evaluates polynomial with coefficients `xs` at point `x`.
/// Returns `xs[0] + xs[1] * x + xs[2] * x^2 + ... + xs[n] * x^n`.
fn poly(xs: &[f64], x: f64) -> f64 {
    xs.iter().enumerate().fold(0.0, |sum, (i, &coeff)| sum + coeff * x.powi(i as i32))
}

/// Finds a root of the polynomial with coefficients `xs` using Newton's method.
/// Returns `Some(x)` if a root is found, otherwise `None`.
fn find_zero(xs: &[f64]) -> Option<f64> {
    let mut ans = 0.0;
    let mut value = poly(xs, ans);
    while value.abs() > 1e-6 {
        let driv: f64 = xs.iter().enumerate().skip(1).fold(0.0, |sum, (i, &coeff)| sum + coeff * ans.powi((i - 1) as i32) * i as f64);
        if driv == 0.0 {
            return None;
        }
        ans = ans - value / driv;
        value = poly(xs, ans);
    }
    Some(ans)
}