//! Custom implementations of special functions not
//! provided by the standard lib.

mod bessel;
mod factorial;
mod airy;
mod double_bessel;

pub use bessel::*;
pub use factorial::*;
pub use airy::*;
pub use double_bessel::*;

const SERIES_MAX_LENGTH: usize = 20;

/// Represents the series expansion of a function
/// in powers of its dependent variable,
/// i.e. f(x) ≈ Σ_i a[i] x^(n[i]).
/// The powers n[i] can either be floating-point numbers
/// or integers.
struct Series<T> {
    a: [f64; SERIES_MAX_LENGTH],
    n: [T; SERIES_MAX_LENGTH],
}

impl Series<i32> {
    /// Returns the value of series expansion at `x`
    #[allow(unused)]
    fn evaluate_at(&self, x: f64) -> f64 {
        self.a.iter()
            .zip(self.n.iter())
            .map(|(a, p)| a * x.powi(*p))
            .sum::<f64>()
    }

    /// Evaluates a series expansion at `x`, including terms up to,
    /// but not including, `x^max`.
    fn evaluate_up_to(&self, x: f64, max: i32) -> f64 {
        self.n.iter()
            .take_while(|&&p| p < max)
            .zip(self.a.iter())
            .map(|(p, a)| a * x.powi(*p))
            .sum::<f64>()
    }
}

impl Series<f64> {
    /// Returns the value of series expansion at `x`
    #[allow(unused)]
    fn evaluate_at(&self, x: f64) -> f64 {
        self.a.iter()
            .zip(self.n.iter())
            .map(|(a, p)| a * x.powf(*p))
            .sum::<f64>()
    }
}