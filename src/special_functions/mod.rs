//! Custom implementations of special functions not
//! provided by the standard lib.

mod bessel;
mod factorial;
mod airy;

pub use bessel::*;
pub use factorial::*;

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
        self.evaluate_up_to(x, SERIES_MAX_LENGTH)
    }

    fn evaluate_up_to(&self, x: f64, max: usize) -> f64 {
        self.a.iter()
            .zip(self.n.iter())
            .take(max)
            .map(|(a, p)| a * x.powi(*p))
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