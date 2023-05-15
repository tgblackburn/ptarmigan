use std::ops::{Add, Div, Mul, Sub};

/// Holds a quantity evaluated at S_1 = 1 and -1.
#[derive(Copy, Clone)]
pub struct PolDep {
    inner: [f64; 2]
}

impl PolDep {
    pub fn new(par: f64, perp: f64) -> Self {
        Self { inner: [par, perp] }
    }

    pub fn par(self) -> f64 {
        self.inner[0]
    }

    pub fn with_par(&mut self, par: f64) {
        self.inner[0] = par;
    }

    pub fn perp(self) -> f64 {
        self.inner[1]
    }

    pub fn with_perp(&mut self, perp: f64) {
        self.inner[1] = perp;
    }

    /// Term-wise exponentiation
    pub fn exp(self) -> Self {
        Self { inner: [self.inner[0].exp(), self.inner[1].exp()] }
    }

    pub fn ln(self) -> Self {
        Self { inner: [self.inner[0].ln(), self.inner[1].ln()] }
    }

    pub fn powf(self, n: PolDep) -> Self {
        Self { inner: [self.inner[0].powf(n.inner[0]), self.inner[1].powf(n.inner[1])] }
    }

    /// Value for given Stokes parameter
    pub fn interp(self, sv1: f64) -> f64 {
        let [a, b] = self.inner;
        0.5 * ((a + b) + sv1 * (a - b))
    }

    pub fn into_inner(self) -> [f64; 2] {
        self.inner
    }
}

impl From<[f64; 2]> for PolDep {
    fn from(inner: [f64; 2]) -> Self {
        Self::new(inner[0], inner[1])
    }
}

impl From<f64> for PolDep {
    fn from(n: f64) -> Self {
        Self { inner: [n, n] }
    }
}

impl Add for PolDep {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self { inner: [self.inner[0] + rhs.inner[0], self.inner[1] + rhs.inner[1]] }
    }
}

impl Div<f64> for PolDep {
    type Output = PolDep;
    fn div(self, rhs: f64) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl Div<PolDep> for PolDep {
    type Output = PolDep;
    fn div(self, rhs: PolDep) -> Self::Output {
        Self { inner: [self.inner[0] / rhs.inner[0], self.inner[1] / rhs.inner[1]] }
    }
}

impl Mul<f64> for PolDep {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self { inner: [self.inner[0] * rhs, self.inner[1] * rhs] }
    }
}

impl Mul<PolDep> for f64 {
    type Output = PolDep;
    fn mul(self, rhs: PolDep) -> Self::Output {
        PolDep { inner: [self * rhs.inner[0], self * rhs.inner[1]] }
    }
}

impl Mul<PolDep> for PolDep {
    type Output = PolDep;
    fn mul(self, rhs: PolDep) -> Self::Output {
        PolDep { inner: [self.inner[0] * rhs.inner[0], self.inner[1] * rhs.inner[1]] }
    }
}

impl Sub for PolDep {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self { inner: [self.inner[0] - rhs.inner[0], self.inner[1] - rhs.inner[1]] }
    }
}