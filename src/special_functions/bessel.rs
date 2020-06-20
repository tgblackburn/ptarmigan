//! Evaluates Bessel J functions, J_n(x), for real 0 < x < n.

use std::f64::consts;
//use super::factorial::Factorial;
use super::airy::Airy;

pub trait BesselJ {
    /// Evaluates the Bessel J function for integer order
    fn j(&self, n: i32) -> Self;

    /// Evaluates the Bessel J function at given argument
    /// and integer orders n-1, n and n+1
    fn j_pm(&self, n: i32) -> (Self, Self, Self) where Self: Sized;
}

impl BesselJ for f64 {
    fn j(&self, n: i32) -> Self {
        match n {
            0 => j0(*self),
            1 => j1(*self),
            _ => triple_j(n, *self).1
        }
    }

    fn j_pm(&self, n: i32) -> (Self, Self, Self) {
        triple_j(n, *self)
    }
}

fn j0(x: f64) -> f64 {
    if x < 0.1 {
        // Fractional error < 1.0e-9
        1.0 - x * x / 4.0 + x * x * x * x / 64.0
    } else if x < 5.0 {
        // Gauss quadrature, 16 nodes
        let integral: f64 = GAUSS_16_NODES.iter()
            .zip(GAUSS_16_WEIGHTS.iter())
            .map(|(t, w)| w * 0.5 * (-x * (consts::FRAC_PI_2 * (t + 1.0)).sin()).cos())
            .sum();
        integral
    } else if x < 20.0 {
        // Gauss quadrature, 32 nodes
        let integral: f64 = GAUSS_32_NODES.iter()
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(t, w)| w * 0.5 * (-x * (consts::FRAC_PI_2 * (t + 1.0)).sin()).cos())
            .sum();
        integral
    } else if x < 60.0 {
        // Asymptotic expansion
        let prefactor = (2.0 / (consts::PI * x)).sqrt();
        let phase = consts::FRAC_PI_4 - x;
        prefactor * (
            (1.0 - 9.0 / (128.0 * x * x) + 3675.0 / (32768.0 * x * x * x * x) - 0.5725014209747314 * x.powi(-6)) * phase.cos()
            - (1.0 / (8.0 * x) - 75.0 / (1024.0 * x * x * x) + 0.2271080017089844 * x.powi(-5)) * phase.sin()
        )
    } else {
        // Asymptotic expansion
        let prefactor = (2.0 / (consts::PI * x)).sqrt();
        let phase = consts::FRAC_PI_4 - x;
        prefactor * (
            (1.0 - 9.0 / (128.0 * x * x) + 3675.0 / (32768.0 * x * x * x * x)) * phase.cos()
            - (1.0 / (8.0 * x) - 75.0 / (1024.0 * x * x * x)) * phase.sin()
        )
    }
}

fn j1(x: f64) -> f64 {
    if x < 0.1 {
        // Fractional error < 1.0e-9
        (x / 2.0) * (1.0 - x * x / 8.0 + x * x * x * x / 192.0)
    } else if x < 5.0 {
        // Gauss quadrature, 16 nodes
        let integral: f64 = GAUSS_16_NODES.iter()
            .zip(GAUSS_16_WEIGHTS.iter())
            .map(|(t, w)| {
                let s = consts::FRAC_PI_2 * (t + 1.0);
                w * 0.5 * (s - x * s.sin()).cos()
            })
            .sum();
        integral
    } else if x < 20.0 {
        // Gauss quadrature, 32 nodes
        let integral: f64 = GAUSS_32_NODES.iter()
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(t, w)| {
                let s = consts::FRAC_PI_2 * (t + 1.0);
                w * 0.5 * (s - x * s.sin()).cos()
            })
            .sum();
        integral
    } else if x < 60.0 {
        // Asymptotic expansion
        let prefactor = (2.0 / (consts::PI * x)).sqrt();
        let phase = consts::FRAC_PI_4 + x;
        prefactor * (
            -(1.0 + 15.0 / (128.0 * x * x) - 4725.0 / (32768.0 * x * x * x * x) + 0.6765925884246826 * x.powi(-6)) * phase.cos()
            + (3.0 / (8.0 * x) - 105.0 / (1024.0 * x * x * x) + 0.2775764465332031 * x.powi(-5)) * phase.sin()
        )
    } else {
        // Asymptotic expansion
        let prefactor = (2.0 / (consts::PI * x)).sqrt();
        let phase = consts::FRAC_PI_4 + x;
        prefactor * (
            -(1.0 + 15.0 / (128.0 * x * x) - 4725.0 / (32768.0 * x * x * x * x)) * phase.cos()
            + (3.0 / (8.0 * x) - 105.0 / (1024.0 * x * x * x)) * phase.sin()
        )
    }
}

/// Returns a triple of Bessel J functions for real argument `x`,
/// (J_{n-1}(x), J_n(x), J_{n+1}(x)),
/// designed for accuracy on the interval 0 < x < n.
fn triple_j(n: i32, x: f64) -> (f64, f64, f64) {
    if x == 0.0 {
        return (0.0, 0.0, 0.0);
    }

    let v: f64 = x / (n as f64);
    let nmax: i32 = 1 + (n as i32) + ((5.0 / (1.0 - 0.9 * v)) as i32);
    // ensure nmax is even
    let nmax = if nmax.rem_euclid(2) == 0 {nmax} else {nmax + 1};
    let n_even = (n.rem_euclid(2) == 0) as i32;
    // when to stop first loop?
    let k_stop = n - 1 - n_even;

    let mut quad: [f64; 4] = [1.0, 0.0, 0.0, 0.0];
    let mut total = 0.0;

    for k in (k_stop..=nmax).rev().step_by(2) {
        quad[3] = quad[1];
        quad[2] = quad[0];
        quad[1] = ((2 * (k + 2)) as f64) * quad[2] / x - quad[3];
        quad[0] = ((2 * (k + 1)) as f64) * quad[1] / x - quad[2];
        total += quad[0];
    }

    // at this point, quad holds either
    //    [n-1, n, n+1, n+2] if n is odd
    // or [n-2, n-1, n, n+1] if n is even

    let triple = {
        let index = n_even as usize;
        (quad[index], quad[index+1], quad[index+2])
    };

    // now finish the loop
    for k in (0..=k_stop).rev().step_by(2).skip(1) {
        quad[3] = quad[1];
        quad[2] = quad[0];
        quad[1] = ((2 * (k + 2)) as f64) * quad[2] / x - quad[3];
        quad[0] = ((2 * (k + 1)) as f64) * quad[1] / x - quad[2];
        total += quad[0];
    }

    // quad[0] is now equal to J(0, x) * a constant
    let norm = 2.0 * total - quad[0];

    if norm.is_nan() {
        (0.0, 0.0, 0.0)
    } else {
        (triple.0 / norm, triple.1 / norm, triple.2 / norm)
    }
}

fn zeta_at(z: f64) -> f64 {
    if z < 1.0 {
        let tmp = ((1.0 + (1.0 - z * z).sqrt()) / z).ln() - (1.0 - z * z).sqrt();
        (3.0 * tmp / 2.0).powf(2.0 / 3.0)
    } else { // if z >= 1.0 {
        let tmp = (z * z - 1.0).sqrt() - (1.0 / z).acos();
        -(3.0 * tmp / 2.0).powf(2.0 / 3.0)
    }
}

#[allow(unused)]
fn j_asymptotic(n: i32, x: f64) -> f64 {
    let m = n as f64;
    let z = x / m;
    let zeta = zeta_at(z);
    let ai = (m.powf(2.0/3.0) * zeta).ai().unwrap_or(0.0);
    let aip = (m.powf(2.0/3.0) * zeta).ai_prime().unwrap_or(0.0);
    let prefactor = 4.0 * zeta / (1.0 - z * z);
    //let prefactor = if prefactor.is_nan() {4.0 * 2.0f64.cbrt() / (1.0 + z)} else {prefactor};
    let ak = 1.0;
    let bk = 5.0 / (48.0 * zeta * zeta) - (2.0 + 3.0 * z * z) / (24.0 * zeta.sqrt() * (1.0 - z * z).powf(1.5));
    prefactor.powf(0.25) * (ai * ak / m.powf(1.0/3.0) - aip * bk / m.powf(5.0/3.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    static MAX_ERROR: f64 = 1.0e-8;

    #[test]
    #[ignore]
    fn jn() {
        use std::fs::File;
        use std::io::Write;
        let mut file = File::create("output/jn.dat").unwrap();
        for i in 1..1000 {
            let f = (i as f64) / 1000.0;
            let j1 = triple_j(1, 1.0 * f).1;
            let j10 = triple_j(10, 10.0 * f).1;
            let j100 = triple_j(100, 100.0 * f).1;
            writeln!(file, "{:.3e} {:.12e} {:.12e} {:.12e}", f, j1, j10, j100).unwrap();
        }
    }

    #[test]
    fn j70_5() {
        let target = 5.484562820751933e-73;
        let value = 5.0f64.j(70);
        let error = (target - value).abs() / target;
        println!("J(31, {}) = {:.6e}, expected {:.6e}, error {:.3e}", 5.0, value, target, error);
        assert!(error < MAX_ERROR);
    }

    #[test]
    fn j31_30() {
        let target = 0.1023416331626096;
        let value = 30.0f64.j(31);
        let error = (target - value).abs() / target;
        println!("J(31, {}) = {:.6e}, expected {:.6e}, error {:.3e}", 30.0, value, target, error);
        assert!(error < MAX_ERROR);
    }

    #[test]
    fn j24_2() {
        let target = 1.548492695958880e-24;
        let value = 2.0f64.j(24);
        let error = (target - value).abs() / target;
        println!("J(24, {}) = {:.6e}, expected {:.6e}, error {:.3e}", 2.0, value, target, error);
        assert!(error < MAX_ERROR);
    }

    #[test]
    fn j8_2() {
        let target = 0.000022179552287925904088;
        let value = 2.0f64.j(8);
        let error = (target - value).abs() / target;
        println!("J(8, {}) = {:.6e}, expected {:.6e}, error {:.3e}", 2.0, value, target, error);
        assert!(error < MAX_ERROR);
    }

    #[test]
    fn j1_0_87() {
        let target = 0.39512125869620047961;
        let value = 0.87f64.j(1);
        let error = (target - value).abs() / target;
        println!("J(1, {}) = {:.6e}, expected {:.6e}, error {:.3e}", 0.87, value, target, error);
        assert!(error < MAX_ERROR);
    }

    #[test]
    fn j1_0_2() {
        let target = 0.099500832639235995398;
        let value = 0.2f64.j(1);
        let error = (target - value).abs() / target;
        println!("J(1, {}) = {:.6e}, expected {:.6e}, error {:.3e}", 0.2, value, target, error);
        assert!(error < MAX_ERROR);
    }

    #[test]
    fn j0_1_2() {
        let target = 0.6711327442643626735;
        let value = 1.2f64.j(0);
        let error = (target - value).abs() / target;
        println!("J(0, {}) = {:.6e}, expected {:.6e}, error {:.3e}", 1.2, value, target, error);
        assert!(error < MAX_ERROR);
    }

    #[test]
    fn j0_12() {
        let target = 0.047689310796833536624;
        let value = 12.0f64.j(0);
        let error = (target - value).abs() / target;
        println!("J(0, {}) = {:.6e}, expected {:.6e}, error {:.3e}", 12.0, value, target, error);
        assert!(error < MAX_ERROR);
    }

    #[test]
    fn j0_120() {
        let target = 0.0718234158291561276;
        let value = 120.0f64.j(0);
        let error = (target - value).abs() / target;
        println!("J(0, {}) = {:.6e}, expected {:.6e}, error {:.3e}", 120.0, value, target, error);
        assert!(error < MAX_ERROR);
    }

    #[test]
    fn j400_350() {
        let target = 5.912510654628083e-10;
        let value = [triple_j(400, 350.0).1, j_asymptotic(400, 350.0)];
        let error = [(target - value[0]).abs() / target, (target - value[1]).abs() / target];
        println!("recurrence: J(400, 350) = {:.6e} with error {:.6e}", value[0], error[0]);
        println!("asymptotic: J(400, 350) = {:.6e} with error {:.6e}", value[1], error[1]);
        assert!(error[0] < MAX_ERROR);
    }

    #[test]
    fn j400_400() {
        let target = 0.06070867128509718;
        let value = [triple_j(400, 400.0).1, j_asymptotic(400, 400.0)];
        let error = [(target - value[0]).abs() / target, (target - value[1]).abs() / target];
        println!("recurrence: J(400, 400) = {:.6e} with error {:.6e}", value[0], error[0]);
        println!("asymptotic: J(400, 400) = {:.6e} with error {:.6e}", value[1], error[1]);
        assert!(error[0] < MAX_ERROR);
    }

    #[test]
    fn j1000_990() {
        let target = 0.012361942456230178547;
        let value = [triple_j(1000, 990.0).1, j_asymptotic(1000, 990.0)];
        let error = [(target - value[0]).abs() / target, (target - value[1]).abs() / target];
        println!("recurrence: J(1000, 990) = {:.6e} with error {:.6e}", value[0], error[0]);
        println!("asymptotic: J(1000, 990) = {:.6e} with error {:.6e}", value[1], error[1]);
        assert!(error[0] < MAX_ERROR);
    }
}


static GAUSS_16_NODES: [f64; 16] = [
    -9.894009349916499e-1,
    -9.445750230732326e-1,
    -8.656312023878317e-1,
    -7.554044083550030e-1,
    -6.178762444026437e-1,
    -4.580167776572274e-1,
    -2.816035507792589e-1,
    -9.501250983763744e-2,
    9.501250983763744e-2,
    2.816035507792589e-1,
    4.580167776572274e-1,
    6.178762444026437e-1,
    7.554044083550030e-1,
    8.656312023878317e-1,
    9.445750230732326e-1,
    9.894009349916499e-1,
];

static GAUSS_16_WEIGHTS: [f64; 16] = [
    2.715245941175400e-2,
    6.225352393864800e-2,
    9.515851168249300e-2,
    1.246289712555340e-1,
    1.495959888165770e-1,
    1.691565193950025e-1,
    1.826034150449236e-1,
    1.894506104550685e-1,
    1.894506104550685e-1,
    1.826034150449236e-1,
    1.691565193950025e-1,
    1.495959888165770e-1,
    1.246289712555340e-1,
    9.515851168249300e-2,
    6.225352393864800e-2,
    2.715245941175400e-2,
];

static GAUSS_32_NODES: [f64; 32] = [
    -9.972638618494816e-1,
    -9.856115115452683e-1,
    -9.647622555875064e-1,
    -9.349060759377397e-1,
    -8.963211557660521e-1,
    -8.493676137325700e-1,
    -7.944837959679424e-1,
    -7.321821187402897e-1,
    -6.630442669302152e-1,
    -5.877157572407623e-1,
    -5.068999089322294e-1,
    -4.213512761306353e-1,
    -3.318686022821276e-1,
    -2.392873622521371e-1,
    -1.444719615827965e-1,
    -4.830766568773832e-2,
    4.830766568773832e-2,
    1.444719615827965e-1,
    2.392873622521371e-1,
    3.318686022821276e-1,
    4.213512761306353e-1,
    5.068999089322294e-1,
    5.877157572407623e-1,
    6.630442669302152e-1,
    7.321821187402897e-1,
    7.944837959679424e-1,
    8.493676137325700e-1,
    8.963211557660521e-1,
    9.349060759377397e-1,
    9.647622555875064e-1,
    9.856115115452683e-1,
    9.972638618494816e-1,
];

static GAUSS_32_WEIGHTS: [f64; 32] = [
    7.018610000000000e-3,
    1.627439500000000e-2,
    2.539206500000000e-2,
    3.427386300000000e-2,
    4.283589800000000e-2,
    5.099805900000000e-2,
    5.868409350000000e-2,
    6.582222280000000e-2,
    7.234579411000000e-2,
    7.819389578700000e-2,
    8.331192422690000e-2,
    8.765209300440000e-2,
    9.117387869576400e-2,
    9.384439908080460e-2,
    9.563872007927486e-2,
    9.654008851472780e-2,
    9.654008851472780e-2,
    9.563872007927486e-2,
    9.384439908080460e-2,
    9.117387869576400e-2,
    8.765209300440000e-2,
    8.331192422690000e-2,
    7.819389578700000e-2,
    7.234579411000000e-2,
    6.582222280000000e-2,
    5.868409350000000e-2,
    5.099805900000000e-2,
    4.283589800000000e-2,
    3.427386300000000e-2,
    2.539206500000000e-2,
    1.627439500000000e-2,
    7.018610000000000e-3,
];
