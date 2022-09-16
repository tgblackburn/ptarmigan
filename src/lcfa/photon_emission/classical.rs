//! Classical synchrotron emission, e -> e + gamma, in a background field

use std::f64::consts;
use crate::constants::*;
use crate::pwmci;
use super::tables;

/// Returns the classical synchrotron rate, per unit time (in seconds)
pub fn rate(chi: f64, gamma: f64) -> f64 {
    let h = 5.0 * consts::FRAC_PI_3;
    3.0f64.sqrt() * ALPHA_FINE * chi * h / (2.0 * consts::PI * gamma * COMPTON_TIME)
}

/// Samples the classical synchrotron spectrum of an electron with
/// quantum parameter `chi` and Lorentz factor `gamma`.
/// 
/// Returns a triple of the photon energy, in units of mc^2,
/// and the polar and azimuthal angles of emission, in the range
/// [0,pi] and [0,2pi] respectively.
/// 
/// As there is no hbar-dependent cutoff, the energy of the photon
/// can exceed that of the electron.
pub fn sample(chi: f64, gamma: f64, rand1: f64, rand2: f64, rand3: f64) -> (f64, Option<f64>, f64) {
    // First determine z:
    // z^(1/3) = (2 + 4 cos(delta/3)) / (5 (1-r)) where 0 <= r < 1
    // and cos(delta) = (-9 + 50r - 25r^2) / 16
    let delta = ((-9.0 + 50.0 * rand2 - 25.0 * rand2.powi(2)) / 16.0).acos();
    let z = ((2.0 + 4.0 * (delta/3.0).cos()) / (5.0 * (1.0 - rand2))).powi(3);

    // now invert cdf(u|z) = (3/pi) \int_0^x t K_{1/3}(t) dt,
    // which is tabulated, to obtain x = 2 u z / (3 chi)
    // for x < 0.01, cdf(u|z) =

    let ln_rand = rand1.ln();
    let x = if ln_rand < tables::CLASSICAL_SPECTRUM_TABLE[0][1] {
        1.020377255 * rand1.powf(0.6)
    } else {
        //println!("Inverting ln(rand = {}) = {}", rand1, ln_rand);
        let (ln_x, _) = pwmci::invert(ln_rand, &tables::CLASSICAL_SPECTRUM_TABLE)
            .unwrap_or( (tables::CLASSICAL_SPECTRUM_TABLE.last().unwrap()[0],1) );
        ln_x.exp()
    };

    let u = 3.0 * chi * x / (2.0 * z);
    let omega_mc2 = u * gamma;

    let cos_theta = (gamma - z.powf(2.0/3.0) / (2.0 * gamma)) / (gamma.powi(2) - 1.0).sqrt();
    let theta = if cos_theta >= 1.0 {
        Some(0.0)
    } else if cos_theta >= -1.0 {
        Some(cos_theta.acos())
    } else {
        None
    };

    (omega_mc2, theta, 2.0 * consts::PI * rand3)
}