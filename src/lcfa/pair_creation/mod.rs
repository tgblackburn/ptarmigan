//! Nonlinear pair creation, gamma -> e- + e+, in a background field

use rand::prelude::*;
use crate::constants::*;

mod tables;

/// Auxiliary function used in calculation of LCFA pair creation
/// rate, defined
/// T(chi) = 1/(6 sqrt3 pi chi) int_1^\infty (8u + 1)/\[u^(3/2) sqrt(u-1)\] K_{2/3}\[8u/(3chi)\] du
fn auxiliary_t(chi: f64) -> f64 {
    use tables::*;
    if chi <= 0.01 {
        // if chi < 5e-3, T(chi) < 1e-117, so ignore
        // 3.0 * 3.0f64.sqrt() / (8.0 * consts::SQRT_2) * (-4.0 / (3.0 * chi)).exp()
        0.0
    } else if chi < 1.0 {
        // use exp(-f/chi) fit
        let i = ((chi.ln() - LN_T_CHI_TABLE[0][0]) / DELTA_LN_CHI) as usize;
        let dx = (chi - LN_T_CHI_TABLE[i][0].exp()) / (LN_T_CHI_TABLE[i+1][0].exp() - LN_T_CHI_TABLE[i][0].exp());
        let tmp = (1.0 - dx) / LN_T_CHI_TABLE[i][1] + dx / LN_T_CHI_TABLE[i+1][1];
        (1.0 / tmp).exp()
    } else if chi < 100.0 {
        // use power-law fit
        let i = ((chi.ln() - LN_T_CHI_TABLE[0][0]) / DELTA_LN_CHI) as usize;
        let dx = (chi.ln() - LN_T_CHI_TABLE[i][0]) / DELTA_LN_CHI;
        ((1.0 - dx) * LN_T_CHI_TABLE[i][1] + dx * LN_T_CHI_TABLE[i+1][1]).exp()
    } else {
        // use asymptotic expression, which is accurate to better than 0.3%
        // for chi > 100:
        //   T(x) = [C - C_1 x^(-2/3)] x^(-1/3)
        // where C = 5 Gamma(5/6) (2/3)^(1/3) / [14 Gamma(7/6)] and C_1 = 2/3
        (0.37961230854357103776 - 2.0 * chi.powf(-2.0/3.0) / 3.0) * chi.powf(-1.0/3.0)
    }
}

/// Returns the nonlinear Breit-Wheeler rate, per unit time (in seconds)
pub fn rate(chi: f64, gamma: f64) -> f64 {
    ALPHA_FINE * chi * auxiliary_t(chi) / (COMPTON_TIME * gamma)
}

/// Proportional to the probability spectrum dW/ds
fn spectrum(s: f64, chi: f64) -> f64 {
    tables::GL_NODES.iter()
        .zip(tables::GL_WEIGHTS.iter())
        .map(|(t, w)| {
            let xi = 2.0 / (3.0 * chi * s * (1.0 - s));
            let prefactor = (-xi * t.cosh() + t).exp();
            w * prefactor * ((s / (1.0 - s) + (1.0 - s) / s) * (1.5 * t).cosh() + (t / 3.0).cosh() / t.cosh())
        })
        .sum()
}

/// Proportional to the angularly resolved spectrum d^2 W/(ds dz),
/// where z^(2/3) = 2ɣ^2(1 - β cosθ).
/// Range is 1 < z < infty, but dominated by 1 < z < 1 + 2 chi
/// Tested and working.
fn angular_spectrum(z: f64, s: f64, chi: f64) -> f64 {
    use crate::special_functions::*;
    let xi = 2.0 / (3.0 * chi * s * (1.0 - s));
    let prefactor = (s * s + (1.0 - s) * (1.0 - s)) / (s * (1.0 - s));
    (1.0 * prefactor * z.powf(2.0 / 3.0)) * (xi * z).bessel_K_1_3().unwrap_or(0.0)
}

/// Samples the positron spectrum of an photon with
/// quantum parameter `chi` and energy (per electron
/// mass) `gamma`, returning the positron Lorentz factor,
/// the cosine of the scattering angle, as well as the
/// equivalent s and z for debugging purposes
pub fn sample<R: Rng>(chi: f64, gamma: f64, rng: &mut R) -> (f64, f64, f64, f64) {
    let max = if chi < 2.5 {
        spectrum(0.5, chi)
    } else {
        spectrum(1.2 / chi, chi)
    };
    let max = 1.2 * max;

    // Rejection sampling for s
    let s = loop {
        let s = rng.gen::<f64>();
        let u = rng.gen::<f64>();
        let f = spectrum(s, chi);
        if u <= f / max {
            break s;
        }
    };

    // Now that s is fixed, sample from the angular spectrum
    // d^2 W/(ds dz), which ranges from 1 < z < infty, or
    // xi/(1+xi) < y < 1 where xi z = y/(1-y)
    let xi = 2.0 / (3.0 * chi * s * (1.0 - s));
    let y_min = xi / (1.0 + xi);
    let max = if y_min > 0.563 {
        let y = y_min;
        let z = y / (xi * (1.0 - y));
        angular_spectrum(z, s, chi) / (xi * (1.0 - y) * (1.0 - y))
    } else {
        let y = 0.563;
        let z = y / (xi * (1.0 - y));
        angular_spectrum(z, s, chi) / (xi * (1.0 - y) * (1.0 - y))
    };
    let max = 1.1 * max;

    // Rejection sampling for z
    let z = if max <= 0.0 {
        0.0
    } else {
        loop {
            let y = y_min + (1.0 - y_min) * rng.gen::<f64>();
            let z = y / (xi * (1.0 - y));
            let u = rng.gen::<f64>();
            let f = angular_spectrum(z, s, chi) / (xi * (1.0 - y) * (1.0 - y));
            if u <= f / max {
                break z;
            }
        }
    };

    // recall z = 2 gamma^2 (1 - beta cos_theta), where
    // beta = sqrt(1 - 1/gamma^2), so cos_theta is close
    // to (2 gamma^2 - z^(2/3)) / (2 gamma^2 - 1)
    // note that gamma here is the positron gamma
    let gamma_p = s * gamma;
    let cos_theta = (2.0 * gamma_p * gamma_p - z.powf(2.0/3.0)) / (2.0 * gamma_p * gamma_p - 1.0);
    let cos_theta = cos_theta.max(-1.0);

    (gamma_p, cos_theta, s, z)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use rand::prelude::*;
    use rand_xoshiro::*;
    use super::*;

    #[test]
    fn lcfa_rate() {
        let max_error = 1.0e-2;

        let pts = [
            (0.042, 6.077538994929929904e-29),
            (0.105, 2.1082097875655204834e-12),
            (0.42,  0.00037796132366581330636),
            (1.05,  0.015977478443872017101),
            (4.2,   0.08917816786414408900),
            (12.0,  0.10884579479913803705),
            (42.0,  0.09266735324318656466),
        ];

        for (chi, target) in &pts {
            let result = auxiliary_t(*chi);
            let error = (result - target).abs() / target;
            //println!("chi = {:.3e}, t(chi) = {:.6e}, error = {:.3e}", chi, result, error);
            assert!(error < max_error);
        }
    }

    #[test]
    #[ignore]
    fn pair_spectrum_sampling() {
        let chi = 1.0;
        let gamma = 1000.0;
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);
        let path = format!("output/lcfa_pair_spectrum_{}.dat", chi);
        let mut file = File::create(path).unwrap();
        for _i in 0..100000 {
            let (_, _, s, z) = sample(chi, gamma, &mut rng);
            assert!(s > 0.0 && s < 1.0);
            assert!(z >= 1.0);
            writeln!(file, "{:.6e} {:.6e}", s, z).unwrap();
        }
    }
}