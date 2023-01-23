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
            w * prefactor * ((s / (1.0 - s) + (1.0 - s) / s) * (2.0 * t / 3.0).cosh() + (t / 3.0).cosh() / t.cosh())
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
    let max = if chi < 2.0 {
        spectrum(0.5, chi)
    } else {
        spectrum(0.9 * chi.powf(-0.875), chi)
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

            let prefactor = 1.0 / (3_f64.sqrt() * std::f64::consts::PI * chi);
            let intgd: f64 = GAUSS_32_NODES.iter()
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(t, w)| {
                    let s = 0.5 * (1.0 + t);
                    prefactor * 0.5 * w * spectrum(s, *chi)
                })
                .sum();

            let error = (result - target).abs() / target;
            let intgd_error = (intgd - target).abs() / target;
            println!("chi = {:>9.3e}, t(chi) = {:>12.6e} | {:>12.6e} [interp|intgd], error = {:.3e} | {:.3e}", chi, result, intgd, error, intgd_error);
            assert!(error < max_error);
        }
    }

    #[test]
    #[ignore]
    fn pair_spectrum_sampling() {
        let chi = 2.0;
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
}