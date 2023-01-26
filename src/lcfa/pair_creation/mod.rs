//! Nonlinear pair creation, gamma -> e- + e+, in a background field

use rand::prelude::*;
use crate::constants::*;
use crate::quadrature::{GL_NODES, GL_WEIGHTS};

mod tables;

/// Returns the value of the auxiliary function T for photons that are polarized parallel,
/// and perpendicular to, the instantaneous acceleration (respectively).
fn auxiliary_t(chi: f64) -> (f64, f64) {
    use tables::*;
    if chi <= 0.01 {
        // if chi < 5e-3, T(chi) < 1e-117, so ignore
        // 3.0 * 3.0f64.sqrt() / (8.0 * consts::SQRT_2) * (-4.0 / (3.0 * chi)).exp()
        (0.0, 0.0)
    } else if chi < 1.0 {
        // use exp(-f/chi) fit
        let i = ((chi.ln() - LN_T_CHI_TABLE[0][0]) / DELTA_LN_CHI) as usize;
        let dx = (chi - LN_T_CHI_TABLE[i][0].exp()) / (LN_T_CHI_TABLE[i+1][0].exp() - LN_T_CHI_TABLE[i][0].exp());
        let par = (1.0 - dx) / LN_T_CHI_TABLE[i][1] + dx / LN_T_CHI_TABLE[i+1][1];
        let perp = (1.0 - dx) / LN_T_CHI_TABLE[i][2] + dx / LN_T_CHI_TABLE[i+1][2];
        ((1.0 / par).exp(), (1.0 / perp).exp())
    } else if chi < 100.0 {
        // use power-law fit
        let i = ((chi.ln() - LN_T_CHI_TABLE[0][0]) / DELTA_LN_CHI) as usize;
        let dx = (chi.ln() - LN_T_CHI_TABLE[i][0]) / DELTA_LN_CHI;
        let par = (1.0 - dx) * LN_T_CHI_TABLE[i][1] + dx * LN_T_CHI_TABLE[i+1][1];
        let perp= (1.0 - dx) * LN_T_CHI_TABLE[i][2] + dx * LN_T_CHI_TABLE[i+1][2];
        (par.exp(), perp.exp())
    } else {
        // use asymptotic expression, which is accurate to better than 0.3%
        // for chi > 100:
        //   T(x) = [C - C_1 x^(-2/3)] x^(-1/3)
        // where <C> = 5 Gamma(5/6) (2/3)^(1/3) / [14 Gamma(7/6)] and C_1 = 2/3
        (0.3036898468348568 / chi.cbrt() - 2.0 / (3.0 * chi), 0.4555347702522852 / chi.cbrt() - 2.0 / (3.0 * chi))
    }
}

/// Returns the nonlinear Breit-Wheeler rate, per unit time (in seconds),
/// for a polarized photon with quantum parameter `chi` and normalized energy `gamma`.
///
/// The photon polarization is defined by  `parallel_proj` and `perp_proj`,
/// the projections of the polarization on `w` and `n × w`, respectively,
/// where `w` is the instantaneous acceleration and `n` is the photon direction.
///
/// The polarization-averaged rate is recovered by setting both projections to 0.5.
pub fn rate(chi: f64, gamma: f64, parallel_proj: f64, perp_proj: f64) -> f64 {
    let (t_par, t_perp) = auxiliary_t(chi);
    ALPHA_FINE * chi * (parallel_proj * t_par + perp_proj * t_perp) / (COMPTON_TIME * gamma)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Polarization {
    Parallel,
    Perpendicular,
    None,
}

/// Proportional to the probability spectrum dW/ds for a polarized photon.
///
/// The rate is obtained by integrating [spectrum] over s and multiplying
/// by ɑ m^2 / (√3 π ω).
fn spectrum(s: f64, chi: f64, pol: Polarization) -> f64 {
    let beta = match pol {
        Polarization::Perpendicular => 1.0,
        Polarization::None => 2.0,
        Polarization::Parallel => 3.0,
    };

    GL_NODES.iter()
        .zip(GL_WEIGHTS.iter())
        .map(|(t, w)| {
            let xi = 2.0 / (3.0 * chi * s * (1.0 - s));
            let prefactor = (-xi * t.cosh() + t).exp();
            w * prefactor * ((1.0 / (s * (1.0 - s)) - beta) * (2.0 * t / 3.0).cosh() + (t / 3.0).cosh() / t.cosh())
        })
        .sum()
}

/// Returns the maximum value of [spectrum] for a polarized photon,
/// padded by a small safety margin.
fn spectrum_ceiling(chi: f64, pol: Polarization) -> f64 {
    let (chi_switch, alpha, beta) = match pol {
        Polarization::Parallel => (1.3, 0.55, -0.75),
        Polarization::Perpendicular => (4.0, 1.54, -1.0),
        Polarization::None => (2.5, 0.9, -0.875),
    };

    let max = if chi < chi_switch {
        spectrum(0.5, chi, pol)
    } else if chi > 100.0 {
        spectrum(4.0 / (3.0 * chi), chi, pol)
    } else {
        spectrum(alpha * chi.powf(beta), chi, pol)
    };

    1.05 * max
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
pub fn sample<R: Rng>(chi: f64, gamma: f64, _parallel_proj: f64, _perp_proj: f64, rng: &mut R) -> (f64, f64, f64, f64) {
    let max = spectrum_ceiling(chi, Polarization::None);

    // Rejection sampling for s
    let s = loop {
        let s = rng.gen::<f64>();
        let u = rng.gen::<f64>();
        let f = spectrum(s, chi, Polarization::None);
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
    use crate::quadrature::*;
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
            let (t_par, t_perp) = auxiliary_t(*chi);
            let result = 0.5 * (t_par + t_perp);

            let prefactor = 1.0 / (3_f64.sqrt() * std::f64::consts::PI * chi);
            let intgd: f64 = GAUSS_32_NODES.iter()
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(t, w)| {
                    let s = 0.5 * (1.0 + t);
                    prefactor * 0.5 * w * spectrum(s, *chi, Polarization::None)
                })
                .sum();

            let error = (result - target).abs() / target;
            let intgd_error = (intgd - target).abs() / target;
            println!("chi = {:>9.3e}, t(chi) = {:>12.6e} | {:>12.6e} [interp|intgd], error = {:.3e} | {:.3e}", chi, result, intgd, error, intgd_error);
            assert!(error < max_error);
        }
    }

    #[test]
    fn lcfa_rate_pol_resolved() {
        use Polarization::*;

        let pts = [
            0.15,
            0.75,
            1.2,
            8.7,
            43.0,
            225.0,
        ];

        for chi in &pts {
            let (target_t_par, target_t_perp) = auxiliary_t(*chi);

            let t_par: f64 = GAUSS_32_NODES.iter()
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(t, w)| {
                    let s = 0.5 * (1.0 + t);
                    w * spectrum(s, *chi, Parallel)
                })
                .sum();

            let t_perp: f64 = GAUSS_32_NODES.iter()
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(t, w)| {
                    let s = 0.5 * (1.0 + t);
                    w * spectrum(s, *chi, Perpendicular)
                })
                .sum();

            let target = target_t_par / target_t_perp;
            let result = t_par / t_perp;
            let error = (target - result).abs() / target;
            println!("chi = {:.3e}, expected T_parallel / T_perp = {:.3}, error = {:.3}%", chi, target, 100.0 * error);
            assert!(error < 1.0e-2);
        }
    }

    #[test]
    fn pair_spectrum_ceiling() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        for _i in 0..100 {
            let chi = (0.1_f64.ln() + (100_f64.ln() - 0.1_f64.ln()) * rng.gen::<f64>()).exp();

            let pol = match rng.gen_range(0, 3) {
                0 => Polarization::Parallel,
                1 => Polarization::Perpendicular,
                2 => Polarization::None,
                _ => unreachable!(),
            };

            let target: f64 = (0..10_000)
                .map(|i| 0.5 * (i as f64) / 10000.0)
                .map(|s| spectrum(s, chi, pol))
                .reduce(f64::max)
                .unwrap();

            let result = spectrum_ceiling(chi, pol);

            let err = (target - result) / target;

            let pol = match pol {
                Polarization::Parallel => "∥",
                Polarization::Perpendicular => "⟂",
                Polarization::None => "x",
            };

            println!(
                "chi = {:>9.3e}, pol = {} => max = {:>9.3e}, predicted = {:>9.3e}, err = {:.2}%",
                chi, pol, target, result, 100.0 * err,
            );

            assert!(err < 0.0);
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
            let (_, _, s, z) = sample(chi, gamma, 0.5, 0.5, &mut rng);
            assert!(s > 0.0 && s < 1.0);
            assert!(z >= 1.0);
            writeln!(file, "{:.6e} {:.6e}", s, z).unwrap();
        }
    }
}