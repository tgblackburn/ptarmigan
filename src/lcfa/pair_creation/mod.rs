//! Nonlinear pair creation, gamma -> e- + e+, in a background field

use std::f64::consts;
use rand::prelude::*;
use crate::constants::*;
use crate::geometry::*;
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

/// Returns the nonlinear Breit-Wheeler probability
/// for a photon with four-momentum `ell` and Stokes vector `sv` in a
/// constant, crossed field.
///
/// The field is defined by the transverse acceleration `a_perp`, quantum
/// parameter `chi`, and duration `dt` (in seconds).
pub fn probability(ell: FourVector, sv: StokesVector, chi: f64, a_perp: ThreeVector, dt: f64) -> (f64, StokesVector) {
    let (sv, cos_2theta, sin_2theta) = sv.in_basis(a_perp, ell.into());
    let parallel_proj = 0.5 * (1.0 + sv[1]);
    let perp_proj = 0.5 * (1.0 - sv[1]);
    let (t_par, t_perp) = auxiliary_t(chi);
    let prob = ALPHA_FINE * chi * (parallel_proj * t_par + perp_proj * t_perp) * dt / (COMPTON_TIME * ell[0]);

    // If pair creation does not occur, photon Stokes parameters should be changed to:
    let sv: StokesVector = {
        let prob_avg = ALPHA_FINE * chi * 0.5 * (t_par + t_perp) * dt / (COMPTON_TIME * ell[0]);
        let delta = ALPHA_FINE * chi * 0.5 * (t_par - t_perp) * dt / (COMPTON_TIME * ell[0]);

        let sv1 = (sv[1] * (1.0 - prob_avg) - delta) / (1.0 - prob);
        let sv2 = sv[2] * (1.0 - prob_avg) / (1.0 - prob);

        // transform to simulation basis, rotating by -theta
        [
            sv[0],
            cos_2theta * sv1 + sin_2theta * sv2,
            -sin_2theta * sv1 + cos_2theta * sv2,
            sv[3] * (1.0 - prob_avg) / (1.0 - prob),
        ].into()
    };

    (prob, sv)
}

/// Proportional to the probability spectrum dW/ds for a photon
/// with quantum parameter `chi` and Stokes parameter `sv1`.
///
/// The rate is obtained by integrating [spectrum] over s and multiplying
/// by ɑ m^2 / (√3 π ω).
fn spectrum(s: f64, chi: f64, sv1: f64) -> f64 {
    GL_NODES.iter()
        .zip(GL_WEIGHTS.iter())
        .map(|(t, w)| {
            let xi = 2.0 / (3.0 * chi * s * (1.0 - s));
            let prefactor = (-xi * t.cosh() + t).exp();
            w * prefactor * ((1.0 / (s * (1.0 - s)) - sv1 - 2.0) * (2.0 * t / 3.0).cosh() + (t / 3.0).cosh() / t.cosh())
        })
        .sum()
}

/// Returns the maximum value of [spectrum] for a polarized photon,
/// padded by a small safety margin.
fn spectrum_ceiling(chi: f64, sv1: f64) -> f64 {
    let chi_switch = ((1.5 - sv1) / 3_f64.sqrt()).exp();

    let max = if chi < chi_switch {
        spectrum(0.5, chi, sv1)
    } else if chi > 100.0 {
        spectrum(4.0 / (3.0 * chi), chi, sv1)
    } else {
        let m = -0.94866 + 0.170159 * sv1;
        let s = 0.5 * chi_switch.powf(-m) * chi.powf(m);
        // println!("\tchi_switch = {:.3}, m = {:.3}, s = {:.3}", chi_switch, m, s);
        spectrum(s, chi, sv1)
    };

    1.05 * max
}

/// Proportional to the angularly resolved spectrum d^2 W/(ds dy),
/// where z = [2ɣ^2(1 - β cosθ)]^(3/2) = 1 + 4 chi y^2.
/// The domain of interest is 0 < y < 1.
fn angular_spectrum(y: f64, s: f64, chi: f64, sv1: f64) -> f64 {
    // The spectrum is given by
    // dW/(ds dy) = y [1 + z^(2/3) (s/(1-s) + (1-s)/s - sv1)] K_{1/3}(xi z)
    // where z = 1 + 4 chi y^2, xi = 2 / [3 chi s (1-s)]
    // In principle, 1 < z < infty, but dominated by 1 < z < 1 + 4 chi
    use crate::special_functions::*;
    let xi = 2.0 / (3.0 * chi * s * (1.0 - s));
    let prefactor = s / (1.0 - s) + (1.0 - s) / s - sv1;
    let z = 1.0 + 4.0 * chi * y * y;
    y * (1.0 + prefactor * z.powf(2.0 / 3.0)) * (xi * z).bessel_K_1_3().unwrap_or(0.0)
}

/// Returns the maximum value of [angular_spectrum] for a polarized photon,
/// padded by a small safety margin.
fn angular_spectrum_ceiling(s: f64, chi: f64, sv1: f64) -> f64 {
    // y that maximises the spectrum, assuming s = 1/2:
    let y_peak = {
        let y_min = 0.216;
        let y_max = 0.259;
        y_min + (y_max - y_min) * (1.0 - (8.3 / chi).powf(2.0/3.0).tanh())
    };

    // and for general s:
    let y = y_peak * (1.0 - (1.0 - 2.0 * s).powi(2)).sqrt();

    1.05 * angular_spectrum(y, s, chi, sv1)
}

fn sample_azimuthal_angle<R: Rng>(s: f64, z: f64, chi: f64, sv: StokesVector, rng: &mut R) -> f64 {
    let arg = 2.0 * z / (3.0 * chi * s * (1.0 - s));
    // ratio of K_{2/3}(arg) / K_{1/3}(arg)
    let k_ratio = if arg < 1.0e-4 {
        0.6368498843179743 / arg.cbrt()
    } else {
        1.0 + 1.0 / (1.4624087952220928 * arg.cbrt() + 1.023821552056939 * arg.sqrt() + 6.0 * arg)
    };
    let a = 1.0 + z.powf(2.0/3.0) * (s.powi(2) + (1.0 - s).powi(2)) / (s * (1.0 - s));
    let b = 1.0;
    let c = z.powf(2.0/3.0);
    let d = z.powf(2.0/3.0) - 1.0;
    let e = z.powf(1.0/3.0) * (z.powf(2.0/3.0) - 1.0) * (s.powi(2) + (1.0 - s).powi(2)) * k_ratio / (s * (1.0 - s));

    fn azimuthal_spectrum(phi: f64, a: f64, b: f64, c: f64, d: f64, e: f64, sv: StokesVector) -> f64 {
        a + b * ((2.0 * phi).cos() * (1.0 - c) - c) * sv[1] - d * (2.0 * phi).sin() * sv[2] + e * phi.sin() * sv[3]
    }

    let max = (0..32)
        .map(|i| azimuthal_spectrum(2.0 * consts::PI * (i as f64) / 32.0, a, b, c, d, e, sv))
        .reduce(f64::max)
        .map(|y| 1.1 * y)
        .unwrap();

    loop {
        let phi = 2.0 * consts::PI * rng.gen::<f64>();
        let u = rng.gen::<f64>();
        let f = azimuthal_spectrum(phi, a, b, c, d, e, sv);
        if u <= f / max {
            break phi;
        }
    }
}

/// Samples the positron spectrum of an photon with
/// quantum parameter `chi` and energy (per electron
/// mass) `gamma`, returning the positron Lorentz factor,
/// the cosine of the scattering angle, as well as the
/// equivalent s and z for debugging purposes
pub fn sample<R: Rng>(ell: FourVector, sv: StokesVector, chi: f64, a_perp: ThreeVector, rng: &mut R) -> (f64, f64, f64, f64, f64) {
    let gamma = ell[0];
    let (sv, _, _) = sv.in_basis(a_perp, ell.into());

    // Rejection sampling for s
    let max = spectrum_ceiling(chi, sv[1]);
    let s = loop {
        let s = rng.gen::<f64>();
        let u = rng.gen::<f64>();
        let f = spectrum(s, chi, sv[1]);
        if u <= f / max {
            break s;
        }
    };

    // Now that s is fixed, sample from the angular spectrum
    let max = angular_spectrum_ceiling(s, chi, sv[1]);
    let z = loop {
        let y = rng.gen::<f64>();
        let u = rng.gen::<f64>();
        let f = angular_spectrum(y, s, chi, sv[1]);
        if u <= f / max {
            break 1.0 + 4.0 * chi * y * y;
        }
    };

    let phi = sample_azimuthal_angle(s, z, chi, sv, rng);

    // recall z = 2 gamma^2 (1 - beta cos_theta), where
    // beta = sqrt(1 - 1/gamma^2), so cos_theta is close
    // to (2 gamma^2 - z^(2/3)) / (2 gamma^2 - 1)
    // note that gamma here is the positron gamma
    let gamma_p = s * gamma;
    let cos_theta = (2.0 * gamma_p * gamma_p - z.powf(2.0/3.0)) / (2.0 * gamma_p * gamma_p - 1.0);
    let cos_theta = cos_theta.max(-1.0);

    (gamma_p, cos_theta, phi, s, z)
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
                    prefactor * 0.5 * w * spectrum(s, *chi, 0.0)
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
                    w * spectrum(s, *chi, 1.0)
                })
                .sum();

            let t_perp: f64 = GAUSS_32_NODES.iter()
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(t, w)| {
                    let s = 0.5 * (1.0 + t);
                    w * spectrum(s, *chi, -1.0)
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
            let chi = (1_f64.ln() + (100_f64.ln() - 1_f64.ln()) * rng.gen::<f64>()).exp();
            let sv1 = -1.0 + 2.0 * rng.gen::<f64>();

            let target: f64 = (0..10_000)
                .map(|i| 0.5 * (i as f64) / 10000.0)
                .map(|s| spectrum(s, chi, sv1))
                .reduce(f64::max)
                .unwrap();

            let result = spectrum_ceiling(chi, sv1);

            let err = (target - result) / target;

            println!(
                "chi = {:>9.3e}, ξ_1 = {:>6.3} => max = {:>9.3e}, predicted = {:>9.3e}, err = {:.2}%",
                chi, sv1, target, result, 100.0 * err,
            );

            assert!(err < 0.0);
        }
    }

    #[test]
    fn pair_angular_spectrum_ceiling() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        for _i in 0..1000 {
            let chi = (0.1_f64.ln() + (100_f64.ln() - 1_f64.ln()) * rng.gen::<f64>()).exp();
            let sv1 = -1.0 + 2.0 * rng.gen::<f64>();
            let s = 0.5 * rng.gen::<f64>();

            let target: f64 = (0..100)
                .map(|i| 0.5 * (i as f64) / 100.0) // search in 0 < y < 0.5
                .map(|y| angular_spectrum(y, s, chi, sv1))
                .reduce(f64::max)
                .unwrap();

            let result = angular_spectrum_ceiling(s, chi, sv1);

            let err = (target - result) / target;

            println!(
                "chi = {:>9.3e}, ξ_1 = {:>6.3}, s = {:.3} => max = {:>9.3e}, predicted = {:>9.3e}, err = {:.2}%",
                chi, sv1, s, target, result, 100.0 * err,
            );

            assert!(err < 0.0 || target < 1.0e-200);
        }
    }

    #[test]
    #[ignore]
    fn pair_spectrum_sampling() {
        let s1 = 0.0;
        let chi = 2.0;
        let gamma = 1000.0;
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);
        let path = format!("output/lcfa_pair_spectrum_{}_{}.dat", chi, s1);
        let mut file = File::create(path).unwrap();
        for _i in 0..200000 {
            let (_, _, phi, s, z) = sample(
                [gamma, 0.0, 0.0, -gamma].into(),
                [1.0, s1, 0.0, 0.0].into(),
                chi,
                [1.0, 0.0, 0.0].into(),
                &mut rng
            );
            assert!(s > 0.0 && s < 1.0);
            assert!(z >= 1.0);
            writeln!(file, "{:.6e} {:.6e} {:.6e}", s, z, phi).unwrap();
        }
    }

    #[test]
    fn pure_states_remain_so() {
        let ell: FourVector = [100.0, 0.0, 0.0, -100.0].into();
        let a_perp: ThreeVector = [1.0, 0.0, 0.0].into();

        let svs: [StokesVector; 2] = [
            [1.0,  1.0,  0.0,  0.0].into(),
            [1.0, -1.0,  0.0,  0.0].into(),
        ];

        for sv in &svs {
            let (_, sv_new) = probability(ell, *sv, 1.0, a_perp, 1.0e-6 / SPEED_OF_LIGHT);
            assert!((sv[1] - sv_new[1]).abs() < 1.0e-12);
        }
    }

    fn positron_yield<R: Rng>(chi_max: f64, gamma: f64, sv1: f64, bias: f64, change_sv1: bool, rng: &mut R) -> f64 {
        let tau = 10.0;
        let dt = 1.0e-6 / (20.0 * SPEED_OF_LIGHT);
        let ell: FourVector = [gamma, 0.0, 0.0, -gamma].into();
        let a_perp: ThreeVector = [1.0, 0.0, 0.0].into();
        let mut sv: StokesVector = [1.0, sv1, 0.0, 0.0].into();
        let mut weight = 1.0;
        let mut count = 0.0;
        for i in -300..300 {
            let t = (i as f64) * 0.05;
            let chi = chi_max * (-(t/tau).powi(2)).exp();
            let (prob, sv_new) = probability(ell, sv, chi, a_perp, dt);
            let bias = if prob * bias > 0.1 { 0.1 / prob } else { bias };
            let prob = prob * bias;
            if rng.gen::<f64>() < prob {
                count += weight / bias;
                weight -= weight / bias;
                if weight <= 0.0 {
                    break;
                }
            }
            if change_sv1 {
                sv = sv_new;
            }
        }
        count
    }

    #[test]
    #[ignore]
    fn polarization_change() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let n = 50_000;
        let gamma = 1000.0;
        // let (chi_max, bias) = (1.0, 1.0);
        let (chi_max, bias) = (0.5, 100.0);

        // incoherent sum of sigma and pi
        let mut ps = vec![];
        for _bx in 0..10 {
            let mut count = 0.0;
            for _i in 0..n {
                count += positron_yield(chi_max, gamma, 1.0, bias, false, &mut rng);
                count += positron_yield(chi_max, gamma, -1.0, bias, false, &mut rng);
            }
            ps.push(count / (2.0 * n as f64));
        }

        let mean = ps.iter().sum::<f64>() / 10.0;
        let sdev = ps.iter().map(|p| (p - mean).powi(2)).sum::<f64>().sqrt() / 10.0;

        // unpolarized (correct)
        let mut ps = vec![];
        for _bx in 0..10 {
            let mut count = 0.0;
            for _i in 0..(2 * n) {
                count += positron_yield(chi_max, gamma, 0.0, bias, true, &mut rng);
            }
            ps.push(count / (2.0 * n as f64));
        }

        let mean2 = ps.iter().sum::<f64>() / 10.0;
        let sdev2 = ps.iter().map(|p| (p - mean2).powi(2)).sum::<f64>().sqrt() / 10.0;

        // unpolarized (incorrect)
        let mut ps = vec![];
        for _bx in 0..10 {
            let mut count = 0.0;
            for _i in 0..(2 * n) {
                count += positron_yield(chi_max, gamma, 0.0, bias, false, &mut rng);
            }
            ps.push(count / (2.0 * n as f64));
        }

        let mean3 = ps.iter().sum::<f64>() / 10.0;
        let sdev3 = ps.iter().map(|p| (p - mean3).powi(2)).sum::<f64>().sqrt() / 10.0;

        println!("50-50 pure = {:.6e} ± {:.6e}, 100 mixed = {:.6e} ± {:.6e}, 100 mixed != {:.6e} ± {:.6e}", mean, sdev, mean2, sdev2, mean3, sdev3);

        let diff = mean - mean2;
        let width = sdev.hypot(sdev2);
        let diff3 = mean - mean3;
        let width3 = sdev.hypot(sdev3);

        println!("diff = {:.6e} ± {:.6e} [{:.2} sigma] or {:.6e} ± {:.6e} [{:.2} sigma]", diff, width, diff.abs() / width, diff3, width3, diff3.abs() / width3);
        assert!(diff.abs() < width);
    }
}
