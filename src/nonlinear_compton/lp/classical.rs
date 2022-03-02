//! Rates and spectra for linearly polarized backgrounds,
//! in the classical regime eta -> 0
use std::f64::consts;
use crate::special_functions::*;
use super::{
    ThetaBound,
    GAUSS_32_NODES, GAUSS_32_WEIGHTS,
};

/// Rate, differential in v ( = s/s_max, where s is the fractional lightfront momentum
/// transfer) and theta (azimuthal angle).
/// Result valid only for 0 < v < 1 and 0 < theta < pi/2.
/// Multiply by alpha / eta to get dP/(ds dtheta dphase)
#[allow(unused)]
fn double_diff_partial_rate(a: f64, v: f64, theta: f64, dj: &mut DoubleBessel) -> f64 {
    let n = dj.n();

    // opposite sign! cos theta > 0 in 0 < theta < pi/2
    let x =2.0 * (n as f64) * a * theta.cos() * (v * (1.0 - v) / (1.0 + 0.5 * a * a)).sqrt();
    let y = 0.25 * (n as f64) * a * a * v / (1.0 + 0.5 * a * a);

    // need to correct for x being negative, using
    // J_n(-|x|, y) = (-1)^n J_n(|x|, y)
    let j = dj.evaluate(x, y);

    let gamma = if n % 2 == 0 {
        // j[1] and j[3] change sign
        [j[2], -0.5 * (j[1] + j[3]), 0.25 * (j[0] + 2.0 * j[2] + j[4])]
    } else {
        // j[0], j[2] and j[4] change sign
        [-j[2], 0.5 * (j[1] + j[3]), -0.25 * (j[0] + 2.0 * j[2] + j[4])]
    };

    (-gamma[0] * gamma[0] - a * a * (gamma[0] * gamma[2] - gamma[1] * gamma[1])) / (2.0 * consts::PI)
}

/// Integrates `double_diff_partial_rate` over 0 < theta < 2 pi, returning
/// the value of the integral and the largest value of the integrand.
#[allow(unused)]
fn single_diff_partial_rate(a: f64, v: f64, theta_max: f64, dj: &mut DoubleBessel) -> (f64, f64) {
    GAUSS_32_NODES.iter()
        // integrate over 0 to pi/2, then multiply by 4
        .map(|x| 0.5 * (x + 1.0) * theta_max)
        .zip(GAUSS_32_WEIGHTS.iter())
        .map(|(theta, w)| {
            let rate = double_diff_partial_rate(a, v, theta, dj);
            (4.0 * (0.5 * theta_max) * w * rate, rate)
        })
        .fold(
            (0.0f64, 0.0f64),
            |a, b| (a.0 + b.0, a.1.max(b.1))
        )
}

/// Integrates `double_diff_partial_rate` over s and theta, returning
/// the value of the integral and the largest value of the integrand.
/// Multiply by alpha / eta to get dP/(ds dtheta dphase)
#[allow(unused)]
fn partial_rate(n: i32, a: f64) -> (f64, f64) {
    // allocate once and reuse
    let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);
    let theta_max = ThetaBound::for_harmonic_low_eta(n, a);

    // from v = 0 to 0.5
    let lower = GAUSS_32_NODES.iter()
        .map(|x| 0.5 * (x + 1.0) * 0.5)
        .zip(GAUSS_32_WEIGHTS.iter())
        .map(|(v, w)| {
            let (rate, max) = single_diff_partial_rate(a, v, theta_max.at(v), &mut dj);
            (w * (0.5 * 0.5) * rate, max)
        })
        .fold(
            (0.0f64, 0.0f64),
            |a, b| (a.0 + b.0, a.1.max(b.1))
        );

    // from v = 0.5 to 0.75
    let middle = GAUSS_32_NODES.iter()
        .map(|x| 0.5 + 0.5 * (x + 1.0) * 0.25)
        .zip(GAUSS_32_WEIGHTS.iter())
        .map(|(v, w)| {
            let (rate, max) = single_diff_partial_rate(a, v, theta_max.at(v), &mut dj);
            (w * (0.5 * 0.25) * rate, max)
        })
        .fold(
            (0.0f64, 0.0f64),
            |a, b| (a.0 + b.0, a.1.max(b.1))
        );

    // from v = 0.75 to 1
    let upper = GAUSS_32_NODES.iter()
        .map(|x| 0.75 + 0.5 * (x + 1.0) * 0.25)
        .zip(GAUSS_32_WEIGHTS.iter())
        .map(|(v, w)| {
            let (rate, max) = single_diff_partial_rate(a, v, theta_max.at(v), &mut dj);
            (w * (0.5 * 0.25) * rate, max)
        })
        .fold(
            (0.0f64, 0.0f64),
            |a, b| (a.0 + b.0, a.1.max(b.1))
        );

    (lower.0 + middle.0 + upper.0, upper.1.max(middle.1).max(lower.1))
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use rand::prelude::*;
    use rand_xoshiro::*;
    use super::*;

    #[test]
    #[ignore]
    fn integration() {
        let (n, a) = (100, 10.0);

        let nodes: Vec<f64> = (1..300).map(|i| (i as f64) / 300.0).collect();
        let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);
        let max_theta = ThetaBound::for_harmonic_low_eta(n, a);

        let filename = format!("output/nlc_lp_cl_dd_rate_{}_{}.dat", n, a);
        let mut file = File::create(&filename).unwrap();
        let mut max = 0.0;

        for v in nodes.iter() {
            for theta in nodes.iter().map(|x| x * consts::FRAC_PI_2) {//.filter(|&theta| theta < 0.3) {
                let rate = if theta > max_theta.at(*v) {
                    0.0
                } else {
                    double_diff_partial_rate(a, *v, theta, &mut dj)
                };
                if rate > max {
                    max = rate;
                }
                writeln!(file, "{:.6e} {:.6e} {:.6e}", v, theta, rate).unwrap();
            }
        }

        let (integral, predicted_max) = partial_rate(n, a);
        println!("integral = {:.6e}, max = {:.6e} [{:.6e} with finer resolution]", integral, predicted_max, max);

        let filename = format!("output/nlc_lp_cl_sd_rate_{}_{}.dat", n, a);
        let mut file = File::create(&filename).unwrap();

        let vs = GAUSS_32_NODES.iter().map(|x| 0.25 * (x + 1.0))
            .chain(GAUSS_32_NODES.iter().map(|x| 0.5 + 0.125 * (x + 1.0)))
            .chain(GAUSS_32_NODES.iter().map(|x| 0.75 + 0.125 * (x + 1.0)));
        for v in vs {
            let theta = max_theta.at(v);
            let (rate, _) = single_diff_partial_rate(a, v, theta, &mut dj);
            writeln!(file, "{:.6e} {:.6e} {:.6e}", v, rate, theta).unwrap();
        }
    }

    #[test]
    #[ignore]
    fn total_rate_accuracy() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let pts = [
            (1,    0.1),
            (3,    0.1),
            (10,   0.1),
            (1,    1.0),
            (5,    1.0),
            (20,   1.0),
            (1,    3.0),
            (10,   3.0),
            (100,  3.0),
            (1,    10.0),
            (300,  10.0),
            (1000, 10.0),
        ];

        for (n, a) in &pts {
            let mut dj = DoubleBessel::at_index(*n, (*n as f64) * consts::SQRT_2, (*n as f64) * 0.5);
            let max_theta = ThetaBound::for_harmonic_low_eta(*n, *a);
            let (integral, max) = partial_rate(*n, *a);
            let mut detected_max = 0.0;
            let mut count = 0_i32;
            let mut sub_count = 0_i32;
            let total = 500_000;

            for i in 0..total {
                let v = rng.gen::<f64>();
                let theta = consts::FRAC_PI_2 * rng.gen::<f64>();
                let z = 1.5 * max * rng.gen::<f64>();
                if theta > max_theta.at(v) {
                    continue;
                }
                let rate = double_diff_partial_rate(*a, v, theta, &mut dj);
                if rate > detected_max {
                    detected_max = rate;
                }
                if z < rate {
                    count +=1;
                    if i < 200_000 {
                        sub_count +=1;
                    }
                }
            }

            let volume = 1.5 * max * 1.0 * consts::FRAC_PI_2;
            let frac = (count as f64) / (total as f64);
            let mc_integral = 4.0 * volume * frac;
            let mc_integral_est = 4.0 * volume * (sub_count as f64) / 200_000.0;
            let mc_error = (mc_integral_est - mc_integral).abs() / mc_integral;
            let error = (integral - mc_integral).abs() / integral;
            println!(
                "n = {:>4}, a = {:>4}: integral = {:>9.3e}, mc = {:>9.3e} [diff = {:.2}%, estd conv = {:.2}%, success = {:.2}%]",
                n, a, integral, mc_integral, 100.0 * error, 100.0 * mc_error, 100.0 * frac,
            );
            assert!(detected_max < 1.5 * max);
            assert!(error < 0.05);
        }
    }

    #[test]
    #[ignore]
    fn qed_vs_classical() {
        let pts = [
            (0.1, 1.0e-3),
            (0.1, 3.0e-4),
            (0.1, 1.0e-4),
            (1.0, 1.0e-3),
            (1.0, 3.0e-4),
            (1.0, 1.0e-4),
            (3.0, 1.0e-3),
            (3.0, 3.0e-4),
            (3.0, 1.0e-4),
            (10.0, 1.0e-4),
        ];

        for (a, eta) in &pts {
            let n_max = (5_f64 * (1.0 + 2.0 * a * a)).ceil() as i32;
            let target = (1..=n_max).map(|n| crate::nonlinear_compton::lp::partial_rate(n, *a, *eta).0).sum::<f64>();
            let qed = crate::nonlinear_compton::lp::rate(*a, *eta).unwrap();
            let prefactor = 2.0 * eta / (1.0 + 0.5 * a * a);
            let classical = prefactor * (1..=n_max).map(|n| (n as f64) * partial_rate(n, *a).0).sum::<f64>();
            let qed_error = (qed - target).abs() / target;
            let classical_error = (classical - target).abs() / target;
            println!(
                "a = {:>4}, eta = {:>6.2e}: target = {:>9.3e}, qed = {:>9.3e} [diff = {:.2}%], classical = {:>9.3e} [diff = {:.2}%]",
                a, eta, target, qed, 100.0 * qed_error, classical, 100.0 * classical_error,
            );
        }
    }
}