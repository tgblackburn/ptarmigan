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
    let x = 2.0 * (n as f64) * a * theta.cos() * (v * (1.0 - v) / (1.0 + 0.5 * a * a)).sqrt();
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
    use rayon::prelude::*;
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
            let n_max = crate::nonlinear_compton::lp::sum_limit(*a, *eta);
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

    #[test]
    fn create_rate_table() {
        use crate::pwmci;
        use super::super::{
            sum_limit,
        };

        const LOW_A_LIMIT: f64 = 0.02;
        const A_DENSITY: usize = 20; // points per order of magnitude
        const N_COLS: usize = 61; // pts in a0 direction
        let mut table = [0.0; N_COLS];

        let num: usize = std::env::var("RAYON_NUM_THREADS")
            .map(|s| s.parse().unwrap_or(1))
            .unwrap_or(1);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num)
            .build()
            .unwrap();

        println!("Running on {:?}", pool);

        let mut pts: Vec<(usize, f64, i32)> = Vec::new();
        for j in 0..N_COLS {
            let a = LOW_A_LIMIT * 10.0f64.powf((j as f64) / (A_DENSITY as f64));
            let n_max = sum_limit(a, 0.0);
            pts.push((j, a, n_max));
        }

        let pts: Vec<(usize, f64, [[f64; 2]; 16])> = pool.install(|| {
            pts.into_par_iter().rev()
            .map(|(j, a, n_max)| {
                let mut cumsum = 0.0;
                let rates: Vec<[f64; 2]> = (1..=n_max)
                    .map(|n| {
                        let (rate, _) = partial_rate(n, a);
                        let rate = (n as f64) * rate;
                        cumsum = cumsum + rate;
                        let interval = if a > 15.0 {100} else {20};
                        if a > 8.0 && n % (n_max / interval) == 0 {
                            println!("\t Progress report from [{:>3}], a = {:.3e}: done {} of {} ({:.0}%)...", rayon::current_thread_index().unwrap_or(1), a, n, n_max, 100.0 * (n as f64) / (n_max as f64));
                        }
                        [n as f64, cumsum]
                    })
                    .collect();

                // Total rate
                let rate: f64 = rates.last().unwrap()[1];

                let mut cdf: [[f64; 2]; 16] = [[0.0, 0.0]; 16];
                cdf[0] = [rates[0][0], rates[0][1] / rate];
                if n_max <= 16 {
                    // Write all the rates
                    for i in 1..=15 {
                        cdf[i] = rates.get(i)
                            .map(|r| [r[0], r[1] / rate])
                            .unwrap_or_else(|| [(i+1) as f64, 1.0]);
                    }
                } else if n_max < 100 {
                    // first 4 four harmonics
                    for i in 1..=3 {
                        cdf[i] = [rates[i][0], rates[i][1] / rate];
                    }
                    // log-spaced for n >= 5
                    let delta = ((n_max as f64).ln() - 5_f64.ln()) / 11.0;
                    for i in 4..=15 {
                        let n = (5_f64.ln() + ((i - 4) as f64) * delta).exp();
                        let limit = rates.last().unwrap()[0];
                        let n = n.min(limit);
                        cdf[i][0] = n;
                        cdf[i][1] = pwmci::evaluate(n, &rates[..]).unwrap() / rate;
                    }
                } else {
                    // Sample CDF at 16 log-spaced points
                    let delta = (n_max as f64).ln() / 15.0;
                    for i in 1..=15 {
                        let n = ((i as f64) * delta).exp();
                        let limit = rates.last().unwrap()[0];
                        let n = n.min(limit);
                        cdf[i][0] = n;
                        cdf[i][1] = pwmci::evaluate(n, &rates[..]).unwrap() / rate;
                    }
                }

                println!("LP classical NLC [{:>3}]: a = {:.3e}, ln(rate) = {:.6e}", rayon::current_thread_index().unwrap_or(1), a, rate.ln());
                (j, rate, cdf)
            })
            .collect()
        });

        for (j, rate, _) in &pts {
            table[*j] = *rate;
        }

        let mut file = File::create("output/classical_rate_table.rs").unwrap();
        //writeln!(file, "use std::f64::NEG_INFINITY;").unwrap();
        writeln!(file, "pub const N_COLS: usize = {};", N_COLS).unwrap();
        writeln!(file, "pub const MIN: f64 = {:.12e};", LOW_A_LIMIT.ln()).unwrap();
        writeln!(file, "pub const STEP: f64 = {:.12e};", consts::LN_10 / (A_DENSITY as f64)).unwrap();
        writeln!(file, "pub const TABLE: [f64; {}] = [", N_COLS).unwrap();
        for row in table.iter() {
            let val = row.ln();
            if val.is_finite() {
                write!(file, "\t{:>18.12e},", val).unwrap();
            } else {
                write!(file, "\t{:>18},", "NEG_INFINITY").unwrap();
            }
        }
        writeln!(file, "];").unwrap();

        let mut file = File::create("output/classical_cdf_table.rs").unwrap();
        writeln!(file, "pub const N_COLS: usize = {};", N_COLS).unwrap();
        writeln!(file, "pub const MIN: f64 = {:.12e};", LOW_A_LIMIT.ln()).unwrap();
        writeln!(file, "pub const STEP: f64 = {:.12e};", consts::LN_10 / (A_DENSITY as f64)).unwrap();
        writeln!(file, "pub const TABLE: [[[f64; 2]; 16]; {}] = [", N_COLS).unwrap();
        for (_, _, cdf) in &pts {
            write!(file, "\t[").unwrap();
            for entry in cdf.iter().take(15) {
                write!(file, "[{:>18.12e}, {:>18.12e}], ", entry[0], entry[1]).unwrap();
            }
            writeln!(file, "[{:>18.12e}, {:>18.12e}]],", cdf[15][0], cdf[15][1]).unwrap();
        }
        writeln!(file, "];").unwrap();
    }
}