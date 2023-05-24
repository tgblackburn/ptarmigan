//! Rates and spectra for linearly polarized backgrounds,
//! in the classical regime eta -> 0
use std::f64::consts;
use rand::prelude::*;
use crate::special_functions::*;
use crate::pwmci;
use crate::geometry::StokesVector;
use super::{
    ThetaBound,
    GAUSS_32_NODES, GAUSS_32_WEIGHTS,
};

mod rate_table;
mod cdf_table;

/// Rate, differential in v ( = s/s_max, where s is the fractional lightfront momentum
/// transfer) and theta (azimuthal angle).
/// Result valid only for 0 < v < 1 and 0 < theta < pi/2.
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

/// Returns the largest value of the double-differential rate, multiplied by a small safety factor.
fn ceiling_double_diff_partial_rate(a: f64, dj: &mut DoubleBessel) -> f64 {
    let n = dj.n();

    let max = if n == 1 {
        let v = 0.01;
        let theta = consts::FRAC_PI_2;
        double_diff_partial_rate(a, v, theta, dj)
    } else if a <= 1.5 {
        let theta = 0.0;

        let v_min = 0.1 + (9.0 / (n as f64).sqrt() - 1.0) / 10.0;
        let v_max = (v_min + 0.4).min(1.0);

        (1..=16)
            .map(|i| {
                let v = v_min + (v_max - v_min) * (i as f64) / 16.0;
                double_diff_partial_rate(a, v, theta, dj)
            })
            .reduce(f64::max)
            .unwrap()
    } else if a < 5.0 {
        let n_switch = (3.0 * (a / 2.0).powf(4.0 / 3.0)) as i32;

        let theta = if n > n_switch {
            0.0
        } else {
            consts::FRAC_PI_2 * (1.0 - (n as f64).ln() / (n_switch as f64).ln())
        };

        let v_min = 0.35 + (7.0 / (n as f64).sqrt() - 1.0) / 9.0;
        let v_min = v_min.min(0.55);
        let v_max = 0.55 + (7.0 / (n as f64).sqrt() - 1.0) / 9.0;

        let num = if a > 4.0 {32} else {16};
        (0..num)
            .map(|i| {
                let v = v_min + (v_max - v_min) * (i as f64) / (num as f64);
                double_diff_partial_rate(a, v, theta, dj)
            })
            .reduce(f64::max)
            .unwrap()
    } else {
        let n_switch = 10.0 * (a / 5.0).powi(2);

        let theta = if n > (n_switch as i32) {
            0.0
        } else {
            let pow = 0.3 + (0.7 - 0.3) * (a.ln() - 5_f64.ln()) / (20_f64.ln() - 5_f64.ln());
            consts::FRAC_PI_2 * ((n as f64).powf(-pow) - n_switch.powf(-pow)) / (1.0 - n_switch.powf(-pow))
        };

        // println!("n_switch = {}, theta = {:.3e}", n_switch as i32, theta);

        let v_min = 0.3 + 0.4 * (n as f64).ln() * (n as f64).powf(-0.7);
        let v_max = v_min + 0.2;

        let num = if a > 8.0 {64} else {32};
        (0..num)
            .map(|i| {
                let v = v_min + (v_max - v_min) * (i as f64) / (num as f64);
                double_diff_partial_rate(a, v, theta, dj)
            })
            .reduce(f64::max)
            .unwrap()
    };

    1.1 * max
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

/// Integrates `double_diff_partial_rate` over v and theta, returning
/// the value of the integral and the largest value of the integrand.
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

/// Returns the total rate of nonlinear Compton scattering.
/// Equivalent to calling
/// ```
/// let nmax = sum_limit(a, eta);
/// let rate = (1..=nmax).map(|n| 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a) * partial_rate(n, a)).sum::<f64>();
/// ```
/// but implemented as a table lookup.
/// Multiply by alpha / eta to get dP/dphase.
#[allow(unused_parens)]
pub fn rate(a: f64, eta: f64) -> Option<f64> {
    let x = a.ln();

    if x < rate_table::MIN {
        // linear Thomson scattering
        Some(a * a * eta / 3.0)
    } else {
        let ix = ((x - rate_table::MIN) / rate_table::STEP) as usize;

        if ix < rate_table::N_COLS - 1 {
            let dx = (x - rate_table::MIN) / rate_table::STEP - (ix as f64);
            let f = (
                (1.0 - dx) * rate_table::TABLE[ix]
                + dx * rate_table::TABLE[ix+1]
            );
            let prefactor = 2.0 * eta / (1.0 + 0.5 * a * a);
            Some(prefactor * f.exp())
        } else {
            eprintln!("NLC (classical LP) rate lookup out of bounds: a = {:.3e}", a);
            None
        }
    }
}

/// Obtain harmonic index by inverting frac = cdf(n), where 0 <= frac < 1 and
/// the cdf is tabulated.
fn get_harmonic_index(a: f64, frac: f64) -> i32 {
    if a.ln() <= cdf_table::MIN {
        // first harmonic only
       1
    } else {
        let ix = ((a.ln() - cdf_table::MIN) / cdf_table::STEP) as usize;
        let dx = (a.ln() - cdf_table::MIN) / cdf_table::STEP - (ix as f64);

        let index = [
            ix,
            ix + 1,
        ];

        let weight = [
            1.0 - dx,
            dx,
        ];

        let n_alt: f64 = index.iter()
            .zip(weight.iter())
            .map(|(i, w)| {
                let table = &cdf_table::TABLE[*i];
                let n = if frac <= table[0][1] {
                    0.9
                } else {
                    pwmci::Interpolant::new(table).invert(frac).unwrap()
                };
                n * w
            })
            .sum();

        n_alt.ceil() as i32
    }
}

// Returns a pseudorandomly sampled n (harmonic order), s (lightfront momentum
/// transfer) and theta (azimuthal angle in the ZMF) for a photon emission that
/// occurs at normalized amplitude a and energy parameter eta.
#[allow(non_snake_case, unused_parens)]
pub fn sample<R: Rng>(a: f64, eta: f64, rng: &mut R, fixed_n: Option<i32>) -> (i32, f64, f64, StokesVector) {
    let n = fixed_n.unwrap_or_else(|| {
        let frac = rng.gen::<f64>();
        get_harmonic_index(a, frac) // via lookup of cdf
    });

    let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);
    let theta_max = ThetaBound::for_harmonic_low_eta(n, a);

    let smax = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a); // classically s is unbounded

    let max = ceiling_double_diff_partial_rate(a, &mut dj);

    // Rejection sampling
    let (v, s, theta) = loop {
        let v = rng.gen::<f64>();
        let s = v * smax;
        let theta = consts::FRAC_PI_2 * rng.gen::<f64>();
        if theta > theta_max.at(v) {
            continue;
        }
        let z = max * rng.gen::<f64>();
        let f = double_diff_partial_rate(a, v, theta, &mut dj);
        if z < f {
            break (v, s, theta);
        }
    };

    // println!("\t... got v = {:.3e}, s = {:.3e}, theta = {:.3e}", v, s, theta);

    // Fix range of theta, which is [0, pi/2] at the moment
    let quadrant = rng.gen_range(0, 4);
    let theta = match quadrant {
        0 => theta,
        1 => consts::PI - theta,
        2 => consts::PI + theta,
        3 => 2.0 * consts::PI - theta,
        _ => unreachable!(),
    };

    // Generate Stokes parameters for emitted photon

    // opposite sign! cos theta > 0 in 0 < theta < pi/2
    let x = 2.0 * (n as f64) * a * theta.cos() * (v * (1.0 - v) / (1.0 + 0.5 * a * a)).sqrt();
    let y = 0.25 * (n as f64) * a * a * v / (1.0 + 0.5 * a * a);

    // need to correct for x being negative, using
    // J_n(-|x|, y) = (-1)^n J_n(|x|, y)
    let j = dj.evaluate(x, y);

    let A = if n % 2 == 0 {
        // j[1] and j[3] change sign
        [j[2], -0.5 * (j[1] + j[3]), 0.25 * (j[0] + 2.0 * j[2] + j[4])]
    } else {
        // j[0], j[2] and j[4] change sign
        [-j[2], 0.5 * (j[1] + j[3]), -0.25 * (j[0] + 2.0 * j[2] + j[4])]
    };

    let pol: StokesVector = {
        let u = ((1.0 + 0.5 * a * a) * (1.0 - v) / v).sqrt();

        let xi_0 = (
            2.0 * (A[1].powi(2) - A[0] * A[2])
            - 2.0 * (A[0] / a).powi(2)
        );

        let xi_1 = (
            2.0 * (A[1].powi(2) - A[0] * A[2])
            - 2.0 * (1.0 + 2.0 * u * u * theta.sin().powi(2)) * (A[0] / a).powi(2)
        );

        let xi_2 = (
            2.0 * (u * A[0] / a).powi(2) * (2.0 * theta).sin()
            + 4.0 * u * A[0] * A[1] * theta.sin() / a
        );

        [1.0, xi_1 / xi_0, xi_2 / xi_0, 0.0].into()
    };

    (n, s, theta, pol)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use rand::prelude::*;
    use rand_xoshiro::*;
    use rayon::prelude::*;
    use super::*;
    use super::super::sum_limit;

    #[test]
    #[ignore]
    fn integration() {
        let (n, a) = (1, 4.0);

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
    fn find_rate_max() {
        let num: usize = std::env::var("RAYON_NUM_THREADS")
            .map(|s| s.parse().unwrap_or(1))
            .unwrap_or(1);

        rayon::ThreadPoolBuilder::new()
            .num_threads(num)
            .build_global()
            .unwrap();

        let a = 8.0;
        let n_max = sum_limit(a, 0.0);
        let nodes: Vec<f64> = (1..100).map(|i| (i as f64) / 100.0).collect();

        let harmonics: Vec<_> = if a <= 3.0 {
            (1..n_max).collect()
        } else if a <= 6.0 {
            (1..10).chain((10..100).step_by(2)).chain((100..n_max).step_by(5)).collect()
        } else if a <= 10.0 {
            (1..10).chain((10..100).step_by(5)).chain((100..n_max).step_by(10)).collect()
        } else {
            (1..10).chain((10..100).step_by(10)).chain((100..n_max).step_by(50)).collect()
        };

        let pts: Vec<_> = harmonics.into_par_iter()
            .map(|n| {
                let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);
                let max_theta = ThetaBound::for_harmonic_low_eta(n, a);

                let mut max = 0.0;
                let mut v0 = 0.0;
                let mut theta0 = 0.0;

                for v in nodes.iter() {
                    for theta in nodes.iter().map(|x| x * max_theta.at(*v)) {
                        let val = double_diff_partial_rate(a, *v, theta, &mut dj);
                        if val > max {
                            max = val;
                            v0 = *v;
                            theta0 = theta;
                        }
                    }
                }

                (n, v0, theta0, max)
            })
            .collect();

        let filename = format!("output/nlc_cl_lp_max_{}.dat", a);
        let mut file = File::create(&filename).unwrap();
        for (n, v, theta, max) in &pts {
            writeln!(file, "{} {:.6e} {:.6e} {:.6e}", n, v, theta, max).unwrap();
        }
    }

    #[test]
    fn rate_ceiling() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        for _i in 0..20 {
            let a = (0.2_f64.ln() + (20_f64.ln() - 0.2_f64.ln()) * rng.gen::<f64>()).exp();
            let n_max = sum_limit(a, 0.0);
            let harmonics: Vec<_> = if n_max > 200 {
                (0..=10).map(|i| (2_f64.ln() + 0.1 * (i as f64) * ((n_max as f64).ln() - 2_f64.ln())).exp() as i32).collect()
            } else if n_max > 10 {
                let mut low = vec![1, 2, 3];
                let mut high: Vec<_> = (0..=4).map(|i| (5_f64.ln() + 0.25 * (i as f64) * ((n_max as f64).ln() - 5_f64.ln())).exp() as i32).collect();
                low.append(&mut high);
                low
            } else {
                (1..n_max).collect()
            };

            for n in &harmonics {
                let (_, true_max) = partial_rate(*n, a);
                let mut dj = DoubleBessel::at_index(*n, (*n as f64) * consts::SQRT_2, (*n as f64) * 0.5);
                let max = ceiling_double_diff_partial_rate(a, &mut dj);
                let err = (true_max - max) / true_max;
                println!(
                    "a = {:>9.3e}, n = {:>4} => max = {:>9.3e}, predicted = {:>9.3e}, err = {:.2}%",
                    a, n, true_max, max, 100.0 * err,
                );
                assert!(err < 0.0);
            }
        }
    }

    #[test]
    #[ignore]
    fn partial_rate_accuracy() {
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
    fn total_rate_accuracy() {
        let pts = [
            (0.1,  1.0e-3, 3.328348674e-6),
            (0.3,  1.0e-3, 2.960584967e-5),
            (1.0,  1.0e-3, 2.943962548e-4),
            (3.0,  1.0e-3, 1.711123935e-3),
            (5.0,  1.0e-3, 3.382663412e-3),
            (8.0,  1.0e-3, 6.000395217e-3),
            (12.0, 1.0e-3, 9.562344320e-3),
            (15.0, 1.0e-3, 1.225662395e-2),
        ];

        // let num: usize = std::env::var("RAYON_NUM_THREADS")
        //     .map(|s| s.parse().unwrap_or(1))
        //     .unwrap_or(1);

        // let pool = rayon::ThreadPoolBuilder::new()
        //     .num_threads(num)
        //     .build()
        //     .unwrap();

        for (a, eta, total_rate) in &pts {
            // let target = {
            //     let n_max = crate::nonlinear_compton::lp::sum_limit(*a, 0.0);
            //     let prefactor = 2.0 * eta / (1.0 + 0.5 * a * a);
            //     pool.install(||
            //         prefactor * (1..=n_max).into_par_iter().map(|n| (n as f64) * partial_rate(n, *a).0).sum::<f64>()
            //     )
            // };
            let target = *total_rate;
            let classical = rate(*a, *eta).unwrap();
            let classical_error = (classical - target).abs() / target;
            let qed = crate::nonlinear_compton::lp::rate(*a, *eta).unwrap();
            let qed_error = (qed - target).abs() / target;
            println!(
                "a = {:>4}, eta = {:>6.2e}: target = {:>15.9e}, predicted = {:>9.3e} [diff = {:.2}%], qed = {:>9.3e} [diff = {:.2}%]",
                a, eta, target,
                classical, 100.0 * classical_error,
                qed, 100.0 * qed_error,
            );
            assert!(classical_error < 1.0e-3);
        }
    }

    #[test]
    fn harmonic_index_sampling() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let pts = [
            0.474275,
            0.946303,
            2.37700,
            4.74275,
        ];

        for a in &pts {
            println!("At a = {}:", a);
            let nmax = sum_limit(*a, 0.0);
            let nmax = nmax.max(19);
            let rates: Vec<f64> = (1..=nmax).map(|n| (n as f64) * partial_rate(n, *a).0).collect();
            let total: f64 = rates.iter().sum();
            println!("\t ... rates computed");

            let bins = [
                1..=1,
                2..=2,
                3..=4,
                5..=9,
                10..=19,
            ];

            let mut counts = [0.0; 5];

            let expected = [
                rates[0] / total,
                rates[1] / total,
                rates[2..=3].iter().sum::<f64>() / total,
                rates[4..=8].iter().sum::<f64>() / total,
                rates[9..=18].iter().sum::<f64>() / total,
            ];

            for _i in 0..1_000_000 {
                //let (n, _, _) = sample(*a, *eta, &mut rng, None);
                let frac = rng.gen::<f64>();
                let n = get_harmonic_index(*a, frac);
                for (j, bin) in bins.iter().enumerate() {
                    if bin.contains(&n) {
                        counts[j] += 1.0e-6;
                        break;
                    }
                }
            }

            for (b, (c, e)) in bins.iter()
                .zip(counts.iter()
                .zip(expected.iter()))
            {
                if *e < 1.0e-3 {
                    continue;
                }
                let error = (c - e).abs() / e;
                println!("\tExpected = {:.3e}, got {:.3e} [{:.1}%] for n in {:?}", e, c, 100.0 * error, b);
                assert!(error < 5.0e-2);
            }
        }
    }

    #[test]
    #[ignore]
    fn create_rate_table() {
        use indicatif::{ProgressStyle, ParallelProgressIterator};

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

        let style = ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})").unwrap();

        let pts: Vec<(usize, f64, [[f64; 2]; 16])> = pool.install(|| {
            pts.into_iter()
            .map(|(j, a, n_max)| {
                let rates: Vec<[f64; 2]> = (1..(n_max+1))
                    .into_par_iter()
                    .progress_with_style(style.clone())
                    .map(|n| {
                        let (rate, _) = partial_rate(n, a);
                        let rate = (n as f64) * rate;
                        [n as f64, rate]
                    })
                    .collect();

                // Cumulative sum
                let rates: Vec<[f64; 2]> = rates
                    .into_iter()
                    .scan(0.0, |cs, [n, r]| {
                        *cs = *cs + r;
                        Some([n, *cs])
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
                        cdf[i][1] = pwmci::Interpolant::new(&rates[..]).evaluate(n).unwrap() / rate;
                    }
                } else {
                    // Sample CDF at 16 log-spaced points
                    let delta = (n_max as f64).ln() / 15.0;
                    for i in 1..=15 {
                        let n = ((i as f64) * delta).exp();
                        let limit = rates.last().unwrap()[0];
                        let n = n.min(limit);
                        cdf[i][0] = n;
                        cdf[i][1] = pwmci::Interpolant::new(&rates[..]).evaluate(n).unwrap() / rate;
                    }
                }

                println!("LP classical NLC: a = {:.3e}, ln(rate) = {:.6e}", a, rate.ln());
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
                writeln!(file, "\t{:>18.12e},", val).unwrap();
            } else {
                writeln!(file, "\t{:>18},", "NEG_INFINITY").unwrap();
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