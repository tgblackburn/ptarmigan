//! Rates and spectra for circularly polarized backgrounds
use std::f64::consts;
use num_complex::Complex;
use rand::prelude::*;
use crate::special_functions::*;
use super::{GAUSS_16_NODES, GAUSS_16_WEIGHTS, GAUSS_32_NODES, GAUSS_32_WEIGHTS};

mod rate_table;

/// Rate, differential in s (fractional lightfront momentum transfer)
/// and theta (polar angle).
/// Result valid only for 0 < s < s_max and 0 < theta < pi/2.
/// Multiply by alpha / eta to get dP/(ds dtheta dphase)
fn double_diff_partial_rate(a: f64, eta: f64, s: f64, theta: f64, dj: &mut DoubleBessel) -> f64 {
    let n = dj.n();

    // opposite sign! cos theta > 0 in 0 < theta < pi/2
    let alpha = {
        let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
        let wn = s / (sn * (1.0 - s));
        let wn = wn.min(1.0);
        2.0 * (n as f64) * a * theta.cos() * (wn * (1.0 - wn)).sqrt() / (1.0 + 0.5 * a * a).sqrt()
    };

    let beta = a * a * s / (8.0 * eta * (1.0 - s));

    // assert!(alpha < (n as f64) * consts::SQRT_2);
    // assert!(beta < (n as f64) * 0.5);

    // need to correct for alpha being negative, using
    // J_n(-|alpha|, beta) = (-1)^n J_n(|alpha|, beta)
    let j = dj.evaluate(alpha, beta); // n-2, n-1, n, n+1, n+2

    let gamma = if n % 2 == 0 {
        // j[1] and j[3] change sign
        [j[2], -0.5 * (j[1] + j[3]), 0.25 * (j[0] + 2.0 * j[2] + j[4])]
    } else {
        // j[0], j[2] and j[4] change sign
        [-j[2], 0.5 * (j[1] + j[3]), -0.25 * (j[0] + 2.0 * j[2] + j[4])]
    };

    (-gamma[0] * gamma[0] - a * a * (1.0 + 0.5 * s * s / (1.0 - s)) * (gamma[0] * gamma[2] - gamma[1] * gamma[1])) / (2.0 * consts::PI)
}

#[derive(Debug)]
struct ThetaBound {
    s: [f64; 16],
    f: [f64; 16],
}

impl ThetaBound {
    fn for_harmonic(n: i32, a: f64, eta: f64) -> Self {
        let mut s = [0.0; 16];
        let mut f = [0.0; 16];

        let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);

        for i in 0..16 {
            let z = (consts::PI * (i as f64) / 15.0).cos();
            s[i] = 0.5 * (z + 1.0) * sn / (1.0 + sn);

            // Coordinates in (x,y) space where integration over theta begins
            let x = {
                let wn = s[i] / (sn * (1.0 - s[i]));
                let wn = wn.min(1.0);
                2.0 * (n as f64) * a * (wn * (1.0 - wn)).sqrt() / (1.0 + 0.5 * a * a).sqrt()
            };

            let y = a * a * s[i] / (8.0 * eta * (1.0 - s[i]));

            // At fixed y, J(n,x,y) is maximised at
            let x_crit = if y > (n as f64) / 6.0 {
                4.0 * y.sqrt() * ((n as f64) - 2.0 * y).sqrt()
            } else {
                (n as f64) + 2.0 * y
            };

            // with value approx J_n(n) ~ 0.443/n^(1/3), here log-scaled
            let ln_j_crit = 0.443f64.ln() - (n as f64).ln() / 3.0;

            // The value of J(n,0,y) is approximately
            let ln_j_bdy = if n % 2 == 0 {
                Self::ln_double_bessel_x_zero(n, y)
            } else {
                Self::ln_double_bessel_x_zero(n + 1, y)
            };
            let ln_j_bdy = ln_j_bdy.min(ln_j_crit);

            // Exponential fit between x = 0 and x = x_crit, looking for the
            // x which is 100x smaller than at the starting point
            let cos_theta = 1.0 - x_crit * 0.01f64.ln() / (x * (ln_j_bdy - ln_j_crit));
            let cos_theta = cos_theta.max(-1.0);

            f[i] = cos_theta.acos();
            if f[i].is_nan() {
                f[i] = consts::PI;
            }

            //println!("(x, y) = ({:.3e}, {:.3e}), (x_crit, j_crit) = ({:.3e}, {:.3e}), j_bdy = {:.3e}, cos_theta = {:.3e}, theta = {:.3e}", x, y, x_crit, ln_j_crit.exp(), ln_j_bdy.exp(), cos_theta, f[i]);
        }

        let mut f_min = f[0];
        for i in 1..16 {
            if f[i] < f_min {
                f_min = f[i];
            } else {
                f[i] = f_min;
            }
        }

        Self {s, f}
    }

    /// Approximate value for the double Bessel function J(n,0,y)
    /// along x = 0, using a saddle point approximation for small y
    /// and a Taylor series for y near n/2.
    fn ln_double_bessel_x_zero(n: i32, y: f64) -> f64 {
        let n = n as f64;
        let z = (n / (4.0 * y) - 0.5).sqrt();
        let theta = Complex::new(
            consts::FRAC_PI_2,
            ((1.0 + z * z).sqrt() - z).ln()
        );
        let f = Complex::<f64>::i() * (-n * theta - y * (2.0 * theta).sin());
        let f2 = Complex::<f64>::i() * 4.0 * y * (2.0 * theta).sin();
        let e = 0.5 * (consts::PI - f2.arg());
        let phase = f + Complex::i() * e;
        (2.0 / (consts::PI * f2.norm())).sqrt().ln() + phase.re + Complex::new(phase.im, 0.0).cos().ln().re
    }

    fn at(&self, s: f64) -> f64 {
        let mut val = self.f[15];

        for i in 1..16 {
            // s[i] is stored backwards, decreasing from s_max
            if s > self.s[i] {
                let weight = (s - self.s[i-1]) / (self.s[i] - self.s[i-1]);
                val = weight * self.f[i] + (1.0 - weight) * self.f[i-1];
                break;
            }
        }

        val.min(consts::FRAC_PI_2)
    }
}

/// Integrates `double_diff_partial_rate` over 0 < theta < 2 pi, returning
/// the value of the integral and the largest value of the integrand.
fn single_diff_partial_rate(a: f64, eta: f64, s: f64, theta_max: f64, dj: &mut DoubleBessel) -> (f64, f64) {
    GAUSS_32_NODES.iter()
        // integrate over 0 to pi/2, then multiply by 4
        .map(|x| 0.5 * (x + 1.0) * theta_max)
        .zip(GAUSS_32_WEIGHTS.iter())
        .map(|(theta, w)| {
            let rate = double_diff_partial_rate(a, eta, s, theta, dj);
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
fn partial_rate(n: i32, a: f64, eta: f64) -> (f64, f64) {
    let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
    let smax = sn / (1.0 + sn);
    // approximate s where rate is maximised
    let s_peak = sn / (2.0 + sn);

    // allocate once and reuse
    let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);
    let theta_max = ThetaBound::for_harmonic(n, a, eta);

    let (integral, max): (f64, f64) = if sn < 1.0 {
        // if s_peak < 2/3 * smax
        // split integral in two: 0 to s_peak
        let lower = GAUSS_32_NODES.iter()
            .map(|x| 0.5 * (x + 1.0) * s_peak)
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(s, w)| {
                let (rate, max) = single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj);
                (w * (0.5 * s_peak) * rate, max)
            })
            .fold(
                (0.0f64, 0.0f64),
                |a, b| (a.0 + b.0, a.1.max(b.1))
            );

        // and then s_peak to s_max:
        let upper = GAUSS_32_NODES.iter()
            .map(|x| s_peak + 0.5 * (smax - s_peak) * (x + 1.0))
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(s, w)| {
                let (rate, max) = single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj);
                (w * 0.5 * (smax - s_peak) * rate, max)
            })
            .fold(
                (0.0f64, 0.0f64),
                |a, b| (a.0 + b.0, a.1.max(b.1))
            );

        (upper.0 + lower.0, upper.1.max(lower.1))
    } else {
        // split domain into three: 0 to sm-2d, sm-2d to sp, sp to sm
        // where d = sm - sp
        let (s0, s1) = (0.0, smax - 2.0 * (smax - s_peak));
        let lower = GAUSS_16_NODES.iter()
            .map(|x| s0 + 0.5 * (s1 - s0) * (x + 1.0))
            .zip(GAUSS_16_WEIGHTS.iter())
            .map(|(s, w)| {
                let (rate, max) = single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj);
                (w * 0.5 * (s1 - s0) * rate, max)
            })
            .fold(
                (0.0f64, 0.0f64),
                |a, b| (a.0 + b.0, a.1.max(b.1))
            );

        let (s0, s1) = (smax - 2.0 * (smax - s_peak), s_peak);
        let mid = GAUSS_32_NODES.iter()
            .map(|x| s0 + 0.5 * (s1 - s0) * (x + 1.0))
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(s, w)| {
                let (rate, max) = single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj);
                (w * 0.5 * (s1 - s0) * rate, max)
            })
            .fold(
                (0.0f64, 0.0f64),
                |a, b| (a.0 + b.0, a.1.max(b.1))
            );

        let (s0, s1) = (s_peak, smax);
        let upper = GAUSS_32_NODES.iter()
            .map(|x| s0 + 0.5 * (s1 - s0) * (x + 1.0))
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(s, w)| {
                let (rate, max) = single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj);
                (w * 0.5 * (s1 - s0) * rate, max)
            })
            .fold(
                (0.0f64, 0.0f64),
                |a, b| (a.0 + b.0, a.1.max(b.1))
            );

        (lower.0 + mid.0 + upper.0, lower.1.max(mid.1).max(upper.1))
    };

    (integral, max)
}

/// Returns the total rate of nonlinear Compton scattering.
/// Equivalent to calling
/// ```
/// let nmax = (5.0 * (1.0 + 2.0 * a * a)) as i32;
/// let rate = (1..=nmax).map(|n| partial_rate(n, a, eta)).sum::<f64>();
/// ```
/// but implemented as a table lookup.
/// Multiply by alpha / eta to get dP/(ds dtheta dphase).
#[allow(unused_parens)]
fn rate(a: f64, eta: f64) -> Option<f64> {
    let (x, y) = (a.ln(), eta.ln());

    if x < rate_table::MIN[0] {
        todo!()
    } else if y < rate_table::MIN[1] {
        todo!()
    } else {
        let ix = ((x - rate_table::MIN[0]) / rate_table::STEP[0]) as usize;
        let iy = ((y - rate_table::MIN[1]) / rate_table::STEP[1]) as usize;
        if ix < rate_table::N_COLS - 1 && iy < rate_table::N_ROWS - 1 {
            // linear interpolation of: log y against log x, best for power law
            let dx = (x - rate_table::MIN[0]) / rate_table::STEP[0] - (ix as f64);
            let dy = (y - rate_table::MIN[1]) / rate_table::STEP[1] - (iy as f64);
            let f = (
                (1.0 - dx) * (1.0 - dy) * rate_table::TABLE[iy][ix]
                + dx * (1.0 - dy) * rate_table::TABLE[iy][ix+1]
                + (1.0 - dx) * dy * rate_table::TABLE[iy+1][ix]
                + dx * dy * rate_table::TABLE[iy+1][ix+1]
            );
            Some(f.exp())
        } else {
            eprintln!("NLC (LP) rate lookup out of bounds: a = {:.3e}, eta = {:.3e}", a, eta);
            None
        }
    }
}

/// Returns a pseudorandomly sampled n (harmonic order), s (lightfront momentum
/// transfer) and theta (azimuthal angle in the ZMF) for a photon emission that
/// occurs at normalized amplitude a and energy parameter eta.
fn sample<R: Rng>(a: f64, eta: f64, rng: &mut R) -> (i32, f64, f64) {
    let nmax = (5.0 * (1.0 + 2.0 * a * a)) as i32;
    let frac = rng.gen::<f64>();
    let target = frac * rate(a, eta).unwrap();
    let mut cumsum: f64 = 0.0;
    let mut n: Option<i32> = None;
    let mut max = 0.0;
    for k in 1..=nmax {
        let tmp = partial_rate(k, a, eta);
        cumsum += tmp.0;
        if cumsum > target {
            n = Some(k);
            max = tmp.1;
            break;
        }
    }

    // interpolation errors mean that even after the sum, cumsum could be < target
    let n = n.unwrap_or_else(|| {
        eprintln!("lp::sample failed to obtain a harmonic order: target = {:.3e}% of rate at a = {:.3e}, eta = {:.3e} (n < {}), falling back to {}.", frac, a, eta, nmax, nmax - 1);
        nmax - 1
    });

    let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);
    let theta_max = ThetaBound::for_harmonic(n, a, eta);

    let sn = 2.0 * (n as f64) * eta / (1.0 + a * a);
    let smax = sn / (1.0 + sn);
    let max = 1.1 * max;

    // Rejection sampling
    let (s, theta) = loop {
        let s = smax * rng.gen::<f64>();
        let theta = consts::FRAC_PI_2 * rng.gen::<f64>();
        if theta > theta_max.at(s) {
            continue;
        }
        let z = max * rng.gen::<f64>();
        let f = double_diff_partial_rate(a, eta, s, theta, &mut dj);
        if z < f {
            break (s, theta);
        }
    };

    // Fix range of theta, which is [0, pi/2] at the moment
    let quadrant = rng.gen_range(0, 4);
    let theta = match quadrant {
        0 => theta,
        1 => consts::PI - theta,
        2 => consts::PI + theta,
        3 => 2.0 * consts::PI - theta,
        _ => unreachable!(),
    };

    (n, s, theta)
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
    fn integration_domain() {
        let (n, a, eta) = (100, 10.0, 0.1);
        // bounds on s
        let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
        let smax = sn / (1.0 + sn);

        let nodes: Vec<f64> = (0..=1000).map(|i| (i as f64) / 1000.0).collect();
        let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);

        let max_theta = ThetaBound::for_harmonic(n, a, eta);

        let filename = format!("output/nlc_lp_dd_rate_{}_{}_{}.dat", n, a, eta);
        let mut file = File::create(&filename).unwrap();
        let mut max = 0.0;
        for s in nodes.iter().map(|x| x * smax) {//.filter(|&s| s > 0.5) {
            for theta in nodes.iter().map(|x| x * consts::FRAC_PI_2) {//.filter(|&theta| theta < 0.3) {
                if theta > max_theta.at(s) {
                    continue;
                }
                let rate = double_diff_partial_rate(a, eta, s, theta, &mut dj);
                if rate > max {
                    max = rate;
                }
                writeln!(file, "{:.6e} {:.6e} {:.6e}", s, theta, rate).unwrap();
            }
        }

        let (integral, predicted_max) = partial_rate(n, a, eta);
        println!("integral = {:.6e}, max = {:.6e} [{:.6e} with finer resolution]", integral, predicted_max, max);
    }

    #[test]
    fn partial_rate_accuracy() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let harmonics = [
            (1,    0.1,   0.01),
            (10,   0.1,   0.01),
            (1,    1.0,   0.01),
            (20,   1.0,   0.01),
            (1,    10.0,  0.01),
            (300,  10.0,  0.01),
            (1000, 10.0,  0.01),
            (1,    0.1,   0.1),
            (10,   0.1,   0.1),
            (1,    1.0,   0.1),
            (20,   1.0,   0.1),
            (1,    10.0,  0.1),
            (300,  10.0,  0.1),
            (1000, 10.0,  0.1),
            (1,    0.1,   1.0),
            (10,   0.1,   1.0),
            (1,    1.0,   1.0),
            (20,   1.0,   1.0),
            (1,    10.0,  1.0),
            (300,  10.0,  1.0),
            (1000, 10.0,  1.0),
        ];

        for (n, a, eta) in &harmonics {
            let mut dj = DoubleBessel::at_index(*n, (*n as f64) * consts::SQRT_2, (*n as f64) * 0.5);
            let max_theta = ThetaBound::for_harmonic(*n, *a, *eta);
            let sn = 2.0 * (*n as f64) * eta / (1.0 + 0.5 * a * a);
            let smax = sn / (1.0 + sn);
            let (integral, max) = partial_rate(*n, *a, *eta);
            let mut detected_max = 0.0;
            let mut count = 0_i32;
            let mut sub_count = 0_i32;
            let total = 400_000;

            for i in 0..total {
                let s = smax * rng.gen::<f64>();
                let theta = consts::FRAC_PI_2 * rng.gen::<f64>();
                let z = 1.1 * max * rng.gen::<f64>();
                if theta > max_theta.at(s) {
                    continue;
                }
                let rate = double_diff_partial_rate(*a, *eta, s, theta, &mut dj);
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

            let volume = 1.1 * max * smax * consts::FRAC_PI_2;
            let frac = (count as f64) / (total as f64);
            let mc_integral = 4.0 * volume * frac;
            let mc_integral_est = 4.0 * volume * (sub_count as f64) / 200_000.0;
            let mc_error = (mc_integral_est - mc_integral).abs() / mc_integral;
            let error = (integral - mc_integral).abs() / integral;
            println!(
                "n = {:>4}, a = {:>4}, eta = {:>4}: integral = {:>9.3e}, mc = {:>9.3e} [diff = {:.2}%, estd conv = {:.2}%, success = {:.2}%]",
                n, a, eta, integral, mc_integral, 100.0 * error, 100.0 * mc_error, 100.0 * frac,
            );
            assert!(detected_max < 1.1 * max);
            assert!(error < 0.05);
        }

    }

    #[test]
    fn total_rate_accuracy() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let pts: Vec<(f64, f64)> = (0..10)
            .map(|_| {
                let a = (0.2_f64.ln() + (20_f64.ln() - 0.2_f64.ln()) * rng.gen::<f64>()).exp();
                let eta = (0.001_f64.ln() + (1.0_f64.ln() - 0.001_f64.ln()) * rng.gen::<f64>()).exp();
                (a, eta)
            })
            .collect();

        let num: usize = std::env::var("RAYON_NUM_THREADS")
            .map(|s| s.parse().unwrap_or(1))
            .unwrap_or(1);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num)
            .build()
            .unwrap();

        println!("Running on {:?}", pool);

        pool.install(|| {
            pts.into_par_iter().for_each(|(a, eta)| {
                if let Some(value) =  rate(a, eta) {
                    let n_max = (5.0 * (1.0 + 2.0 * a * a)) as i32;
                    let target = (1..=n_max).map(|n| partial_rate(n, a, eta).0).sum::<f64>();
                    let error = (target - value).abs() / target;
                    println!(
                        "[{:>2}]: a = {:>9.3e}, eta = {:>9.3e}: target = {:>9.3e}, lookup = {:>9.3e}, diff = {:.3e}",
                        rayon::current_thread_index().unwrap_or(0), a, eta, target, value, error,
                    );
                    // assert!(error < 1.0e-3);
                }
            })
        });
    }

    #[test]
    fn theta_bounds() {
        let (n, a, eta) = (1, 10.0, 0.1);
        let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
        let smax = sn / (1.0 + sn);
        let bound = ThetaBound::for_harmonic(n, a, eta);
        println!("{:?}", bound);

        for i in 0..10 {
            let s = (i as f64) * smax / 10.0;
            let theta = bound.at(s);
            println!("{} {}", s, theta);
        }
    }

    #[test]
    fn partial_spectra() {
        let (n, a, eta): (i32, f64, f64) = (15, 10.0, 0.1);
        let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
        let smax = sn / (1.0 + sn);
        let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);
        let theta_max = ThetaBound::for_harmonic(n, a, eta);

        let pts: Vec<(f64, f64)> = (0..100)
            .map(|i| (i as f64) * smax / 100.0)
            .map(|s| (s, single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj).0))
            .collect();

        let filename = format!("output/nlc_lp_sd_rate_{}_{}_{}.dat", n, a, eta);
        let mut file = File::create(&filename).unwrap();
        for pt in &pts {
            writeln!(file, "{:.6e} {:.6e}", pt.0, pt.1).unwrap();
        }
    }

    #[test]
    fn harmonic_limit() {
        let a_s: [f64; 9] = [0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0, 7.0, 10.0];
        let eta = 0.001;
        for &a in &a_s {
            let mut sum = 0.0;
            let mut n = 1;
            let nstop = loop {
                let (rate, _) = partial_rate(n, a, eta);
                sum += rate;
                if rate / sum < 1.0e-4 {
                    break n;
                }
                n += 1;
            };
            println!("a = {:.3e}, stopped at n = {}", a, nstop);
        }
    }

    #[test]
    fn create_rate_table() {
        const LOW_ETA_LIMIT: f64 = 0.001;
        const LOW_A_LIMIT: f64 = 0.2;
        // 20, 20, 40, 60
        const A_DENSITY: usize = 10; // points per order of magnitude
        const ETA_DENSITY: usize = 10;
        const N_COLS: usize = 20; // pts in a0 direction
        const N_ROWS: usize = 30; // pts in eta direction
        let mut table = [[0.0; N_COLS]; N_ROWS];

        let num: usize = std::env::var("RAYON_NUM_THREADS")
            .map(|s| s.parse().unwrap_or(1))
            .unwrap_or(1);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num)
            .build()
            .unwrap();

        println!("Running on {:?}", pool);

        let mut pts: Vec<(usize, usize, f64, f64, i32)> = Vec::new();
        for i in 0..N_ROWS {
            let eta = LOW_ETA_LIMIT * 10.0f64.powf((i as f64) / (ETA_DENSITY as f64));
            for j in 0..N_COLS {
                let a = LOW_A_LIMIT * 10.0f64.powf((j as f64) / (A_DENSITY as f64));
                let n_max = (5.0 * (1.0 + 2.0 * a * a)).ceil() as i32;
                pts.push((i, j, a, eta, n_max));
            }
        }

        let pts: Vec<(usize, usize, f64)> = pool.install(|| {
            pts.into_par_iter()
            .map(|(i, j, a, eta, n_max)| {
                let rate = (1..=n_max).map(|n| partial_rate(n, a, eta).0).sum::<f64>();
                println!("LP NLC [{:>3}]: eta = {:.3e}, a = {:.3e}, ln(rate) = {:.6e}", rayon::current_thread_index().unwrap_or(1),eta, a, rate.ln());
                (i, j, rate)
            })
            .collect()
        });

        for (i, j, rate) in pts {
            table[i][j] = rate;
        }

        let mut file = File::create("output/rate_table.rs").unwrap();
        //writeln!(file, "use std::f64::NEG_INFINITY;").unwrap();
        writeln!(file, "pub const N_COLS: usize = {};", N_COLS).unwrap();
        writeln!(file, "pub const N_ROWS: usize = {};", N_ROWS).unwrap();
        writeln!(file, "pub const MIN: [f64; 2] = [{:.12e}, {:.12e}];", LOW_A_LIMIT.ln(), LOW_ETA_LIMIT.ln()).unwrap();
        writeln!(file, "pub const STEP: [f64; 2] = [{:.12e}, {:.12e}];", consts::LN_10 / (A_DENSITY as f64), consts::LN_10 / (ETA_DENSITY as f64)).unwrap();
        writeln!(file, "pub const TABLE: [[f64; {}]; {}] = [", N_COLS, N_ROWS).unwrap();
        for row in table.iter() {
            let val = row.first().unwrap().ln();
            if val.is_finite() {
                write!(file, "\t[{:>18.12e}", val).unwrap();
            } else {
                write!(file, "\t[{:>18}", "NEG_INFINITY").unwrap();
            }
            for val in row.iter().skip(1) {
                let tmp = val.ln();
                if tmp.is_finite() {
                    write!(file, ", {:>18.12e}", tmp).unwrap();
                } else {
                    write!(file, ", {:>18}", "NEG_INFINITY").unwrap();
                }
            }
            writeln!(file, "],").unwrap();
        }
        writeln!(file, "];").unwrap();
    }
}
