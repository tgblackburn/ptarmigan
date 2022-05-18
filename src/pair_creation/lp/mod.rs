//! Nonlinear Breit-Wheeler pair creation in LP backgrounds

use std::f64::consts;
use num_complex::Complex;
use rand::prelude::*;
use crate::special_functions::*;
use crate::pwmci;
use super::{GAUSS_16_NODES, GAUSS_16_WEIGHTS, GAUSS_32_NODES, GAUSS_32_WEIGHTS};

mod tables;

/// Returns the sum, over harmonic index, of the partial nonlinear
/// Breit-Wheeler rates. Equivalent to calling
/// ```
/// let rate = (n_min..=n_max).map(|n| partial_rate(n, a, eta)).sum::<f64>();
/// ```
/// but implemented as a table lookup.
#[allow(unused_parens)]
pub(super) fn rate(a: f64, eta: f64) -> Option<f64> {
    if rate_too_small(a, eta) {
        println!("rate too small at {} {}", a, eta);
        Some(0.0)
    } else if tables::mid_range::contains(a, eta) {
        Some(tables::mid_range::interpolate(a, eta))
    } else if tables::contains(a, eta) {
        Some(tables::interpolate(a, eta))
    } else {
        println!("out of bounds at {} {}", a, eta);
        Some(0.0)
    }
}

/// Returns a pseudorandomly sampled n (harmonic order), s (lightfront momentum
/// transfer) and theta (azimuthal angle in the ZMF) for a pair creatione event that
/// occurs at normalized amplitude a and energy parameter eta.
pub(super) fn sample<R: Rng>(a: f64, eta: f64, rng: &mut R) -> (i32, f64, f64) {
    let frac = rng.gen::<f64>();

    let n = if tables::mid_range::contains(a, eta) {
        tables::mid_range::invert(a, eta, frac)
    } else {
        panic!("out of bounds at {} {}", a, eta);
    };

    let s_min = 0.5;
    let s_max = 0.5 + (0.25 - (1.0 + 0.5 * a * a) / (2.0 * (n as f64) * eta)).sqrt();

    let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);
    let theta_max = ThetaBound::for_harmonic(n, a, eta);

    let max = ceiling_double_diff_partial_rate(a, eta, &mut dj);

    // Rejection sampling
    let (s, theta) = loop {
        let s = s_min + (s_max - s_min) * rng.gen::<f64>();
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

    // Fix s, which is [1/2, s_max] at the moment
    let s = match rng.gen_range(0, 1) {
        0 => 1.0 - s,
        1 => s,
        _ => unreachable!(),
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

/// Rate, differential in s (fractional lightfront momentum transfer)
/// and theta (azimuthal angle).
/// Result valid only for s_min < s < s_max and 0 < theta < pi/2.
/// Multiply by alpha / eta to get dP/(ds dtheta dphase)
fn double_diff_partial_rate(a: f64, eta: f64, s: f64, theta: f64, dj: &mut DoubleBessel) -> f64 {
    let n = dj.n();

    let r_n_sqd = 2.0 * (n as f64) * eta * s * (1.0 - s) - (1.0 + 0.5 * a * a);

    let x = if r_n_sqd > 0.0 {
        a * r_n_sqd.sqrt() * theta.cos() / (eta * s * (1.0 - s))
    } else {
        return 0.0;
    };

    let y = a * a / (8.0 * eta * s * (1.0 - s));

    let j = dj.evaluate(x, y); // n-2, n-1, n, n+1, n+2

    let gamma = [j[2], 0.5 * (j[1] + j[3]), 0.25 * (j[0] + 2.0 * j[2] + j[4])];

    let h_s = 0.5 / (s * (1.0 - s)) - 1.0;

    (gamma[0].powi(2) + a * a * h_s * (gamma[1].powi(2) - gamma[0] * gamma[2])) / (2.0 * consts::PI)
}

/// Returns the largest possible value of [double_diff_partial_rate], multiplied by a small safety factor.
fn ceiling_double_diff_partial_rate(a: f64, eta: f64, dj: &mut DoubleBessel) -> f64 {
    let n = dj.n();
    let s_min = 0.5;
    let s_max = 0.5 + (0.25 - (1.0 + 0.5 * a * a) / (2.0 * (n as f64) * eta)).sqrt();

    // double-diff rate always maxmised along theta = 0
    let theta = 0.0;

    let centre = 0.5 + (0.25 - (1.0 + a * a) / (2.0 * (n as f64) * eta)).sqrt();
    let (lower, upper) = if centre.is_finite() {
        ((centre - 0.1).max(s_min), 0.5 * (centre + s_max))
    } else {
        (s_min, s_max)
    };

    let max = (0..40)
        .map(|i| {
            let s = lower + (upper - lower) * (i as f64) / 40.0;
                double_diff_partial_rate(a, eta, s, theta, dj)
            })
        .reduce(f64::max)
        .unwrap();

    1.1 * max
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

        let s_min = 0.5;
        let s_max = 0.5 + (0.25 - (1.0 + 0.5 * a * a) / (2.0 * (n as f64) * eta)).sqrt();

        for i in 0..16 {
            let z = (consts::PI * (i as f64) / 15.0).cos();
            s[i] = s_min + 0.5 * (z + 1.0) * (s_max - s_min);

            // Coordinates in (x,y) space where integration over theta begins
            let x = {
                let r_n = (2.0 * (n as f64) * eta * s[i] * (1.0 - s[i]) - (1.0 + 0.5 * a * a)).sqrt();
                a * r_n / (eta * s[i] * (1.0 - s[i]))
            };
        
            let y = a * a / (8.0 * eta * s[i] * (1.0 - s[i]));

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

/// Integrates [double_diff_partial_rate] over 0 < theta < 2 pi, ignoring contributions from theta_max < theta < pi/2.
fn single_diff_partial_rate(a: f64, eta: f64, s: f64, theta_max: f64, dj: &mut DoubleBessel) -> f64 {
    GAUSS_32_NODES.iter()
        // integrate over 0 to pi/2, then multiply by 4
        .map(|x| 0.5 * (x + 1.0) * theta_max)
        .zip(GAUSS_32_WEIGHTS.iter())
        .map(|(theta, w)| 4.0 * w * 0.5 * theta_max * double_diff_partial_rate(a, eta, s, theta, dj))
        .sum()
}

/// Integrates [double_diff_partial_rate] over s and theta, returning
/// the value of the integral.
/// Multiply by alpha / eta to get dP/(ds dtheta dphase)
fn partial_rate(n: i32, a: f64, eta: f64) -> f64 {
    if (n as f64) <= 2.0 * (1.0 + 0.5 * a * a) / eta {
        return 0.0;
    }

    let s_min = 0.5;
    let s_max = 0.5 + (0.25 - (1.0 + 0.5 * a * a) / (2.0 * (n as f64) * eta)).sqrt();

    let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);
    let theta_max = ThetaBound::for_harmonic(n, a, eta);

    // approx s where rate is maximised
    let s_peak = (0.5 + (0.25 - (1.0 + a * a) / (2.0 * (n as f64) * eta)).sqrt()).min(0.5);

    let integral = if s_peak < s_min + 2.0 * (s_max - s_min) / 3.0 {
        let s_peak = s_peak.min(s_min + (s_max - s_min) / 3.0);

        // split domain into two
        let bounds = [
            (s_min, s_peak),
            (s_peak, s_max)
        ];

        bounds.iter()
            .map(|(s0, s1)| -> f64 {
                GAUSS_32_NODES.iter()
                    .map(|x| s0 + 0.5 * (x + 1.0) * (s1 - s0))
                    .zip(GAUSS_32_WEIGHTS.iter())
                    .map(|(s, w)| {
                        let rate = single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj);
                        0.5 * w * (s1 - s0) * rate
                    })
                    .sum()
            })
            .sum()
    } else {
        // split domain into three: s_min to s_max-2d, s_max-2d to s_peak,
        // s_peak to s_max where d = s_max - s_peak
        let delta = s_max - s_peak;
        let bounds: [(f64, f64, &[f64], &[f64]); 3] = [
            (s_min,               s_max - 2.0 * delta, &GAUSS_16_NODES, &GAUSS_16_WEIGHTS),
            (s_max - 2.0 * delta, s_peak,              &GAUSS_32_NODES, &GAUSS_32_WEIGHTS),
            (s_peak,              s_max,               &GAUSS_32_NODES, &GAUSS_32_WEIGHTS),
        ];

        bounds.iter()
            .map(|(s0, s1, nodes, weights)| -> f64 {
                nodes.iter()
                    .map(|x| s0 + 0.5 * (x + 1.0) * (s1 - s0))
                    .zip(weights.iter())
                    .map(|(s, w)| {
                        let rate = single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj);
                        0.5 * w * (s1 - s0) * rate
                    })
                    .sum()
            })
            .sum()
    };

    integral
}

/// Returns the range of harmonics that contribute to the total rate.
fn sum_limits(a: f64, eta: f64) -> (i32, i32) {
    let n_min = (2.0f64 * (1.0 + 0.5 * a * a) / eta).ceil();
    let range = if a < 1.0 {
        2.0 + (2.0 + 20.0 * a * a) * (-(0.5 * eta).sqrt()).exp()
    } else {
        // if a < 10
        30.0 * (a * a + eta) * (-(1.5 * eta).sqrt()).exp()
    };

    let test = 0.25 - (1.0 + 0.5 * a * a) / (2.0 * (n_min as f64) * eta);
    if test <= 0.0 {
        ((n_min as i32) + 1, (n_min + 1.0 + range) as i32)
    } else {
        (n_min as i32, (n_min + range) as i32)
    }
}

/// Checks if a and eta are small enough such that the rate < exp(-200)
fn rate_too_small(a: f64, eta: f64) -> bool {
    eta.log10() < -1.0 - (a.log10() + 2.0).powi(2) / 4.5
}

/// Returns the total rate of emission by summing [partial_rate] over the relevant
/// range of harmonics and the cumulative density function.
/// Multiply by alpha / eta to get dP/dphase.
#[allow(unused)]
fn rate_by_summation(a: f64, eta: f64) -> (f64, [[f64; 2]; 16]) {
    let (n_min, n_max) = sum_limits(a, eta);
    let len = n_max - n_min + 1; // inclusive bounds

    let mut n_mode = n_min;
    let mut max = 0.0;
    let mut total = 0.0;
    let rates: Vec<[f64; 2]> = (n_min..=n_max)
        .map(|n| {
            let pr = partial_rate(n, a, eta);
            if pr > max {
                max = pr;
                n_mode = n;
            }
            total = total + pr;
            // if (n - n_min) % ((n_max - n_min) / 100) == 0 {
            //     println!("done n = {} of {} to {}...", n, n_min, n_max);
            // }
            [n as f64, total]
        })
        .collect();

    let n_mode = n_mode;
    let total = total;
    // println!("got n_mode = {}", n_mode);

    let mut cdf: [[f64; 2]; 16] = [[0.0, 0.0]; 16];
    cdf[0] = [rates[0][0], rates[0][1] / total];

    if rates.len() <= 16 {
        // Write all the rates
        for i in 1..=15 {
            cdf[i] = rates.get(i)
                .map(|r| [r[0], r[1] / total])
                .unwrap_or_else(|| [(i as f64) + (n_min as f64), 1.0]);
        }
    } else {
        // make sure we get the most probable harmonic
        if n_mode - n_min <= 3 {
            // first four harmonics
            for i in 1..=3 {
                cdf[i] = [rates[i][0], rates[i][1] / total];
            }

            // power-law spaced for n >= n_min + 4
            let alpha = if n_max - n_mode < 120 {1.5} else {2.0};
            for i in 4..=15 {
                let n = ((n_min + 4) as f64) + ((n_max - (n_min + 4)) as f64) * (((i - 4) as f64) / 11.0).powf(alpha);
                let limit = rates.last().unwrap()[0];
                let n = n.min(limit);
                cdf[i][0] = n;
                cdf[i][1] = pwmci::evaluate(n, &rates[..]).unwrap() / total;
            }
        } else {
            // n_mode >= n_min + 4
            // can't get everything b/t n_min and n_mode
            cdf[1] = {
                let i = ((n_mode - n_min) / 2) as usize;
                [rates[i][0], rates[i][1] / total]
            };

            // log-spaced for n >= n_mode
            // quadratically spaced
            // let delta = ((n_max as f64).ln() - (n_mode as f64).ln()) / 13.0;
            for i in 2..=15 {
                //let n = ((n_mode as f64).ln() + ((i - 2) as f64) * delta).exp();
                let n = (n_mode as f64) + ((n_max - n_mode) as f64) * (((i - 2) as f64) / 13.0).powi(2);
                let limit = rates.last().unwrap()[0];
                let n = n.min(limit);
                cdf[i][0] = n;
                cdf[i][1] = pwmci::evaluate(n, &rates[..]).unwrap() / total;
            }
        }
    }

    (total, cdf)
}

/// Returns the total rate of emission by integrating [partial_rate] as a function of
/// nover the relevant range of harmonics and the cumulative density function.
/// Multiply by alpha / eta to get dP/dphase.
#[allow(unused)]
fn rate_by_integration(a: f64, eta: f64) -> (f64, f64, [[f64; 2]; 16]) {
    // the number of harmonics is > 150
    assert!(a >= 5.0);

    let (n_min, n_max) = sum_limits(a, eta);
    let delta = (n_max - n_min) as f64;

    // Fill nodes of CDF
    let mut cdf = {
        let start = n_min as f64;
        let n_mode = (start + a.powf(1.1)).round();

        let mut cdf = [[0.0; 2]; 16];

        cdf[0][0] = start.round();
        cdf[1][0] = (0.5 * (start + n_mode)).round();

        for i in 2..=15 {
            let n = n_mode + ((n_max as f64) - n_mode) * (((i - 2) as f64) / 13.0).powi(2);
            cdf[i][0] = n.round();
        }

        cdf
    };

    cdf[0][1] = partial_rate(n_min, a, eta);

    for i in 1..3 {
        let min = cdf[i-1][0] as i32; // exclusive
        let max = cdf[i][0] as i32; // inclusive
        cdf[i][1] = cdf[i-1][1] + (min+1..=max).map(|n| partial_rate(n, a, eta)).sum::<f64>()
    }

    for i in 3..16 {
        let x_min = cdf[i-1][0]; // exclusive
        let x_max = cdf[i][0]; // inclusive
        let delta = x_max - x_min;

        // print!("from {} to {} with {} nodes: ", x_min, x_max, if delta < 4.0 {"all"} else if delta < 8.0 {"2"} else if delta < 64.0 {"4"} else {"8"});

        let integral = if delta < 4.0 {
            let min = x_min as i32;
            let max = x_max as i32;
            (min+1..=max).map(|n| partial_rate(n, a, eta)).sum::<f64>()
        } else {
            let (nodes, weights): (&[f64], &[f64]) = if delta < 8.0 {
                (&GAUSS_2_NODES, &GAUSS_2_WEIGHTS)
            } else if delta < 64.0 {
                (&GAUSS_4_NODES, &GAUSS_4_WEIGHTS)
            } else {
                (&GAUSS_8_NODES, &GAUSS_8_WEIGHTS)
            };

            nodes.iter()
                .map(|x| x_min + 0.5 * (x + 1.0) * delta)
                .map(|n| {
                    // let frac = n.fract();
                    // let n = n.floor() as i32;
                    // (1.0 - frac) * partial_rate(n, a, eta) + frac * partial_rate(n+1, a, eta)
                    // print!("{} ", n.round());
                    partial_rate(n.round() as i32, a, eta)
                })
                .zip(weights.iter())
                .map(|(y, w)| 0.5 * w * delta * y)
                .sum()
        };

        // println!("");
        //let y_max = partial_rate(cdf[i][0]  as i32, a, eta);

        cdf[i][1] = cdf[i-1][1] + integral;
    }

    let cdf_total = cdf[15][1];
    for i in 0..16 {
        cdf[i][1] = cdf[i][1] / cdf_total;
    }

    let total: f64 = GAUSS_32_NODES.iter()
        .map(|x| (n_min as f64) + 0.5 * (x + 1.0) * delta)
        .zip(GAUSS_32_WEIGHTS.iter())
        .map(|(n, w)| 0.5 * w * delta * partial_rate(n.round() as i32, a, eta))
        .sum();

    (total, cdf_total, cdf)
}

static GAUSS_2_NODES: [f64; 2] = [
    -0.57735026918962576451,
    0.57735026918962576451,
];

static GAUSS_2_WEIGHTS: [f64; 2] = [
    1.0,
    1.0,
];

static GAUSS_4_NODES: [f64; 4] = [
    -0.86113631159405257522,
    -0.33998104358485626480,
    0.33998104358485626480,
    0.86113631159405257522,
];

static GAUSS_4_WEIGHTS: [f64; 4] = [
    0.3478548451374538574,
    0.6521451548625461426,
    0.6521451548625461426,
    0.3478548451374538574,
];

static GAUSS_8_NODES: [f64; 8] = [
    -0.96028985649753623168,
    -0.79666647741362673959,
    -0.52553240991632898582,
    -0.18343464249564980494,
    0.18343464249564980494,
    0.52553240991632898582,
    0.79666647741362673959,
    0.96028985649753623168,
];

static GAUSS_8_WEIGHTS: [f64; 8] = [
    0.101228536290376259,
    0.22238103445337447,
    0.313706645877887287,
    0.3626837833783619830,
    0.3626837833783619830,
    0.313706645877887287,
    0.22238103445337447,
    0.101228536290376259,
];

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use rand::prelude::*;
    use rand_xoshiro::*;
    use rayon::prelude::*;
    use super::*;

    #[test]
    fn integration() {
        let (n, a, eta): (i32, f64, f64) = (21, 3.0, 1.0);
        let n_min = (2.0 * (1.0 + 0.5 * a * a) / eta).ceil() as i32;

        // bounds on s
        let s_min = 0.5;
        let s_max = 0.5 + (0.25 - (1.0 + 0.5 * a * a) / (2.0 * (n as f64) * eta)).sqrt();
        println!("a = {}, eta = {}, n = {} [min = {}], {:.3e} < s < {:.3e}", a, eta, n, n_min, s_min, s_max);

        let nodes: Vec<f64> = (0..100).map(|i| (i as f64) / 100.0).collect();
        let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);

        let filename = format!("output/nbw_lp_dd_rate_{}_{}_{}.dat", n, a, eta);
        let mut file = File::create(&filename).unwrap();
        let mut max = 0.0;
        let mut pt = (0.0, 0.0);
        let mut total = 0.0;

        for s in nodes.iter().map(|x| s_min + (s_max - s_min) * x) {
            for theta in nodes.iter().map(|x| x * consts::FRAC_PI_2) {
                let rate = double_diff_partial_rate(a, eta, s, theta, &mut dj);
                total += rate;
                if rate > max {
                    max = rate;
                    pt = (s, theta);
                }
                writeln!(file, "{:.6e} {:.6e} {:.6e}", s, theta, rate).unwrap();
            }
        }

        let filename = format!("output/nbw_lp_dd_rate_bounds_{}_{}_{}.dat", n, a, eta);
        let mut file = File::create(&filename).unwrap();
        let theta_max = ThetaBound::for_harmonic(n, a, eta);
        for s in nodes.iter().map(|x| s_min + (s_max - s_min) * x) {
            let theta = theta_max.at(s);
            writeln!(file, "{:.6e} {:.6e}", s, theta).unwrap();
        }

        let integral = 4.0 * 2.0 * total * 1.0e-4 * (s_max - s_min) * consts::FRAC_PI_2;
        let rate = partial_rate(n, a, eta);
        //let predicted_max = ceiling_double_diff_partial_rate(a, eta, &mut dj);
        println!("integral = {:.6e} [predicted {:.6e}]", integral, rate);
        println!("max = {:.6e} @ s = {:.3e}, theta = {:.3e}", max, pt.0, pt.1);
    }

    #[test]
    fn rate_ceiling() {
        let pts = vec![
            (0.5, 0.1, 23),
            (0.5, 0.1, 24),
            (2.0, 0.1, 61),
            (2.0, 0.1, 70),
            (2.0, 0.1, 80),
            (2.0, 0.1, 100),
            (2.0, 0.01, 601),
            (2.0, 0.01, 700),
            (2.0, 0.01, 800),
            (2.0, 0.01, 1000),
            (6.0, 1.0, 39),
            (6.0, 1.0, 80),
            (6.0, 1.0, 160),
            (6.0, 1.0, 350),
            (6.0, 0.1, 400),
            (6.0, 0.1, 500),
            (6.0, 0.1, 750),
            (6.0, 0.1, 1000),
            (6.0, 0.01, 3801),
            (6.0, 0.01, 3900),
            (6.0, 0.01, 4000),
            (6.0, 0.01, 4200),
            (15.0, 1.0, 228),
            (15.0, 1.0, 500),
            (15.0, 1.0, 1000),
            (15.0, 1.0, 2000),
            (15.0, 0.1, 2280),
            (15.0, 0.1, 3800),
            (15.0, 0.1, 4500),
            (15.0, 0.1, 6800),
            (15.0, 0.01, 22800),
            (15.0, 0.01, 24000),
            (15.0, 0.01, 26000),
            (15.0, 0.01, 28500),
        ];

        for (a, eta, n) in pts.into_iter() {
            let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);

            let target = {
                let s_min = 0.5;
                let s_max = 0.5 + (0.25 - (1.0 + 0.5 * a * a) / (2.0 * (n as f64) * eta)).sqrt();
                (0..1000)
                    .map(|i| {
                        let s = s_min + (s_max - s_min) * (i as f64) / 1000.0;
                            double_diff_partial_rate(a, eta, s, 0.0, &mut dj)
                        })
                    .reduce(f64::max)
                    .unwrap()
            };

            let max = ceiling_double_diff_partial_rate(a, eta, &mut dj);
            let err = (target - max) / target;

            println!(
                "a = {:>9.3e}, eta = {:>9.3e}, n = {:>4} => max = {:>9.3e}, predicted = {:>9.3e}, err = {:.2}%",
                a, eta, n, target, max, 100.0 * err,
            );

            assert!(err < 0.0);
        }
    }

    #[test]
    fn sum_vs_integral() {
        let pts = [
            (5.0, 1.0, 1.300522926497e-1),
            (5.0, 0.1, 7.770395919096e-5),
            (5.0, 0.01, 5.173715305605e-27),
            (7.5, 1.0, 2.278728143894e-1),
            (7.5, 0.1, 7.618916024118e-4),
            (7.5, 0.05, 8.803325768014e-6),
            (10.0, 1.0, 3.206858045905e-1),
            (10.0, 0.1, 2.623276241871e-3),
            (10.0, 0.05, 7.481145913658e-5),
        ];

        for (a, eta, target) in &pts {
            //let (target, tcdf) = rate_by_summation(*a, *eta);
            let (value, value2, _cdf) = rate_by_integration(*a, *eta);
            let error = (target - value) / target;
            let error2 = (target - value2) / target;
            println!("a = {}, eta = {}: target = {:.12e}, value = {:.12e} [{:.12e}], err = {:.2}% [{:.2}%]", a, eta, target, value, value2, 100.0 * error, 100.0 * error2);
            // for row in &tcdf {
            //     println!("\t{:.6e} {:.3e}", row[0], row[1]);
            // }
            // for row in &cdf {
            //     println!("\t{:.6e} {:.3e}", row[0], row[1]);
            // }
        }
    }

    #[test]
    fn total_rate() {
        // let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let num: usize = std::env::var("RAYON_NUM_THREADS")
            .map(|s| s.parse().unwrap_or(1))
            .unwrap_or(1);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num)
            .build()
            .unwrap();

        // let pts: Vec<_> = (0..1000)
        //     .map(|_i| {
        //         let a = (0.05_f64.ln() + (10_f64.ln() - 0.05_f64.ln()) * rng.gen::<f64>()).exp();
        //         let eta = (0.002_f64.ln() + (2.0_f64.ln() - 0.002_f64.ln()) * rng.gen::<f64>()).exp();
        //         (a, eta)
        //     })
        //     .collect();

        let pts = include!("total_rate_test.in");

        let pts: Vec<_> = pool.install(|| {
            pts.into_par_iter()
                .filter(|(a, eta, _)| {
                    !rate_too_small(*a, *eta)
                })
                .map(|(a, eta, target)| {
                    // let target = if a < 5.0 {
                    //     rate_by_summation(a, eta).0
                    // } else {
                    //     rate_by_integration(a, eta).0
                    // };
                    let value = rate(a, eta).unwrap();
                    let error = (target - value) / target;
                    println!("a = {:.6e}, eta = {:.6e}, target = {:.6e}, value = {:.6e}, err = {:.2}%", a, eta, target, value, 100.0 * error);
                    //println!("\t({:.12e}, {:.12e}, {:.12e}),", a, eta, target);
                    (a, eta, target, value, error)
                })
                .collect()
            });

        let rms_error: f64 = pts.iter()
            .fold(0_f64, |acc, (_, _, _, _, error)| acc.hypot(*error));
        let rms_error = rms_error / (pts.len() as f64);

        // let filename = format!("output/nbw_lp_rate_error.dat");
        // let mut file = File::create(&filename).unwrap();
        // for (a, eta, target, value, error) in &pts {
        //     writeln!(file, "{:.6e} {:.6e} {:.6e} {:.6e} {:.6e}", a, eta, target, value, error).unwrap();
        // }

        println!("=> rms error = {:.3}%", 100.0 * rms_error);
        assert!(rms_error < 0.01);
    }

    #[test]
    fn slice_total_rate() {
        let a = 1.0;
        let (eta_min, eta_max) = (0.03_f64, 2.0_f64);

        let mut rms_error = 0_f64;
        let mut count = 0;
        for i in 0..100 {
            let eta = (eta_min.ln() + (i as f64) * (eta_max.ln() - eta_min.ln()) / 100.0).exp();
            if !tables::mid_range::contains(a, eta) {
                continue;
            }
            let target = rate_by_summation(a, eta).0;
            let value = rate(a, eta).unwrap();
            let error = (target - value) / target;
            rms_error = rms_error.hypot(error);
            count +=1;
            println!("{:.6e} {:.6e} {:.6e} {:.6e} {:.3}%", a, eta, target, value, 100.0 * error);
            assert!(error.abs() < 0.1);
        }

        let rms_error = rms_error / (count as f64);
        println!("=> rms error = {:.3}%", 100.0 * rms_error);
        assert!(rms_error < 0.01);
    }

    #[test]
    fn summation_limits() {
        let max_error = 1.0e-4;

        let filename = format!("output/nbw_lp_rates.dat");
        let mut file = File::create(&filename).unwrap();

        for a in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0].iter() { //[1.0, 2.0, 3.0, 5.0, 10.0].iter() {
            for eta in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0].iter() {
                // if a * eta < 0.02 {
                //     continue;
                // }

                let n_min = (2.0f64 * (1.0 + 0.5 * a * a) / eta).ceil() as i32;
                //let n_stop = n_min + ((30.0 * (a * a + eta) * (-(1.5 * eta).sqrt()).exp()) as i32);
                let mut n = n_min;
                let mut total = 0.0;
                let mut step = 1;
                let mut prev = 0.0;

                loop {
                    let rate = partial_rate(n, *a, *eta);
                    total += 0.5 * (rate + prev) * (step as f64);
                    prev = rate;
                    println!("n = {:>4}, rate = {:>9.3e}, total = {:>9.3e}", n, rate, total);
                    writeln!(file, "{:.6e} {:.6e} {} {:.6e} {:.6e}", a, eta, n, rate, total).unwrap();

                    // if n > n_stop {
                    //     break;
                    // }

                    if n - n_min > 10_000 {
                        step = 3000;
                    } else if n - n_min > 3000 {
                        step = 1000;
                    } else if n - n_min > 1000 {
                        step = 300;
                    } else if n - n_min > 300 {
                        step = 100;
                    } else if n - n_min > 100 {
                        step = 30;
                    } else if n - n_min > 30 {
                        step = 10;
                    } else if n - n_min > 10 {
                        step = 3;
                    }

                    n += step;

                    if rate / total < max_error {
                        break;
                    }
                }
            }
        }
    }

    #[test]
    fn create_rate_tables() {
        let do_mid_range = true;
        let do_high_range = false;

        let num: usize = std::env::var("RAYON_NUM_THREADS")
            .map(|s| s.parse().unwrap_or(1))
            .unwrap_or(1);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num)
            .build()
            .unwrap();

        if do_mid_range {
            const LN_MIN_A: f64 = -7.0 * consts::LN_10 / 5.0; // ~0.04
            const LN_MAX_ETA_PRIME: f64 = consts::LN_2; // 2.0
            const A_DENSITY: usize = 20; // points per order of magnitude
            const ETA_PRIME_DENSITY: usize = 4; // points per harmonic step
            const N_COLS: usize = 36; // points in a0
            const N_ROWS: usize = 157; // points in eta_prime

            let mut pts = vec![];
            for i in 0..N_ROWS {
                for j in 0..N_COLS {
                    // eta' = 2 density / (i + 1)
                    let eta_prime = LN_MAX_ETA_PRIME.exp() / (1.0 + (i as f64) / (ETA_PRIME_DENSITY as f64));
                    let a = LN_MIN_A.exp() * 10_f64.powf((j as f64) / (A_DENSITY as f64));
                    pts.push((i, j, a, eta_prime));
                }
            }

            let pts: Vec<_> = pool.install(|| {
                pts.into_par_iter().map(|(i, j, a, eta_prime)| {
                    let eta = (1.0 + 0.5 * a * a) * eta_prime;
                    let (total, cdf) = rate_by_summation(a, eta);
                    println!(
                        "LP NBW [{:>3}]: eta = {:.3e}, a = {:.3e}, ln(rate) = {:.6e}",
                        rayon::current_thread_index().unwrap_or(1),
                        eta, a, total.ln()
                    );
                    (i, j, total, cdf)
                })
                .collect()
            });

            // Build rate table
            let mut table = [[0.0; N_COLS]; N_ROWS];
            for (i, j, rate, _) in pts.iter() {
                table[*i][*j] = *rate;
            }

            let mut file = File::create("output/nbw_mid_a_rate_table.rs").unwrap();
            // writeln!(file, "pub const N_COLS: usize = {};", N_COLS).unwrap();
            // writeln!(file, "pub const N_ROWS: usize = {};", N_ROWS).unwrap();
            writeln!(file, "pub const LN_MIN_A: f64 = {:.12e};", LN_MIN_A).unwrap();
            writeln!(file, "pub const LN_MAX_ETA_PRIME: f64 = {:.12e};", LN_MAX_ETA_PRIME).unwrap();
            writeln!(file, "pub const LN_MIN_ETA_PRIME: f64 = {:.12e};", LN_MAX_ETA_PRIME - (1.0 + ((N_ROWS - 1) as f64) / (ETA_PRIME_DENSITY as f64)).ln()).unwrap();
            writeln!(file, "pub const LN_A_STEP: f64 = {:.12e};", consts::LN_10 / (A_DENSITY as f64)).unwrap();
            writeln!(file, "pub const ETA_PRIME_DENSITY: f64 = {:.12e};", ETA_PRIME_DENSITY).unwrap();
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

            let mut file = File::create("output/nbw_mid_a_cdf_table.rs").unwrap();
            writeln!(file, "pub const N_COLS: usize = {};", N_COLS).unwrap();
            //writeln!(file, "pub const N_ROWS: usize = {};", N_ROWS).unwrap();
            writeln!(file, "pub const LN_MIN_A: f64 = {:.12e};", LN_MIN_A).unwrap();
            writeln!(file, "pub const LN_MAX_ETA_PRIME: f64 = {:.12e};", LN_MAX_ETA_PRIME).unwrap();
            // writeln!(file, "pub const LN_MIN_ETA_PRIME: f64 = {:.12e};", LN_MAX_ETA_PRIME - (1.0 + ((N_ROWS - 1) as f64) / (ETA_PRIME_DENSITY as f64)).ln()).unwrap();
            writeln!(file, "pub const LN_A_STEP: f64 = {:.12e};", consts::LN_10 / (A_DENSITY as f64)).unwrap();
            writeln!(file, "pub const TABLE: [[[f64; 2]; 16]; {}] = [", N_COLS * N_ROWS).unwrap();
            for (_, _, _, cdf) in &pts {
                write!(file, "\t[").unwrap();
                for entry in cdf.iter().take(15) {
                    write!(file, "[{:>18.12e}, {:>18.12e}], ", entry[0], entry[1]).unwrap();
                }
                writeln!(file, "[{:>18.12e}, {:>18.12e}]],", cdf[15][0], cdf[15][1]).unwrap();
            }
            writeln!(file, "];").unwrap();
        }

        if do_high_range {
            const LN_MIN_A: f64 = -consts::LN_10; // 0.1
            const A_DENSITY: usize = 20; // points per order of magnitude
            const N_COLS: usize = 2 * A_DENSITY + 1; // points in a0, a <= 10

            const MIN_ETA: f64 = 0.002;
            const ETA_DENSITY: usize = 20;
            const N_ROWS: usize = 3 * ETA_DENSITY + 1; // points in eta, eta <= 2

            let mut pts = vec![];
            for i in 0..N_ROWS {
                for j in 0..N_COLS {
                    let a = LN_MIN_A.exp() * 10_f64.powf((j as f64) / (A_DENSITY as f64));
                    let eta = MIN_ETA * 10_f64.powf((i as f64) / (ETA_DENSITY as f64));
                    pts.push((i, j, a, eta));
                }
            }

            let pts: Vec<_> = pool.install(|| {
                pts.into_par_iter().map(|(i, j, a, eta)| {
                    let (total, cdf) = if a > 5.0 {
                        let (total, _, cdf) = rate_by_integration(a, eta);
                        (total, cdf)
                    } else if !rate_too_small(a, 1.2 * eta) { // pad the edge of the table
                        rate_by_summation(a, eta)
                    } else {
                        (0.0, [[0.0; 2]; 16])
                    };

                    println!(
                        "LP NBW [{:>3}]: eta = {:.3e}, a = {:.3e}, ln(rate) = {:.6e}",
                        rayon::current_thread_index().unwrap_or(1),
                        eta, a, total.ln()
                    );
                    (i, j, total, cdf)
                })
                .collect()
            });

            // Build rate table
            let mut table = [[0.0; N_COLS]; N_ROWS];
            for (i, j, rate, _) in pts.iter() {
                table[*i][*j] = *rate;
            }

            let mut file = File::create("output/nbw_rate_table.rs").unwrap();
            writeln!(file, "use std::f64::NEG_INFINITY;").unwrap();
            writeln!(file, "pub const N_COLS: usize = {};", N_COLS).unwrap();
            writeln!(file, "pub const N_ROWS: usize = {};", N_ROWS).unwrap();
            writeln!(file, "pub const LN_MIN_A: f64 = {:.12e};", LN_MIN_A).unwrap();
            writeln!(file, "pub const LN_MIN_ETA: f64 = {:.12e};", MIN_ETA.ln()).unwrap();
            writeln!(file, "pub const LN_A_STEP: f64 = {:.12e};", consts::LN_10 / (A_DENSITY as f64)).unwrap();
            writeln!(file, "pub const LN_ETA_STEP: f64 = {:.12e};", consts::LN_10 / (ETA_DENSITY as f64)).unwrap();
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

            let mut file = File::create("output/nbw_cdf_table.rs").unwrap();
            writeln!(file, "pub const N_COLS: usize = {};", N_COLS).unwrap();
            writeln!(file, "pub const N_ROWS: usize = {};", N_ROWS).unwrap();
            writeln!(file, "pub const MIN: [f64; 2] = [{:.12e}, {:.12e}];", LN_MIN_A, MIN_ETA.ln()).unwrap();
            writeln!(file, "pub const STEP: [f64; 2] = [{:.12e}, {:.12e}];", consts::LN_10 / (A_DENSITY as f64), consts::LN_10 / (ETA_DENSITY as f64)).unwrap();
            writeln!(file, "pub const TABLE: [[[f64; 2]; 16]; {}] = [", N_COLS * N_ROWS).unwrap();
            for (_, _, _, cdf) in &pts {
                write!(file, "\t[").unwrap();
                for entry in cdf.iter().take(15) {
                    write!(file, "[{:>18.12e}, {:>18.12e}], ", entry[0], entry[1]).unwrap();
                }
                writeln!(file, "[{:>18.12e}, {:>18.12e}]],", cdf[15][0], cdf[15][1]).unwrap();
            }
            writeln!(file, "];").unwrap();
        }
    }
}
