//! Nonlinear Breit-Wheeler pair creation in LP backgrounds

use std::f64::consts;
use num_complex::Complex;
use crate::special_functions::*;
use crate::pwmci;
use super::{GAUSS_16_NODES, GAUSS_16_WEIGHTS, GAUSS_32_NODES, GAUSS_32_WEIGHTS};

/// Returns the sum, over harmonic index, of the partial nonlinear
/// Breit-Wheeler rates. Equivalent to calling
/// ```
/// let rate = (n_min..=n_max).map(|n| partial_rate(n, a, eta)).sum::<f64>();
/// ```
/// but implemented as a table lookup.
pub(super) fn rate(_a: f64, _eta: f64) -> Option<f64> {
    Some(0.0)
}

/// Rate, differential in s (fractional lightfront momentum transfer)
/// and theta (azimuthal angle).
/// Result valid only for s_min < s < s_max and 0 < theta < pi/2.
/// Multiply by alpha / eta to get dP/(ds dtheta dphase)
fn double_diff_partial_rate(a: f64, eta: f64, s: f64, theta: f64, dj: &mut DoubleBessel) -> f64 {
    let n = dj.n();

    let x = {
        let r_n = (2.0 * (n as f64) * eta * s * (1.0 - s) - (1.0 + 0.5 * a * a)).sqrt();
        a * r_n * theta.cos() / (eta * s * (1.0 - s))
    };

    let y = a * a / (8.0 * eta * s * (1.0 - s));

    let j = dj.evaluate(x, y); // n-2, n-1, n, n+1, n+2

    let gamma = [j[2], 0.5 * (j[1] + j[3]), 0.25 * (j[0] + 2.0 * j[2] + j[4])];

    let h_s = 0.5 / (s * (1.0 - s)) - 1.0;

    (gamma[0].powi(2) + a * a * h_s * (gamma[1].powi(2) - gamma[0] * gamma[2])) / (2.0 * consts::PI)
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
                        w * (s1 - s0) * rate
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
                        w * (s1 - s0) * rate
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
    // if 1 < a < 10
    let range = 30.0 * (a * a + eta) * (-(1.5 * eta).sqrt()).exp();

    let test = 0.25 - (1.0 + 0.5 * a * a) / (2.0 * n_min * eta);
    if test <= 0.0 {
        ((n_min as i32) + 1, (n_min + 1.0 + range) as i32)
    } else {
        (n_min as i32, (n_min + range) as i32)
    }
}

/// Returns the total rate of emission by summing [partial_rate] over the relevant
/// range of harmonics and the cumulative density function.
/// Multiply by alpha / eta to get dP/dphase.
#[allow(unused)]
fn rate_by_summation(a: f64, eta: f64) -> (f64, [[f64; 2]; 16]) {
    let (n_min, n_max) = sum_limits(a, eta);
    let len = n_max - n_min + 1; // inclusive bounds

    let mut total = 0.0;
    let rates: Vec<[f64; 2]> = (n_min..=n_max)
        .map(|n| {
            total = total + partial_rate(n, a, eta);
            [n as f64, total]
        })
        .collect();

    let total = total;

    let mut cdf: [[f64; 2]; 16] = [[0.0, 0.0]; 16];
    cdf[0] = [rates[0][0], rates[0][1] / total];

    if rates.len() <= 16 {
        // Write all the rates
        for i in 1..=15 {
            cdf[i] = rates.get(i)
                .map(|r| [r[0], r[1] / total])
                .unwrap_or_else(|| [(i as f64) + (n_min as f64), 1.0]);
        }
    } else if rates.len() <= 100 {
        // first four harmonics
        for i in 1..=3 {
            cdf[i] = [rates[i][0], rates[i][1] / total];
        }
        // log-spaced for n >= n_min + 4
        let delta = ((n_max as f64).ln() - ((n_min + 4) as f64).ln()) / 11.0;
        for i in 4..=15 {
            let n = (((n_min + 4) as f64).ln() + ((i - 4) as f64) * delta).exp();
            let limit = rates.last().unwrap()[0];
            let n = n.min(limit);
            cdf[i][0] = n;
            cdf[i][1] = pwmci::evaluate(n, &rates[..]).unwrap() / total;
        }
    } else {
        // Sample CDF at 16 log-spaced points
        let delta = ((n_max as f64).ln() - (n_min as f64).ln())/ 15.0;
        for i in 1..=15 {
            let n = ((n_min as f64).ln() + (i as f64) * delta).exp();
            let limit = rates.last().unwrap()[0];
            let n = n.min(limit);
            cdf[i][0] = n;
            cdf[i][1] = pwmci::evaluate(n, &rates[..]).unwrap() / total;
        }
    }

    (total, cdf)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use rand::prelude::*;
    use rand_xoshiro::*;
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
    fn total_rate() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let pts: Vec<_> = (0..1000)
            .map(|_| {
                let a = (0.1_f64.ln() + (1_f64.ln() - 0.1_f64.ln()) * rng.gen::<f64>()).exp();
                let eta = (0.1_f64.ln() + (1.0_f64.ln() - 0.1_f64.ln()) * rng.gen::<f64>()).exp();
                (a, eta)
            })
            .collect();

        let pts: Vec<_> = pts.iter()
            .map(|(a, eta)| {
                (*a, *eta, rate_by_summation(*a, *eta).0)
            })
            .collect();

        let filename = format!("output/nbw_lp_rate.dat");
        let mut file = File::create(&filename).unwrap();
        for (a, eta, rate) in &pts {
            writeln!(file, "{:.6e} {:.6e} {:.6e}", a, eta, rate).unwrap();
        }
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
}