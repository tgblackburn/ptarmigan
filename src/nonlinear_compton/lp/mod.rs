//! Rates and spectra for linearly polarized backgrounds
use std::f64::consts;
use num_complex::Complex;
use rand::prelude::*;
use crate::special_functions::*;
use crate::pwmci;
use crate::geometry::StokesVector;
use crate::quadrature::*;

mod rate_table;
mod cdf_table;
mod linear;
pub mod classical;

/// Rate, differential in s (fractional lightfront momentum transfer)
/// and theta (azimuthal angle).
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

/// Returns the largest value of the double-differential rate, multiplied by a small safety factor.
fn ceiling_double_diff_partial_rate(a: f64, eta: f64, dj: &mut DoubleBessel) -> f64 {
    let n = dj.n();
    let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
    let smax = sn / (1.0 + sn);

    let max = if n == 1 {
        let theta = consts::FRAC_PI_2;
        f64::max(
            double_diff_partial_rate(a, eta, 0.01 * smax, theta, dj),
            double_diff_partial_rate(a, eta, 0.99 * smax, theta, dj)
        )
    } else if a <= 2.0 {
        // we know that n > 1
        let theta = 0.0;
        // search 0.4 < s / s_max < 1
        (1..=32)
            .map(|i| {
                let s = smax * (0.4 + 0.6 * (i as f64) / 32.0);
                double_diff_partial_rate(a, eta, s, theta, dj)
            })
            .reduce(f64::max)
            .unwrap()
    } else if a <= 5.0 {
        // n > 1, 2 < a < 5
        let n_switch = (1.3 * a.powf(1.2)).ceil() as i32;
        if n < n_switch {
            // linear fit in theta_opt as a function of log(n)
            let theta = consts::FRAC_PI_2 * (1.0 - (n as f64).ln() / (n_switch as f64).ln());
            // search 0 < s / s_max < 1.0
            (0..=32)
                .map(|i| {
                    let s = smax * (i as f64) / 32.0;
                    double_diff_partial_rate(a, eta, s, theta, dj)
                })
                .reduce(f64::max)
                .unwrap()
        } else {
            let theta = 0.0;
            // search max(x_min, 0.4) < s / s_max < 1.0
            let x_min = (4.0 * (n as f64) * eta / (3.0 * a * a + 4.0 * (n as f64) * eta)).max(0.4);
            // in 2 < a < 5, harmonic cutoff is now ~twice as large as in v1.2.1
            let n_max = sum_limit(a, eta);
            let n = if n < n_max / 2 { 32 } else { 64 };
            (1..=n)
                .map(|i| {
                    let s = smax * (x_min + (1.0 - x_min) * (i as f64) / (n as f64));
                    double_diff_partial_rate(a, eta, s, theta, dj)
                })
                .reduce(f64::max)
                .unwrap()
        }
    } else {
        // n > 1, a > 5
        let n_switch = (a * a).ceil() as i32;

        if n < n_switch {
            let theta = {
                let n = n as f64;
                let n_switch = n_switch as f64;
                consts::FRAC_PI_2 * (n.powf(-0.7) - n_switch.powf(-0.7)) / (1.0 - n_switch.powf(-0.7))
            };

            // search in 0.3 < s / s_max < 1
            (1..=64)
                .map(|i| {
                    let s = smax * (0.3 + 0.7 * (i as f64) / 64.0);
                    double_diff_partial_rate(a, eta, s, theta, dj)
                })
                .reduce(f64::max)
                .unwrap()
        } else {
            let theta = 0.0;
            // search x_min < s < x_peak
            let x_min = (4.0 * (n as f64) * eta / (3.0 * a * a + 4.0 * (n as f64) * eta)).max(0.3);
            let x_peak = (1.0 + sn) / (2.0 + sn);
            let len = if n > 10 * n_switch {64} else if n > 3 * n_switch {42} else {32};
            (1..=len)
                .map(|i| {
                    let s = smax * (x_min + (x_peak - x_min) * (i as f64) / (len as f64));
                    double_diff_partial_rate(a, eta, s, theta, dj)
                })
                .reduce(f64::max)
                .unwrap()
        }
    };

    1.15 * max
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

    fn for_harmonic_low_eta(n: i32, a: f64) -> Self {
        let mut s = [0.0; 16];
        let mut f = [0.0; 16];

        for i in 0..16 {
            let z = (consts::PI * (i as f64) / 15.0).cos();
            let v = 0.5 * (z + 1.0);
            s[i] = v;

            // Coordinates in (x,y) space where integration over theta begins
            let x = 2.0 * (n as f64) * a * (v * (1.0 - v) / (1.0 + 0.5 * a * a)).sqrt();
            let y = 0.25 * (n as f64) * a * a * v / (1.0 + 0.5 * a * a);

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

    /// Returns the largest theta that contributes meaningfully to the
    /// partial rate at *fixed s* (if ThetaBound was constructed using
    /// [`for_harmonic`] or *fixed v* (if ThetaBound was constructed using
    /// [`for_harmonic_low_eta`])
    fn at(&self, arg: f64) -> f64 {
        let mut val = self.f[15];

        for i in 1..16 {
            // s[i] is stored backwards, decreasing from s_max
            if arg > self.s[i] {
                let weight = (arg - self.s[i-1]) / (self.s[i] - self.s[i-1]);
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
/// Multiply by alpha / eta to get dP/dphase
#[allow(unused)]
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

/// Returns the largest harmonic that contributes to the total rate.
#[allow(unused)]
fn sum_limit(a: f64, eta: f64) -> i32 {
    let m = 3.0 - 0.4 * eta.powf(0.25);
    // let n_max = 6.0 + 2.5 * a.powf(m);
    let n_max = 6.0 * (1.0 + a * a) + 2.0 * a.powf(m);
    n_max.ceil() as i32
}

/// Returns the total rate of nonlinear Compton scattering.
/// Equivalent to calling
/// ```
/// let nmax = sum_limit(a, eta);
/// let rate = (1..=nmax).map(|n| partial_rate(n, a, eta)).sum::<f64>();
/// ```
/// but implemented as a table lookup.
/// Multiply by alpha / eta to get dP/dphase.
#[allow(unused_parens)]
pub(super) fn rate(a: f64, eta: f64) -> Option<f64> {
    let (x, y) = (a.ln(), eta.ln());

    if x < rate_table::MIN[0] {
        Some(linear::rate(a, eta))
    } else if y < rate_table::MIN[1] {
        // rate is proportional to eta as eta -> 0
        let ix = ((x - rate_table::MIN[0]) / rate_table::STEP[0]) as usize;
        let dx = (x - rate_table::MIN[0]) / rate_table::STEP[0] - (ix as f64);
        let f = (
            (1.0 - dx) * rate_table::TABLE[0][ix]
            + dx * rate_table::TABLE[0][ix+1]
        );
        Some((f - rate_table::MIN[1]).exp() * eta)
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

fn rescale(frac: f64, table: &[[f64; 2]; 16]) -> (f64, [[f64; 2]; 15]) {
    let mut output = [[0.0; 2]; 15];
    for i in 0..15 {
        output[i][0] = table[i][0].ln();
        output[i][1] = (-1.0 * (1.0 - table[i][1]).ln()).ln();
    }
    let frac2 = (-1.0 * (1.0 - frac).ln()).ln();
    (frac2, output)
}

/// Obtain harmonic index by inverting frac = cdf(n), where 0 <= frac < 1 and
/// the cdf is tabulated.
#[cfg(not(feature = "explicit-harmonic-summation"))]
fn get_harmonic_index(a: f64, eta: f64, frac: f64) -> i32 {
    if a.ln() <= cdf_table::MIN[0] {
        // first harmonic only
       1
    } else if eta.ln() <= cdf_table::MIN[1] {
        // cdf(n) is independent of eta as eta -> 0
        let ix = ((a.ln() - cdf_table::MIN[0]) / cdf_table::STEP[0]) as usize;
        let dx = (a.ln() - cdf_table::MIN[0]) / cdf_table::STEP[0] - (ix as f64);

        let index = [ix, ix + 1];
        let weight = [1.0 - dx, dx];

        let n: f64 = index.iter()
            .zip(weight.iter())
            .map(|(i, w)| {
                let table = &cdf_table::TABLE[*i];
                let n = if frac <= table[0][1] {
                    0.9
                } else if a < 5.0 {
                    pwmci::Interpolant::new(table).invert(frac).unwrap()
                } else {
                    let (rs_frac, rs_table) = rescale(frac, table);
                    pwmci::Interpolant::new(&rs_table)
                        .extrapolate(true)
                        .invert(rs_frac)
                        .map(f64::exp)
                        .unwrap()
                };
                n * w
            })
            .sum();

        n.ceil() as i32
    } else {
        let ix = ((a.ln() - cdf_table::MIN[0]) / cdf_table::STEP[0]) as usize;
        let iy = ((eta.ln() - cdf_table::MIN[1]) / cdf_table::STEP[1]) as usize;
        let dx = (a.ln() - cdf_table::MIN[0]) / cdf_table::STEP[0] - (ix as f64);
        let dy = (eta.ln() - cdf_table::MIN[1]) / cdf_table::STEP[1] - (iy as f64);

        let index = [
            cdf_table::N_COLS * iy + ix,
            cdf_table::N_COLS * iy + ix + 1,
            cdf_table::N_COLS * (iy + 1) + ix,
            cdf_table::N_COLS * (iy + 1) + (ix + 1),
        ];

        let weight = [
            (1.0 - dx) * (1.0 - dy),
            dx * (1.0 - dy),
            (1.0 - dx) * dy,
            dx * dy,
        ];

        let n_alt: f64 = index.iter()
            .zip(weight.iter())
            .map(|(i, w)| {
                let table = &cdf_table::TABLE[*i];
                let n = if frac <= table[0][1] {
                    0.9
                } else if a < 5.0 {
                    pwmci::Interpolant::new(table).invert(frac).unwrap()
                } else {
                    let (rs_frac, rs_table) = rescale(frac, table);
                    pwmci::Interpolant::new(&rs_table)
                        .extrapolate(true)
                        .invert(rs_frac)
                        .map(f64::exp)
                        .unwrap()
                };
                n * w
            })
            .sum();

        n_alt.ceil() as i32
    }
}

/// Obtain harmonic index by inverting frac = cdf(n), where 0 <= frac < 1 and
/// the cdf is calculated by integrating and summing the double-differential rates.
#[cfg(feature = "explicit-harmonic-summation")]
fn get_harmonic_index(a: f64, eta: f64, frac: f64) -> i32 {
    let nmax = sum_limit(a, eta);
    let target = frac * rate(a, eta).unwrap();
    // println!("Aiming for {:.1}% of total rate...", 100.0 * frac);
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
    n.unwrap_or_else(|| {
        eprintln!("lp::sample failed to obtain a harmonic order: target = {:.3e}% of rate at a = {:.3e}, eta = {:.3e} (n < {}), falling back to {}.", frac, a, eta, nmax, nmax - 1);
        nmax - 1
    })
}

/// Returns a pseudorandomly sampled n (harmonic order), s (lightfront momentum
/// transfer) and theta (azimuthal angle in the ZMF) for a photon emission that
/// occurs at normalized amplitude a and energy parameter eta.
#[allow(non_snake_case, unused_parens)]
pub(super) fn sample<R: Rng>(a: f64, eta: f64, rng: &mut R, fixed_n: Option<i32>) -> (i32, f64, f64, StokesVector) {
    let n = fixed_n.unwrap_or_else(|| {
        let frac = rng.gen::<f64>();
        get_harmonic_index(a, eta, frac) // via lookup of cdf
    });

    let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);
    let theta_max = ThetaBound::for_harmonic(n, a, eta);

    let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
    let smax = sn / (1.0 + sn);

    let max = ceiling_double_diff_partial_rate(a, eta, &mut dj);
    // let (_, max) = partial_rate(n, a, eta);
    // let max = 1.1 * max;

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

    // println!("\t... got s = {:.3e}, theta = {:.3e}", s, theta);

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

    let x = {
        let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
        let wn = s / (sn * (1.0 - s));
        let wn = wn.min(1.0);
        -2.0 * (n as f64) * a * theta.cos() * (wn * (1.0 - wn)).sqrt() / (1.0 + 0.5 * a * a).sqrt()
    };

    let y = a * a * s / (8.0 * eta * (1.0 - s));

    let j = dj.evaluate(x.abs(), y); // n-2, n-1, n, n+1, n+2

    let A = if x > 0.0 {
        [j[2], 0.5 * (j[1] + j[3]), 0.25 * (j[0] + 2.0 * j[2] + j[4])]
        // or x is negative, J_n(-|x|, y) = (-1)^n J_n(|x|, y)
    } else if n % 2 == 0 {
        // j[1] and j[3] change sign
        [j[2], -0.5 * (j[1] + j[3]), 0.25 * (j[0] + 2.0 * j[2] + j[4])]
    } else {
        // j[0], j[2] and j[4] change sign
        [-j[2], 0.5 * (j[1] + j[3]), -0.25 * (j[0] + 2.0 * j[2] + j[4])]
    };

    let pol: StokesVector = {
        let u = (2.0 * (n as f64) * eta * (1.0 - s) / s - (1.0 + 0.5 * a * a)).sqrt();

        let xi_0 = (
            (1.0 - s + 1.0 / (1.0 - s)) * (A[1].powi(2) - A[0] * A[2])
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

    // assert!(pol.dop() <= 1.0);

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

    #[test]
    fn rate_ceiling() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        for _i in 0..100 {
            let a = (0.2_f64.ln() + (20_f64.ln() - 0.2_f64.ln()) * rng.gen::<f64>()).exp();
            let eta = (0.001_f64.ln() + (1.0_f64.ln() - 0.001_f64.ln()) * rng.gen::<f64>()).exp();
            let n_max = sum_limit(a, eta);
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
                let (_, true_max) = partial_rate(*n, a, eta);
                let mut dj = DoubleBessel::at_index(*n, (*n as f64) * consts::SQRT_2, (*n as f64) * 0.5);
                let max = ceiling_double_diff_partial_rate(a, eta, &mut dj);
                let err = (true_max - max) / true_max;
                println!(
                    "a = {:>9.3e}, eta = {:>9.3e}, n = {:>4} => max = {:>9.3e}, predicted = {:>9.3e}, err = {:.2}%",
                    a, eta, n, true_max, max, 100.0 * err,
                );
                assert!(err < 0.0);
            }
        }
    }

    #[test]
    #[ignore]
    fn find_rate_max() {
        let a = 15.0;
        let eta = 1.0;
        let n_max = sum_limit(a, eta);
        let nodes: Vec<f64> = (1..100).map(|i| (i as f64) / 100.0).collect();
        let pts: Vec<_> =
            //(1..n_max)
            //(1..10).chain((10..100).step_by(2)).chain((100..n_max).step_by(5)) // 5
            //(1..10).chain((10..100).step_by(5)).chain((100..n_max).step_by(10)) // 10
            (1..10).chain((10..100).step_by(10)).chain((100..n_max).step_by(50)) // 20
            .map(|n| {
                let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);
                let max_theta = ThetaBound::for_harmonic(n, a, eta);
                let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
                let smax = sn / (1.0 + sn);

                let mut max = 0.0;
                let mut s0 = 0.0;
                let mut theta0 = 0.0;

                for s in nodes.iter().map(|x| x * smax) {
                    for theta in nodes.iter().map(|x| x * max_theta.at(s)) {
                        let val = double_diff_partial_rate(a, eta, s, theta, &mut dj);
                        if val > max {
                            max = val;
                            s0 = s;
                            theta0 = theta;
                        }
                    }
                }

                (n, s0 / smax, theta0, max)
            })
            .collect();

        let filename = format!("output/nlc_lp_max_{}_{}.dat", a, eta);
        let mut file = File::create(&filename).unwrap();
        for (n, s, theta, max) in &pts {
            writeln!(file, "{} {:.6e} {:.6e} {:.6e}", n, s, theta, max).unwrap();
        }
    }

    #[test]
    #[ignore]
    fn integration() {
        let (n, a, eta) = (100, 10.0, 0.1);
        // bounds on s
        let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
        let smax = sn / (1.0 + sn);

        let nodes: Vec<f64> = (1..300).map(|i| (i as f64) / 300.0).collect();
        let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);
        let max_theta = ThetaBound::for_harmonic(n, a, eta);

        let filename = format!("output/nlc_lp_dd_rate_{}_{}_{}.dat", n, a, eta);
        let mut file = File::create(&filename).unwrap();
        let mut max = 0.0;

        for s in nodes.iter().map(|x| x * smax) {
            for theta in nodes.iter().map(|x| x * consts::FRAC_PI_2) {
                let rate = if theta > max_theta.at(s) {
                    0.0
                } else {
                    double_diff_partial_rate(a, eta, s, theta, &mut dj)
                };
                if rate > max {
                    max = rate;
                }
                writeln!(file, "{:.6e} {:.6e} {:.6e}", s, theta, rate).unwrap();
            }
        }

        let (integral, predicted_max) = partial_rate(n, a, eta);
        println!("integral = {:.6e}, max = {:.6e} [{:.6e} with finer resolution]", integral, predicted_max, max);

        let filename = format!("output/nlc_lp_sd_rate_{}_{}_{}.dat", n, a, eta);
        let mut file = File::create(&filename).unwrap();

        let s_peak = sn / (2.0 + sn);
        let pts: Box<dyn Iterator<Item = f64>> = if sn < 1.0 {
            let lower = GAUSS_32_NODES.iter().map(|x| 0.5 * (x + 1.0) * s_peak);
            let upper = GAUSS_32_NODES.iter().map(|x| s_peak + 0.5 * (smax - s_peak) * (x + 1.0));
            Box::new(lower.chain(upper))
        } else {
            let (s0, s1) = (0.0, smax - 2.0 * (smax - s_peak));
            let lower = GAUSS_16_NODES.iter().map(move |x| s0 + 0.5 * (s1 - s0) * (x + 1.0));

            let (s0, s1) = (smax - 2.0 * (smax - s_peak), s_peak);
            let mid = GAUSS_32_NODES.iter().map(move |x| s0 + 0.5 * (s1 - s0) * (x + 1.0));

            let (s0, s1) = (s_peak, smax);
            let upper = GAUSS_32_NODES.iter().map(move |x| s0 + 0.5 * (s1 - s0) * (x + 1.0));

            Box::new(lower.chain(mid).chain(upper))
        };

        for s in pts {
            let theta = max_theta.at(s);
            let (rate, _) = single_diff_partial_rate(a, eta, s, theta, &mut dj);
            writeln!(file, "{:.6e} {:.6e} {:.6e}", s, rate, theta).unwrap();
        }
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
                let a = (0.02_f64.ln() + (20_f64.ln() - 0.02_f64.ln()) * rng.gen::<f64>()).exp();
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
                    let n_max = sum_limit(a, eta);
                    let target = (1..=n_max).map(|n| partial_rate(n, a, eta).0).sum::<f64>();
                    let error = (target - value).abs() / target;
                    println!(
                        "[{:>2}]: a = {:>9.3e}, eta = {:>9.3e}: target = {:>9.3e}, lookup = {:>9.3e}, diff = {:.3e}",
                        rayon::current_thread_index().unwrap_or(0), a, eta, target, value, error,
                    );
                    assert!(error < 1.0e-3);
                }
            })
        });
    }

    #[test]
    fn harmonic_index_sampling() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let pts = [
            (0.946303, 0.105925),
            (2.37700, 0.105925),
            (4.74275, 0.105925),
            //(9.46303, 0.105925),
            (0.946303, 1.0e-4),
            (2.37700, 1.0e-4),
            (4.74275, 1.0e-4),
        ];

        for (a, eta) in &pts {
            println!("At a = {}, eta = {}:", a, eta);
            let nmax = sum_limit(*a, *eta);
            let nmax = nmax.max(19);
            let rates: Vec<f64> = (1..=nmax).map(|n| partial_rate(n, *a, *eta).0).collect();
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
                let n = get_harmonic_index(*a, *eta, frac);
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
    fn partial_spectrum() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);
        let n = 100;
        let a = 10.0;
        let eta = 0.1;

        let rt = std::time::Instant::now();
        let vs: Vec<(i32,f64,f64,_)> = (0..10_000)
            .map(|_n| {
                sample(a, eta, &mut rng, Some(n))
            })
            .collect();
        let rt = rt.elapsed();

        println!("a = {:.3e}, eta = {:.3e}, {} samples takes {:?}", a, eta, vs.len(), rt);
        let filename = format!("output/nlc_lp_partial_spectrum_{}_{}_{}.dat", n, a, eta);
        let mut file = File::create(&filename).unwrap();
        for (_, s, phi, _) in vs {
            writeln!(file, "{:.6e} {:.6e}", s, phi).unwrap();
        }
    }

    #[test]
    #[ignore]
    fn total_spectrum() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);
        let a = 10.0; // 0.946393; // 2.37700; // 4.74275; // 9.46303;
        let eta = 0.1; // 0.105925;

        let rt = std::time::Instant::now();
        let vs: Vec<(i32,f64,f64,_)> = (0..100)
            .map(|_n| {
                sample(a, eta, &mut rng, None)
            })
            .collect();
        let rt = rt.elapsed();

        println!("a = {:.3e}, eta = {:.3e}, {} samples takes {:?}", a, eta, vs.len(), rt);
        let filename = format!("output/nlc_lp_spectrum_{}_{}.dat", a, eta);
        let mut file = File::create(&filename).unwrap();
        for (n, s, phi, _) in vs {
            writeln!(file, "{} {:.6e} {:.6e}", n, s, phi).unwrap();
        }
    }

    #[test]
    #[ignore]
    fn harmonic_limit() {
        let max_error = 1.0e-3;

        let filename = format!("output/nlc_lp_rates.dat");
        let mut file = File::create(&filename).unwrap();

        let filename = format!("output/nlc_harmonic_limits.dat");
        let mut file2 = File::create(&filename).unwrap();

        for a in [0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0].iter() {
            for eta in [2.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.001].iter() {
                let n_min = 1;
                let mut n = n_min;
                let mut total = 0.0;
                let mut step = 1;
                let mut prev = 0.0;

                loop {
                    let (rate, _) = partial_rate(n, *a, *eta);
                    total += 0.5 * (rate + prev) * (step as f64);

                    // linear extrapolation: estimate fraction left to sum
                    let frac_remaining = 0.5 * (step as f64) * rate * rate / (total * (prev - rate));
                    let frac_remaining = if frac_remaining < 0.0 || frac_remaining > 1.0 {
                        1.0
                    } else {
                        frac_remaining
                    };

                    println!("n = {:>4}, rate = {:>9.3e}, total = {:>9.3e}, estd frac left = {:>9.3}%", n, rate, total, 100.0 * frac_remaining);
                    writeln!(file, "{:.6e} {:.6e} {} {:.6e} {:.6e}", a, eta, n, rate, total).unwrap();

                    if frac_remaining < max_error {
                        writeln!(file2, "{:.6e} {:.6e} {}", a, eta, n).unwrap();
                        break;
                    }

                    if n - n_min > 10_000 {
                        step = 1500;
                    } else if n - n_min > 3000 {
                        step = 500;
                    } else if n - n_min > 1000 {
                        step = 150;
                    } else if n - n_min > 300 {
                        step = 50;
                    } else if n - n_min > 100 {
                        step = 15;
                    } else if n - n_min > 30 {
                        step = 5;
                    } else if n - n_min > 10 {
                        step = 2;
                    }

                    prev = rate;
                    n += step;
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn create_rate_table() {
        const LOW_ETA_LIMIT: f64 = 0.001;
        const LOW_A_LIMIT: f64 = 0.02;
        // 20, 20, 60, 60
        const A_DENSITY: usize = 20; // points per order of magnitude
        const ETA_DENSITY: usize = 20;
        const N_COLS: usize = 61; // pts in a0 direction
        const N_ROWS: usize = 68; // pts in eta direction
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
                let n_max = sum_limit(a, eta);
                pts.push((i, j, a, eta, n_max));
            }
        }

        let pts: Vec<(usize, usize, f64, [[f64; 2]; 16])> = pts.into_iter()
            .map(|(i, j, a, eta, n_max)| {
                let mut rates: Vec<[f64; 2]> = pool.install(|| {
                    (1..=n_max).into_par_iter()
                        .map(|n| [n as f64, partial_rate(n, a, eta).0])
                        .collect()
                });

                let mut cumsum = 0.0;
                for [_, pr] in rates.iter_mut() {
                    cumsum += *pr;
                    *pr = cumsum;
                }

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

                println!("LP NLC: eta = {:.3e}, a = {:.3e}, ln(rate) = {:.6e}", eta, a, rate.ln());
                (i, j, rate, cdf)
            })
            .collect();

        for (i, j, rate, _) in &pts {
            table[*i][*j] = *rate;
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

        let mut file = File::create("output/cdf_table.rs").unwrap();
        //writeln!(file, "pub const LENGTH: usize = {};", N_COLS * N_ROWS).unwrap();
        writeln!(file, "pub const N_COLS: usize = {};", N_COLS).unwrap();
        //writeln!(file, "pub const N_ROWS: usize = {};", N_ROWS).unwrap();
        writeln!(file, "pub const MIN: [f64; 2] = [{:.12e}, {:.12e}];", LOW_A_LIMIT.ln(), LOW_ETA_LIMIT.ln()).unwrap();
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
